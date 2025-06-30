#!/usr/bin/env python3
"""
Convert BRAT annotations to SDOH‑style JSONL files ready for FLAN‑T5 training.

Usage:
    python brat_to_sdoh.py /path/to/brat/annotations --out-dir ./globales_FlanT5_format
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np  # type: ignore


def remove_prefix(text: str, prefix: str) -> str:
    """Return *text* with *prefix* removed if present."""
    return text[len(prefix) :] if text.startswith(prefix) else text


##########################
#  BRAT parsing
##########################

def parse_brat_file(txt_file: Path, annotation_file_suffixes: List[str] = None, parse_notes: bool = False) -> Dict:

    example = {}
    example["document_id"] = txt_file.with_suffix("").name
    with txt_file.open() as f:
        example["text"] = f.read()

    # If no specific suffixes of the to-be-read annotation files are given - take standard suffixes
    # for event extraction
    if annotation_file_suffixes is None:
        annotation_file_suffixes = [".a1", ".a2", ".ann"]

    if len(annotation_file_suffixes) == 0:
        raise AssertionError(
            "At least one suffix for the to-be-read annotation files should be given!"
        )

    ann_lines = []
    for suffix in annotation_file_suffixes:
        annotation_file = txt_file.with_suffix(suffix)
        if annotation_file.exists():
            with annotation_file.open() as f:
                ann_lines.extend(f.readlines())

    example["text_bound_annotations"] = []
    example["events"] = []
    example["relations"] = []
    example["equivalences"] = []
    example["attributes"] = []
    example["normalizations"] = []

    if parse_notes:
        example["notes"] = []

    for line in ann_lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("T"):  # Text bound
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]
            ann["type"] = fields[1].split()[0]
            ann["offsets"] = []
            span_str = remove_prefix(fields[1], (ann["type"] + " "))
            text = fields[2]
            for span in span_str.split(";"):
                start, end = span.split()
                ann["offsets"].append([int(start), int(end)])

            # Heuristically split text of discontiguous entities into chunks
            ann["text"] = []
            if len(ann["offsets"]) > 1:
                i = 0
                for start, end in ann["offsets"]:
                    chunk_len = end - start
                    ann["text"].append(text[i : chunk_len + i])
                    i += chunk_len
                    while i < len(text) and text[i] == " ":
                        i += 1
            else:
                ann["text"] = [text]

            example["text_bound_annotations"].append(ann)

        elif line.startswith("E"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]

            ann["type"], ann["trigger"] = fields[1].split()[0].split(":")

            ann["arguments"] = []
            for role_ref_id in fields[1].split()[1:]:
                argument = {
                    "role": (role_ref_id.split(":"))[0],
                    "ref_id": (role_ref_id.split(":"))[1],
                }
                ann["arguments"].append(argument)

            example["events"].append(ann)

        elif line.startswith("R"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]
            ann["type"] = fields[1].split()[0]

            ann["head"] = {
                "role": fields[1].split()[1].split(":")[0],
                "ref_id": fields[1].split()[1].split(":")[1],
            }
            ann["tail"] = {
                "role": fields[1].split()[2].split(":")[0],
                "ref_id": fields[1].split()[2].split(":")[1],
            }

            example["relations"].append(ann)

        # '*' seems to be the legacy way to mark equivalences,
        # but I couldn't find any info on the current way
        # this might have to be adapted dependent on the brat version
        # of the annotation
        elif line.startswith("*"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]
            ann["ref_ids"] = fields[1].split()[1:]

            example["equivalences"].append(ann)

        elif line.startswith("A") or line.startswith("M"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]

            info = fields[1].split()
            ann["type"] = info[0]
            ann["ref_id"] = info[1]

            if len(info) > 2:
                ann["value"] = info[2]
            else:
                ann["value"] = ""

            example["attributes"].append(ann)

        elif line.startswith("N"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]
            ann["text"] = fields[2]

            info = fields[1].split()

            ann["type"] = info[0]
            ann["ref_id"] = info[1]
            ann["resource_name"] = info[2].split(":")[0]
            ann["cuid"] = info[2].split(":")[1]
            example["normalizations"].append(ann)

        elif parse_notes and line.startswith("#"):
            ann = {}
            fields = line.split("\t")

            ann["id"] = fields[0]
            ann["text"] = fields[2] if len(fields) == 3 else "<BB_NULL_STR>"

            info = fields[1].split()

            ann["type"] = info[0]
            ann["ref_id"] = info[1]
            example["notes"].append(ann)
    return example


##########################
#  SDOH conversion helpers
##########################

MAIN_ENTITIES: Set[str] = {
    "Living_Alone",
    "Living_WithOthers",
    "MaritalStatus_Single",
    "MaritalStatus_InRelationship",
    "MaritalStatus_Divorced",
    "MaritalStatus_Widowed",
    "Descendants_Yes",
    "Descendants_No",
    "Job",
    "Last_job",
    "Employment_Working",
    "Employment_Unemployed",
    "Employment_Student",
    "Employment_Pensioner",
    "Employment_Other",
    "Housing_Yes",
    "Housing_No",
    "PhysicalActivity_Yes",
    "PhysicalActivity_No",
    "Income",
    "Education",
    "Ethnicity",
}

EVENT_ENTITIES: Set[str] = {"Tobacco", "Alcohol", "Drug"}


def convert_to_sdoh(example: Dict) -> str:
    """Collapse BRAT annotations into a compact SDOH string."""

    ent_by_id = {e["id"]: e for e in example["text_bound_annotations"]}
    attr_by_id = {a["ref_id"]: a for a in example["attributes"]}

    # sort entities by starting offset
    sorted_entities = sorted(
        example["text_bound_annotations"], key=lambda e: e["offsets"][0][0]
    )

    chunks: List[str] = []

    for ent in sorted_entities:
        header = f"«{ent['type']}» {ent['text'][0]}"

        if ent["type"] in MAIN_ENTITIES:
            for rel in example["relations"]:
                if rel["head"]["ref_id"] == ent["id"]:
                    tail = ent_by_id[rel["tail"]["ref_id"]]
                    header += f" [{tail['type']}] {tail['text'][0]}"

        elif ent["type"] in EVENT_ENTITIES:
            extras: List[str] = []
            for rel in example["relations"]:
                if rel["head"]["ref_id"] == ent["id"]:
                    tail = ent_by_id[rel["tail"]["ref_id"]]
                    if tail["type"] == "StatusTime":
                        value = attr_by_id[tail["id"]]["value"]
                        extras.insert(0, f" [{tail['type']}:{value}] {tail['text'][0]}")
                    else:
                        extras.append(f" [{tail['type']}] {tail['text'][0]}")
            header += "".join(extras)

        chunks.append(header)

    return " €AND ".join(chunks)


##########################
#  Dataset builder
##########################

def build_dataset(brat_dir: Path, out_dir: Path) -> None:
    """Convert an entire BRAT directory into train/val/test JSONL files."""
    txt_files = sorted(brat_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {brat_dir}")

    rows = []
    for txt_path in txt_files:
        example = parse_brat_file(txt_path, include_notes=True)
        rows.append(
            {
                "translation": {
                    "document_id": example["document_id"],
                    "fr": example["text"],
                    "sdoh": convert_to_sdoh(example),
                }
            }
        )

    train, val, test = np.split(rows, [int(len(rows) * 0.70), int(len(rows) * 0.80)])

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.json", train)
    _write_jsonl(out_dir / "validation.json", val)
    _write_jsonl(out_dir / "test.json", test)


def _write_jsonl(path: Path, data) -> None:  # type: ignore[override]
    with path.open("w", encoding="utf‑8") as fp:
        for item in data:
            json.dump(item, fp, ensure_ascii=False)
            fp.write("\n")


##########################
#  Entry‑point CLI
##########################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert BRAT → SDOH JSONL for FLAN‑T5 training")
    parser.add_argument("brat_dir", type=Path, help="Directory containing BRAT .txt and annotation files")
    parser.add_argument("--out-dir", type=Path, default=Path("./globales_FlanT5_format"), help="Output directory")

    args = parser.parse_args()
    build_dataset(args.brat_dir, args.out_dir)
