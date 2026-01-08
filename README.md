# 🏥 SDOH Extraction from French Clinical Notes using Flan-T5-Large

## 📌 Overview

A sequence-to-sequence (seq2seq) approach for extracting **Social Determinants of Health (SDOH)** from **French clinical notes** using the [Flan-T5-Large](https://huggingface.co/google/flan-t5-large) language model. A total of 13 SDoH categories is included in this study: living condition, marital status, descendants, employment status, occupation, tobacco use, alcohol use, drug use, housing, education, physical activity, income, and ethnicity/country of birth.

Description of the method and results are available in our paper: [Improving Social Determinants of Health Documentation in French EHRs Using Large Language Models](https://arxiv.org/abs/2507.03433)

This repository contains resources to reproduce our implementation: annotated corpora, annotation guidelines, and scripts for inter-annotator agreement computation, and model fine-tuning and evaluation.

## 📁 Repository Structure

```shell
SDOH/
├── annotation 				
    ├── annotation_scheme 			# BRAT annotation config files
    └── inter-annotator_agreement	# IAA computation scripts
├── data
    ├── MUSCADET-Synthetic			# Synthetic annotated dataset in BRAT format
    └── UW-FrenchSDOH				# Translated dataset in BRAT format
├── finetuning						# Fine-tuning script for Flan-T5-Large
└── evaluation						# Inference and evaluation scripts
```

## 📂 Available Datasets

### 📚 Sources

- **MUSCADET-Synthetic**: Synthetic social history sections written by a physician.
- **UW-FrenchSDOH**: French-translated social history sections from the University of Washington SDOH dataset, originally derived from MTSamples.

All datasets are distributed under the CC BY-NC 4.0 licence and stored in BRAT format in `annotation/data/`.

### 📝 Annotation

- **Format:** [BRAT standoff format](http://brat.nlplab.org/standoff.html)
- **Schema:** Entities covering 13 SDoH categories (living condition, marital status, descendants, employment status, occupation, tobacco use, alcohol use, drug use, housing, education, physical activity, income, and ethnicity/country of birth) and 6 relations (Status, Amount, Duration, Frequency, History, Type).
- **Tooling:** See `annotation/annotation_scheme/` for BRAT configuration files.

## 🔧 Requirements

```bash
pip install torch==1.13.1
pip install transformers==4.24.0
pip install datasets==2.13.1
pip install pandas==2.2.0
pip install scikit-learn==1.4.1.post1
pip install python-docx==1.1.2
pip install pycm==4.0
```

## 📖 Citation

```bibtex
@misc{bazoge2025sdoh,
    title={Improving social determinants of health documentation in French electronic health records using large language models}, 
    author={Adrien Bazoge and Pacôme Constant dit Beaufils and Mohammed Hmitouch and Romain Bourcier and Emmanuel Morin and Richard Dufour and Béatrice Daille and Pierre-Antoine Gourraud and Matilde Karakachoff},
    journal={Scientific Reports},
    year={2025},
    month={Nov},
    day={26},
    volume={15},
    number={1},
    pages={45427},
    issn={2045-2322},
    doi={10.1038/s41598-025-29987-z},
    url={https://doi.org/10.1038/s41598-025-29987-z}
}
```
