import re
import json
import itertools
from pycm import *
from collections import Counter
from sklearn.metrics import classification_report

def parse_row(input_text):
    """
    Parses a single structured SDOH (Social Determinants of Health) text entry into its components.

    The input text is expected to follow this format:
        «Entity» description [relation] related_text

    Example:
        «Tobacco» Tabagisme [StatusTime:current] actif

    Parameters:
        input_text (str): A single string containing an SDOH entity and one or more relations.

    Returns:
        dict: A dictionary containing:
            - "entity" (tuple or None): A tuple of the entity name and its description,
                                        or None if no entity is found.
            - "relations" (list): A list of dictionaries, each containing:
                - "relation" (str): The relation type, including an optional attribute (e.g., StatusTime).
                - "text" (str): The associated text.
    """
    if input_text != "":
        # Extract entities (between « and »)
        entity_pattern = r"«(.*?)» (.*?)(?= \[|$)"
        entity_match = re.search(entity_pattern, input_text)
        entities = (entity_match.group(1), entity_match.group(2).strip()) if entity_match else None

        # Extract relations (between [] and additional attributes if applicable)
        relation_pattern = r"\[(.*?)\](?:\[(.*?)\])? (.*?)(?=\[|$)"
        relations = []

        for match in re.finditer(relation_pattern, input_text):
            relation = match.group(1)
            attribute = match.group(2)  # This captures StatusTime attributes like current, past, etc.
            text = match.group(3).strip()

            if attribute:  # If an attribute exists (like for StatusTime)
                relations.append({"relation": relation, "attribute": attribute, "text": text})
            else:
                relations.append({"relation": relation, "text": text})

        # Combine entities and relations into final structure
        structured_output = {
            "entity": entities,
            "relations": relations
        }
    else:
        structured_output = {
            "entity": None,
            "relations": []
        }

    return structured_output

def format_struct_output(text):
    """
    Parses a list of structured SDOH (Social Determinants of Health) text entries from a single string.

    Parameters:
        text (str): A string containing multiple SDOH-formatted entries,
                         separated by " €AND ". Each entry follows the pattern:
                         "<Entity1> text [relation1] text AND <Entity1> text [relation1] text".

    Returns:
        list: A list of parsed entries, each processed by the `parse_row` function.

    """
    return [parse_row(l) for l in text.split(" €AND ")]

def get_events_counts(events):

    counter = Counter()
    for event in events:
        entity = event["entity"]

        if event['relations'] == {}:
            arg_type = 'N/A'
            arg_text = 'N/A'
        else:
            for i, key, value in enumerate(event['relations'].items()):

                arg_type = key
                arg_text = value

    return counter

def eval_main_entities(json_file):

    def get_labels(list_dict):

        entities_g = []

        for ent_g in list_dict:
            if ent_g['entity'] == None:
                pass
            elif ent_g['entity'][0] in ["Alcohol", "Drug", "Tobacco"]:
                if "StatusTime:current" in [r['relation'] for r in ent_g['relations']]:
                    entities_g.append(ent_g['entity'][0]+"_StatusTime:current")
                if "StatusTime:past" in [r['relation'] for r in ent_g['relations']]:
                    entities_g.append(ent_g['entity'][0]+"_StatusTime:past")
                if "StatusTime:none" in [r['relation'] for r in ent_g['relations']]:
                    entities_g.append(ent_g['entity'][0]+"_StatusTime:none")
            elif ent_g['entity'][0] in main_entities:
                entities_g.append(ent_g['entity'][0])

        return entities_g

    final_json_file = []

    for i in json_file:
        i['gold_format'] = format_struct_output(i['sdoh'])
        i['pred_format'] = format_struct_output(i['sdoh_generated'])

        entities_p = get_labels(i['pred_format'])
        entities_g = get_labels(i['gold_format'])

        i['gold_labels'] = entities_g
        i['pred_labels'] = entities_p
        final_json_file.append(i)

    return final_json_file

# path to json
jsonl_file_path_gold = "/Volumes/Extreme SSD/SDOH_clean/data/predictions/muscadet_synthetic_generated_flan-t5-large_fast.json"

json_gold = []
with open(jsonl_file_path_gold, 'r', encoding="utf-8") as file_gold:
    for line in file_gold:
        entry = json.loads(line)
        json_gold.append(entry)

main_entities = ["Living_Alone", "Living_WithOthers", "MaritalStatus_Single", "MaritalStatus_Divorced",
 "MaritalStatus_InRelationship", "MaritalStatus_Widowed", "Descendants_Yes", "Descendants_No", "Housing_Yes",
 "Housing_No", "Employment_Pensioner", "Employment_Student", "Employment_Unemployed", "Employment_Working", 
 "Employment_Other", "PhysicalActivity_Yes", "PhysicalActivity_No", "Alcohol_StatusTime:current",
 "Alcohol_StatusTime:past", "Alcohol_StatusTime:none", "Drug_StatusTime:current", "Drug_StatusTime:past",
 "Drug_StatusTime:none", "Tobacco_StatusTime:current", "Tobacco_StatusTime:past", "Tobacco_StatusTime:none"]

json_file_labels = eval_main_entities(json_gold)

res_variable = {}

for entity in main_entities:
    var_labels = [1 if entity in j['gold_labels'] else 0 for j in json_file_labels]
    var_predictions = [1 if entity in j['pred_labels'] else 0 for j in json_file_labels]
    res_variable[entity] = {'labels': var_labels, 'predictions': var_predictions}

results = []

# Compute evaluation metrics for each SDoH category.
for key, value in res_variable.items():
    cm = ConfusionMatrix(value['labels'], value['predictions'],digit=5)
    specificity = cm.TNR
    sensitivity = cm.TPR # ou recall
    precision = cm.PPV
    f1 = cm.F1
    cr = classification_report(value['labels'], value['predictions'], output_dict=True)
    results.append({
        'variable': key,
        'nb_pos': sum(value['labels']),
        'precision': round(cr['1']['precision'], 4) if '1' in cr else 0,
        'recall': round(cr['1']['recall'], 4) if '1' in cr else 0,
        'f1_score': round(cr['1']['f1-score'], 4) if '1' in cr else 0,
    })


with open(f"./results_level_1.json", 'w') as f:
    json.dump(results, f, indent=4)
