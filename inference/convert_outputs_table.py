import re
import json
import itertools
from pycm import *
from collections import Counter
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np

def parse_row(input_text):

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

def format_json(json_file):
	"""
		input : list de json avec des textes au format SDOH structurés "<Entite1> texte [relation1] texte AND <Entite1> texte [relation1] texte"
	"""
	return [parse_row(l) for l in json_file.split(" €AND ")]


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
		i['pred_format'] = format_json(i['sdoh_generated'])
		entities_p = get_labels(i['pred_format'])
		i['pred_labels'] = entities_p
		final_json_file.append(i)

	return final_json_file


def func_final_labels(labels):
    liste_maritalstatus = ["MaritalStatus_Single", "MaritalStatus_Divorced", "MaritalStatus_InRelationship", "MaritalStatus_Widowed"]
    liste_employment = ["Employment_Pensioner", "Employment_Student", "Employment_Unemployed", "Employment_Working", "Employment_Other"]
    liste_alcool = ["Alcohol_StatusTime:current", "Alcohol_StatusTime:past", "Alcohol_StatusTime:none"]
    liste_tabagisme = ["Tobacco_StatusTime:current", "Tobacco_StatusTime:past", "Tobacco_StatusTime:none"]
    liste_drogue = ["Drug_StatusTime:current", "Drug_StatusTime:past", "Drug_StatusTime:none"]

    if len(list(set([j for j in labels if j in liste_maritalstatus]))) > 1:
        if ('MaritalStatus_InRelationship' in labels) and ('MaritalStatus_Widowed' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'MaritalStatus_InRelationship']
        if ('MaritalStatus_InRelationship' in labels) and ('MaritalStatus_Single' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'MaritalStatus_InRelationship']
        if ('MaritalStatus_InRelationship' in labels) and ('MaritalStatus_Divorced' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'MaritalStatus_InRelationship']
        if ('MaritalStatus_Single' in labels) and ('MaritalStatus_Divorced' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'MaritalStatus_Single']
        if ('MaritalStatus_Single' in labels) and ('MaritalStatus_Widowed' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'MaritalStatus_Single']
        if ('MaritalStatus_Divorced' in labels) and ('MaritalStatus_Widowed' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'MaritalStatus_Divorced']
            
    if len(list(set([j for j in labels if j in liste_tabagisme]))) > 1:
        tabagisme_doc = list(set([j for j in labels if j in liste_tabagisme]))
        if 'Tobacco_StatusTime:past' in tabagisme_doc:
            labels = [j for k,j in enumerate(labels) if j != 'Tobacco_StatusTime:none' and j != 'Tobacco_StatusTime:current']
        elif 'Tobacco_StatusTime:current' in tabagisme_doc:
            labels = [j for k,j in enumerate(labels) if j != 'Tobacco_StatusTime:none']

    if len(list(set([j for j in labels if j in liste_alcool]))) > 1:
        alcool_doc = list(set([j for j in labels if j in liste_alcool]))
        if 'Alcohol_StatusTime:past' in alcool_doc:
            labels = [j for k,j in enumerate(labels) if j != 'Alcohol_StatusTime:none' and j != 'Alcohol_StatusTime:current']
        elif 'Alcohol_StatusTime:current' in alcool_doc:
            labels = [j for k,j in enumerate(labels) if j != 'Alcohol_StatusTime:none']

    if len(list(set([j for j in labels if j in liste_drogue]))) > 1:
        drogue_doc = list(set([j for j in labels if j in liste_drogue]))
        if 'Drug_StatusTime:current' in drogue_doc:
            labels = [j for k,j in enumerate(labels) if j != 'Drug_StatusTime:none' and j != 'Drug_StatusTime:past']
        elif 'Drug_StatusTime:past' in drogue_doc:
            labels = [j for k,j in enumerate(labels) if j != 'Drug_StatusTime:none']

    if len(list(set([j for j in labels if j in liste_employment]))) > 1:
        if ('Employment_Working' in labels) and ('Employment_Student' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'Employment_Student']
        if ('Employment_Working' in labels) and ('Employment_Pensioner' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'Employment_Pensioner']
        if ('Employment_Working' in labels) and ('Employment_Other' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'Employment_Other']
        if ('Employment_Working' in labels) and ('Employment_Unemployed' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'Employment_Working' and j != 'Employment_Unemployed']
        if ('Employment_Pensioner' in labels) and ('Employment_Other' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'Employment_Other']
        if ('Employment_Pensioner' in labels) and ('Employment_Unemployed' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'Employment_Unemployed']
        if ('Employment_Student' in labels) and ('Employment_Unemployed' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'Employment_Unemployed']
        if ('Employment_Other' in labels) and ('Employment_Unemployed' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'Employment_Other']
        if ('Employment_Other' in labels) and ('Employment_Student' in labels):
            labels = [j for k,j in enumerate(labels) if j != 'Employment_Other']

    if 'Living_Alone' in labels and "Living_WithOthers" in labels:
        labels = [j for k,j in enumerate(labels) if j != 'Living_WithOthers']

    if 'Descendants_Yes' in labels and "Descendants_No" in labels:
        labels = [j for k,j in enumerate(labels) if j != 'Descendants_Yes']

    if 'Housing_Yes' in labels and "Housing_No" in labels:
        labels = [j for k,j in enumerate(labels) if j != 'Housing_No']

    if 'PhysicalActivity_Yes' in labels and "PhysicalActivity_No" in labels:
        labels = [j for k,j in enumerate(labels) if j != 'PhysicalActivity_No' and j != 'PhysicalActivity_Yes']

    if len(list(set([j for j in labels if j in liste_employment]))) == 0 and ('Last_job' in labels or 'Job' in labels):
        labels.append('Employment_Working')
        
    return labels

dict_maritalstatus = {"MaritalStatus_Single": "0", "MaritalStatus_InRelationship": "1", 
                      "MaritalStatus_Divorced": "2", "MaritalStatus_Widowed": "3"}
dict_employment = {"Employment_Unemployed": "0", "Employment_Working": "1", 
                      "Employment_Student": "2", "Employment_Pensioner": "3", "Employment_Other": "4"}
dict_alcool = {"Alcohol_StatusTime:none": "0", "Alcohol_StatusTime:current": "1", 
                      "Alcohol_StatusTime:past": "2"}
dict_tabac = {"Tobacco_StatusTime:none": "0", "Tobacco_StatusTime:current": "1", 
                      "Tobacco_StatusTime:past": "2"}
dict_drogue = {"Drug_StatusTime:none": "0", "Drug_StatusTime:current": "1", 
                      "Drug_StatusTime:past": "2"}
dict_living = {"Living_Alone": "0", "Living_WithOthers": "1"}
dict_descendants = {"Descendants_No": "0", "Descendants_Yes": "1"}
dict_housing = {"Housing_No": "0", "Housing_Yes": "1"}
dict_physicalactivity = {"PhysicalActivity_No": "0", "PhysicalActivity_Yes": "1"}

liste_maritalstatus = ["MaritalStatus_Single", "MaritalStatus_Divorced", "MaritalStatus_InRelationship", "MaritalStatus_Widowed"]
liste_employment = ["Employment_Pensioner", "Employment_Student", "Employment_Unemployed", "Employment_Working", "Employment_Other"]
liste_alcool = ["Alcohol_StatusTime:current", "Alcohol_StatusTime:past", "Alcohol_StatusTime:none"]
liste_tabagisme = ["Tobacco_StatusTime:current", "Tobacco_StatusTime:past", "Tobacco_StatusTime:none"]
liste_drogue = ["Drug_StatusTime:current", "Drug_StatusTime:past", "Drug_StatusTime:none"]
liste_living = ["Living_Alone", "Living_WithOthers"]
liste_descendants = ["Descendants_Yes", "Descendants_No"]
liste_housing = ["Housing_Yes", "Housing_No"]
liste_physicalactivity = ["PhysicalActivity_Yes", "PhysicalActivity_No"]

jsonl_file_path_gold = "../../data/inference_pathos/data_inference_tuber_all_generated_flan-t5-large_fast.json"

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


final_list_doc = []

for i in json_file_labels:
    
    labels_b = func_final_labels(i['pred_labels'])
    final_labels = list(set(labels_b))
    i['final_labels'] = final_labels

    final_list_doc.append(i)



final_dict_output = []
for value in final_list_doc:

    dict_label_fin_var = {
        "marital_status": [l for l in value['final_labels'] if l in liste_maritalstatus],
        "employment": [l for l in value['final_labels'] if l in liste_employment],
        "tabac": [l for l in value['final_labels'] if l in liste_tabagisme],
        "alcool": [l for l in value['final_labels'] if l in liste_alcool],
        "drogue": [l for l in value['final_labels'] if l in liste_drogue],
        "living": [l for l in value['final_labels'] if l in liste_living],
        "descendants": [l for l in value['final_labels'] if l in liste_descendants],
        "housing": [l for l in value['final_labels'] if l in liste_housing],
        "physical_activity": [l for l in value['final_labels'] if l in liste_physicalactivity],
    }

    if len([l for l in value['final_labels'] if l in liste_maritalstatus]) > 1: print("marital", value['final_labels'])
    if len([l for l in value['final_labels'] if l in liste_employment]) > 1: print("employment", value['final_labels']) 
    if len([l for l in value['final_labels'] if l in liste_alcool]) > 1: print("alcool", value['final_labels']) 
    if len([l for l in value['final_labels'] if l in liste_tabagisme]) > 1: print("tabac", value['final_labels']) 
    if len([l for l in value['final_labels'] if l in liste_drogue]) > 1: print("drogue", value['final_labels']) 
    if len([l for l in value['final_labels'] if l in liste_living]) > 1: print("living", value['final_labels']) 
    if len([l for l in value['final_labels'] if l in liste_descendants]) > 1: print("descendants", value['final_labels']) 
    if len([l for l in value['final_labels'] if l in liste_housing]) > 1: print("housing", value['final_labels']) 
    if len([l for l in value['final_labels'] if l in liste_physicalactivity]) > 1: print("physical_activity", value['final_labels'])

    final_dict_output.append({
        "id_pat": value['ID_PAT'],
        "id_sej": value['ID_SEJ'],
        "marital_status": dict_maritalstatus[dict_label_fin_var["marital_status"][0]] if len(dict_label_fin_var["marital_status"]) > 0 else 'NA',
        "employment": dict_employment[dict_label_fin_var["employment"][0]] if len(dict_label_fin_var["employment"]) > 0 else 'NA',
        "tabac": dict_tabac[dict_label_fin_var["tabac"][0]] if len(dict_label_fin_var["tabac"]) > 0 else 'NA',
        "alcool": dict_alcool[dict_label_fin_var["alcool"][0]] if len(dict_label_fin_var["alcool"]) > 0 else 'NA',
        "drogue": dict_drogue[dict_label_fin_var["drogue"][0]] if len(dict_label_fin_var["drogue"]) > 0 else 'NA',
        "living": dict_living[dict_label_fin_var["living"][0]] if len(dict_label_fin_var["living"]) > 0 else 'NA',
        "descendants": dict_descendants[dict_label_fin_var["descendants"][0]] if len(dict_label_fin_var["descendants"]) > 0 else 'NA',
        "housing": dict_housing[dict_label_fin_var["housing"][0]] if len(dict_label_fin_var["housing"]) > 0 else 'NA',
        "physical_activity": dict_physicalactivity[dict_label_fin_var["physical_activity"][0]] if len(dict_label_fin_var["physical_activity"]) > 0 else 'NA',
    })


df_sdoh = pd.DataFrame(final_dict_output)
df_sdoh.replace('NA', np.nan, inplace=True)
df_sdoh.to_csv('../../data/inference_pathos/all_predictions_tuber.csv', index=False)


#    print(key, unique_labels)
#    print("#"*10)
