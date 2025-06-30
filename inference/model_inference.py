import json
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load JSON data into Hugging Face Dataset
dataset = load_dataset('json', data_files='data_inference.json', split='train')

# Load Model and Tokenizer
model_name = "./flan-t5-large-finetuned/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Tokenize Inputs and Perform Inference in Batches
def generate_outputs(example):
    inputs = tokenizer(example['fr'], truncation=True, padding='max_length', max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Perform batch inference
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_length=512)
    # Decode generated text
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Add the generated text as a new column
    example['sdoh_generated'] = decoded_outputs
    return example

# Apply the function in batches
batch_size = 8  # Adjust based on memory
processed_dataset = dataset.map(generate_outputs, batched=True, batch_size=batch_size)

# Save the outputs to JSON
output_file = f"data_inference_flan-t5-large.json"
processed_dataset.to_json(output_file)