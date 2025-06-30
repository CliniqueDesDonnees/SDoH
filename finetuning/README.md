# 🏥 SDOH Extraction from French Clinical Notes using Flan-T5-Large

## 🧾 Formatting Input

Before fine-tuning, BRAT-annotated data must be converted into a sequence-to-sequence format suitable for Flan-T5.

Use the provided script to convert annotations into JSONL format:

```bash
python format_dataset.py /path/to/brat/annotations --out-dir ./globales_FlanT5_format
```

This will generate `train.json`, `validation.json`, and `test.json` files in the specified output directory.

## 🛠 Fine-Tuning

The script finetunes Flan-T5-Large on the annotated corpora as a translation task (e.g., converting text to structured SDoH outputs).

```bash
python run_train.py \
    --model_name_or_path "../models/flan-t5-large" \
    --tokenizer_name "../models/flan-t5-large" \
    --do_train \
    --do_eval \
    --source_lang "fr" \
    --target_lang "sdoh" \
    --train_file "../data/globales_FlanT5_format/train.json" \
    --validation_file "../data/globales_FlanT5_format/validation.json" \
    --test_file "../data/globales_FlanT5_format/test.json" \
    --output_dir "./output_models/flan-t5-large-finetuned" \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --learning_rate 1e-4 \
    --weight_decay 3e-06 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --seed 42 \
    --source_prefix "translate French to structured social determinants of health: " \
    --logging_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --num_train_epochs 10
```

