# 🏥 SDOH Extraction from French Clinical Notes using Flan-T5-Large

## 📊 Inference & Evaluation Scripts

These three scripts let you **generate predictions** and **measure performance** at two levels of granularity.

---

### 🔮 `model_inference.py`

Run the fine-tuned model on a JSON dataset and add `sdoh_generated` predictions.

- **Input**: JSON file with a `fr` field (French clinical note).
- **Output**: JSON file with an extra `sdoh_generated` field. 

```bash
python model_inference.py
```

### 📊 `level_1_eval.py` — Entity-level

Compute Precision / Recall / F1 for each SDoH entity.

- **Input**: JSONL file containing `sdoh` (gold) and `sdoh_generated` (predicted).
- **Output**: Performance metrics for level 1 evaluation: `results_level_1.json`. 

```bash
python level_1_eval.py
```


### 📊 `level_2_eval.py` — Entity + Relation

Evaluate entities and their relations/attributes (exact or overlap span match).

- **Input**: JSONL predictions and BRAT gold annotation folder
- **Output**: Word report (`sdoh_eval_results.docx`) with P/R/F1 per entity–relation–attribute triple.

```bash
python level_2_eval.py
```

