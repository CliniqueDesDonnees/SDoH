# 🏥 SDOH Extraction from French Clinical Notes using Flan-T5-Large

## 🤝 Inter-Annotator Agreement

To assess annotation quality, inter-annotator agreement (IAA) is computed at both the entity and relation levels.

- Entity-level IAA is calculated using bratiaa, a tool tailored for BRAT-format annotations.
- Relation-level IAA is computed using a custom Python script adapted for multi-annotator setups.

Both approaches output standard evaluation metrics such as precision, recall, and F1-score.

### Entities

To compute IAA for entities, provide a directory structured as follows, where each subfolder contains one annotator's annotations:

```shell
example-project/
├── annotation.conf
├── annotator-1/
    ├── doc-3.ann
    ├── doc-3.txt
    ├── doc-4.ann
    └── doc-4.txt
└── annotator-2/
    ├── doc-3.ann
    ├── doc-3.txt
    ├── doc-4.ann
    └── doc-4.txt
```

The tool will align annotations across annotators by document name and compute overlap-based agreement metrics.

### Relations

Input for computing inter-annotator agreement for relations should be multiple folder `example-project/annotator-1`. The present code is adapted for three annotators and should be modified in the main for other configuration.

For relation agreement, the input format is similar, each annotator’s annotations are stored in a separate folder. The current script is configured for three annotators, but this can be easily adapted in the script’s main function.

⚠️ Note: Update the script if your setup includes a different number of annotators.


```shell
example-project/
├── annotation.conf
├── annotator-1/
    ├── doc-3.ann
    ├── doc-3.txt
    ├── doc-4.ann
    └── doc-4.txt
├── annotator-2/
    ├── doc-3.ann
    ├── doc-3.txt
    ├── doc-4.ann
    └── doc-4.txt
└── annotator-3/
    ├── doc-3.ann
    ├── doc-3.txt
    ├── doc-4.ann
    └── doc-4.txt
```
