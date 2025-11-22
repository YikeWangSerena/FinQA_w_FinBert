# FinQA with LLM + LoRA

This repository explores whether a modern large language model (LLM) with **minimal task-specific architecture** can solve the **FinQA** numerical reasoning task, using **LoRA fine-tuning** on the official FinQA dataset.

The standard FinQA pipeline predicts a symbolic program and then executes it to obtain the final answer. In this project, we keep the same task (given a financial report and a question, recover the final numeric answer), but replace the traditional pipeline with an LLM-based approach.

---

## Repository Structure

```text
.
├── code/
│   ├── FinQA_v0.ipynb   # First LoRA experiment on a 2k training subset
│   └── FinQA_v1.ipynb   # LoRA on the full training set with longer context
├── dataset/
│   ├── train.json       # Official FinQA training split
│   ├── dev.json         # Official FinQA dev split
│   ├── test.json        # Official FinQA public test split
│   └── private_test.json# Official FinQA private test split
├── .gitignore
├── LICENSE
└── README.md
```
