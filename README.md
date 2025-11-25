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
├── Report.pdf           # Full project report
├── .gitignore
├── LICENSE
└── README.md
```

---

## Method Overview

### 1. Zero-shot Baseline (Qwen2.5-7B)
We evaluate the model without any fine-tuning using:
- the question  
- retrieved evidence sentences  
- an instruction-style prompt  
- a numeric extraction helper  

This establishes a baseline for comparison.

### 2. LoRA Fine-Tuning
We apply **LoRA (Low-Rank Adaptation)** to Qwen2.5-7B-Instruct:
- 4-bit quantization  
- rank = 16  
- LoRA applied to attention projection layers  
- trained in bf16 on Google Colab A100  

Two settings are provided:
- **FinQA_v0**: ~2,000 training samples  
- **FinQA_v1**: full training set (6,251 samples)  

The model is trained to output the final numerical answer conditioned on question + evidence.

---

## Results

Accuracy on a 50-example dev slice:

| Model | Train Size | Epochs | Max Length | Accuracy |
|-------|------------|--------|------------|----------|
| Zero-shot baseline | – | – | 768 | 0.15 |
| FinQA_v0 (LoRA) | ~2k subset | 1 | 512 | 0.44 |
| FinQA_v1 (LoRA) | full train | 2 | 768 | 0.50 |

LoRA significantly improves accuracy over the zero-shot baseline.

---

## How to Use

1. Download the FinQA dataset JSON files.  
2. Place them under the `dataset/` directory.  
3. Open either `FinQA_v0.ipynb` or `FinQA_v1.ipynb` in Google Colab.  
4. Run the notebook to reproduce fine-tuning and evaluation.

---

## References

**FinQA Dataset:**  
Chen et al., *FinQA: A Dataset of Numerical Reasoning over Financial Data*, EMNLP 2021.

**Model:**  
Qwen2.5-7B-Instruct  
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

