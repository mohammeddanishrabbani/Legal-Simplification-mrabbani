# Decode the Law: Legal Text Simplification with Large Language Models

This repository contains the code and dataset for the paper **"Decode the Law: Towards Legal Text Simplification with Large Language Models"** published at [LREC 2026](https://lrec2026.lrec-conf.org/).

**Authors:** Mohammed Danish Rabbani, Subhadeep Roy, Sayantan Mitra, Tulika Saha

**Affiliations:** IIIT Bangalore, University of Technology Nuremberg, Samsung Research Institute Bangalore

## Overview

Legal documents are often verbose and structurally complex, posing significant barriers to public understanding. This project introduces **SIMPLE-LAW**, a curated benchmark dataset of over 6,000 aligned pairs of original and simplified legal passages, and evaluates multiple LLMs on the task of legal text simplification.

### Models Evaluated

| Model | Size | Method |
|-------|------|--------|
| LLaMA 3.2 | 1B, 3B | QLoRA Fine-tuning |
| Mistral 7B | 7B | QLoRA Fine-tuning |
| Qwen 2.5 | 1.5B, 7B | QLoRA Fine-tuning |
| Gemma 3 | 4B | QLoRA Fine-tuning |
| Legal-BERT + GPT-2 | - | Encoder-Decoder Baseline |

### Evaluation Metrics

- **BERTScore** - Semantic similarity
- **ROUGE** - N-gram overlap
- **SARI** - Simplification quality
- **Hallucination Score** - Factual consistency
- **Human Evaluation** - Readability and adequacy

## Repository Structure

```
├── config/                    # Training configuration files
│   ├── accelerate_config.yaml # HuggingFace Accelerate config
│   └── ds_config.json         # DeepSpeed ZeRO Stage 2 config
│
├── src/                       # Shared library modules
│   ├── __init__.py
│   ├── config.py              # Training hyperparameter configuration
│   ├── trainer.py             # SFT trainer wrapper
│   ├── model_manager.py       # Model save/load utilities
│   ├── dataset_handler.py     # Dataset loading and chat template formatting
│   └── evaluation.py          # Evaluation metrics (BERTScore, ROUGE, SARI)
│
├── data/
│   ├── processed/             # Final train/test splits
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── ...
│   ├── raw/                   # Intermediate data (gitignored)
│   └── prompt.csv             # Few-shot prompt examples
│
├── scripts/
│   ├── dataset_creation/      # Notebooks for data generation pipeline
│   ├── training/              # Per-model fine-tuning scripts
│   │   ├── gemma_3_4B.py
│   │   ├── llama_3.2_1B.py
│   │   ├── llama_3.2_3B.py
│   │   ├── mistral_7B.py
│   │   ├── qwen_2.5_1.5B.py
│   │   ├── qwen_2.5_7B.py
│   │   └── legal_bert_gpt2.py
│   └── evaluation/            # Evaluation scripts and results
│       ├── {model_name}/
│       │   ├── eval.py
│       │   ├── script.sh
│       │   └── results/
│       ├── dataset_evaluation.py
│       ├── entailment_evaluation.py
│       └── aggregator.py
│
├── notebooks/                 # Analysis and visualization notebooks
├── results/                   # Aggregated metrics and visualizations
├── legacy/                    # Early prototype notebooks
├── Dockerfile
├── requirements.txt
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 2x GPUs with 24GB+ VRAM)
- Docker (optional, for containerized setup)

### Installation

```bash
# Clone the repository
git clone https://github.com/mohammeddanishrabbani/Legal-Simplification-mrabbani.git
cd Legal-Simplification-mrabbani

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
docker build -t legal-simplification .
docker run --gpus all -it legal-simplification
```

## Usage

### 1. Dataset Creation

The dataset creation notebooks in `scripts/dataset_creation/` walk through:
- Extracting Indian laws from raw sources
- Generating simplified versions using LLMs
- Exploring and splitting the dataset

### 2. Training

Each model has a dedicated training script:

```bash
# Example: Fine-tune LLaMA 3.2 3B
python scripts/training/llama_3.2_3B.py

# Example: Fine-tune Qwen 2.5 7B
python scripts/training/qwen_2.5_7B.py
```

### 3. Evaluation

Run evaluations with in-context learning (0/1/2-shot) and fine-tuned models:

```bash
# Example: Evaluate Gemma 3 4B
bash scripts/evaluation/gemma_3_4B/script.sh

# Aggregate all results
python scripts/evaluation/aggregator.py
```

### 4. Analysis

Use the notebooks in `notebooks/` for:
- Plotting comparison charts (`plot_results.ipynb`)
- Generating LaTeX tables (`latex_generator.ipynb`)
- Human evaluation analysis (`human_eval.ipynb`)
- Inference demos (`inference.ipynb`)

## Dataset

**SIMPLE-LAW** contains 6,230 aligned pairs of complex legal clauses and their simplified versions, sourced from:
- Indian laws (~3,000 samples)
- US laws (~1,500 samples)
- UK laws (~1,500 samples)

## Citation

```bibtex
@inproceedings{rabbani2026decode,
  title={"Decode the Law": Towards Legal Text Simplification with Large Language Models},
  author={Rabbani, Mohammed Danish and Roy, Subhadeep and Mitra, Sayantan and Saha, Tulika},
  booktitle={Proceedings of the 13th Language Resources and Evaluation Conference (LREC 2026)},
  year={2026}
}
```

## License

This project is for research purposes. Please refer to the license file for details.
