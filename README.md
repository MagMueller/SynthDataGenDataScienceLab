# Synthetic Data Generation Pipeline

This repository contains a comprehensive pipeline for generating and evaluating synthetic data using various state-of-the-art models, including privacy-preserving approaches.

## Project Structure

### Core Pipeline (`notebooks/pipeline/main.py`)
The main pipeline implementation that handles:
- Data loading and preprocessing
- Model training and evaluation 
- Synthetic data generation
- Quality and privacy metrics evaluation
- Results visualization and logging

Supported models:
- CTGAN
- TVAE 
- GReaT (with LoRA fine-tuning)
- GaussianCopula
- CopulaGAN
- Privacy-preserving models:
  - PATE-CTGAN
  - DP-CTGAN

### Exploration & Development
- `notebooks/GReaT_benchmark.ipynb`: Initial experiments with GReaT model
- `notebooks/dp_models/dp_model_benchmark.ipynb`: Development and testing of differential privacy models
- `eval.ipynb`: Evaluation and visualization of model outputs

## Installation

1. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline with different configurations:

```bash
python -m notebooks.pipeline.main --experiment_name default_run
```



## Features
- Modular architecture supporting multiple synthetic data generation models
- Comprehensive evaluation metrics for quality and privacy
- Integration with Weights & Biases for experiment tracking
- Automated visualization of results
- Support for both standard and privacy-preserving models

## Research Paper
For more details on the methodology and results, see our report:
[Synthetic Data Generation](https://www.)

