## Overview

This project analyzes toxicity classification and counterspeech generation using GPT-2 medium through value vector analysis.

## Files and Execution Order

### 1. `binary_classification_train.ipynb`
Trains a binary classifier for toxicity detection:
- Base model: GPT-2 Medium
- Dataset: Jigsaw Toxic Comment Classification Dataset
- Output: Saves finetuned model as `best_model_state.bin`
- Metrics tracked: Accuracy, F1 Score, Loss

### 2. `counterspeech_generation.ipynb`
Generates non-toxic responses to toxic comments:
- Models used: Finetuned GPT-2 medium
- Dataset: Counter Speech Dataset for Toxic Language 
- Output: Counter-speech generation model saved in `saved_models` directory with naming format `LR_{Learning_Rate}_BS_{Batch_Size}_E_{Epochs}`

### 3. `value_vector_analysis.ipynb`
Initial value vector analysis:
- Models analyzed: Base GPT-2 Medium and fine-tuned classifier (`best_model_state.bin`) [Path to binary classification model can be modified if needed]
- Extracts and compares value vectors

### 4. `analysis-1.ipynb`
Deep dive into value vectors:
- Performs SVD on value vectors
- Maps vectors to vocabulary space
- Compares base vs finetuned model differences
- Uses models from steps 1 & 2

### 5. `analysis-2.ipynb`
Advanced analysis:
- Key-value vector interactions
- Cross-model vector alignments 
- Token-level impact analysis
- Uses models and vectors from previous steps

## Requirements

Required libraries:
```bash
pip install torch==1.9.0 transformers==4.11.3 transformer_lens==0.3.0 nltk==3.6.3 matplotlib==3.4.3 numpy==1.21.2
```

## Models & Data
- Base Model: GPT-2 Medium (345M parameters)
- Training Data: Jigsaw Toxic Comment Dataset (~561K samples)
- Validation Data: Hold-out toxic comments (20%)
- Counter Speech Data: CONAN dataset (~15k paired examples)

## Notes
- Models require GPU with 12GB+ memory
- Full training takes 4-6 hours on P100
- Save all best model checkpoints
