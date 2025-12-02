# ProtTrans Scripts for Protein Fitness Prediction

This directory contains a complete suite of scripts for protein fitness prediction using ProtTrans models (ProtT5-XL and ProtAlbert).

## Overview

The scripts follow a similar pattern to the ESM implementation and provide:
1. **Embedding extraction** from protein sequences
2. **Model training** with cross-validation
3. **Fitness prediction** using trained models
4. **Log-likelihood calculation** for mutations (using ProtBERT)

## Available Scripts

### 1. prottrans_embedding.py
Extracts ProtTrans embeddings from a CSV file containing protein sequences.

**Usage:**
```bash
python prottrans_embedding.py -i data/proteins.csv -m ProtT5-XL
```

**Arguments:**
- `-i, --input_path`: Path to CSV file containing protein sequences (required)
- `-m, --model`: Model name (default: ProtT5-XL, choices: ProtT5-XL, ProtAlbert)
- `-s, --seq_col`: Column name containing protein sequences (default: seq)
- `-d, --id_column`: Column name containing sequence IDs (optional)
- `-b, --batch_size`: Batch size for processing (default: 100)
- `--device`: Device to run on (default: cuda, choices: cuda, cpu)

**Output:**
- `sequences.fasta`: FASTA file with unique sequences
- `{model_name}/{model_name}.npy`: Numpy array of embeddings

**Example:**
```bash
# Extract embeddings using ProtT5-XL
python prottrans_embedding.py -i data/proteins.csv -m ProtT5-XL

# Use ProtAlbert with custom sequence column
python prottrans_embedding.py -i data/proteins.csv -m ProtAlbert -s sequence

# Use CPU instead of GPU
python prottrans_embedding.py -i data/proteins.csv --device cpu
```

---

### 2. prottrans_train_fitness.py
Trains regression models on ProtTrans embeddings for protein fitness prediction.

**Usage:**
```bash
python prottrans_train_fitness.py -i data/proteins -o results -b ProtT5-XL
```

**Arguments:**
- `-i, --input_dir`: Input directory containing data.csv and embeddings (required)
- `-o, --output_dir`: Output directory for models and results (required)
- `-b, --backbone_model`: ProtTrans model name (default: ProtT5-XL)
- `-m, --head_model`: Regression model type (default: svr)
- `-n, --n_components`: Number of PCA components (default: 60)
- `-s, --seed`: Random seed for reproducibility (default: 42)
- `--target_col`: Target column name in CSV (default: log_fitness)
- `--no_final_model`: Skip training final model (only do CV)

**Available head models:**
- svr (default)
- random_forest
- knn
- gbdt
- ridge
- lasso
- elastic_net
- mlp
- xgboost

**Output:**
- `cv_folds/`: Cross-validation results for each fold
- `final_model/`: Final model trained on all data
  - `head_model_{model_type}.joblib`: Trained regression model
  - `pca_model.joblib`: PCA transformer
- `training_summary.csv`: Summary of training results
- `{backbone}_{head}_cv_results.csv`: Detailed CV results

**Example:**
```bash
# Train with default settings
python prottrans_train_fitness.py -i data/proteins -o results

# Use ProtAlbert with random forest
python prottrans_train_fitness.py -i data/proteins -o results -b ProtAlbert -m random_forest

# Custom PCA components
python prottrans_train_fitness.py -i data/proteins -o results -n 100
```

---

### 3. prottrans_predict_fitness.py
Uses a pre-trained model to predict fitness values for new protein sequences.

**Usage:**
```bash
python prottrans_predict_fitness.py -i data.csv -m results/final_model -b ProtT5-XL
```

**Arguments:**
- `-i, --input_csv`: Path to CSV file containing protein sequences (required)
- `-m, --model_dir`: Directory containing trained model (required)
- `-b, --backbone_model`: ProtTrans model name (default: ProtT5-XL)
- `-s, --seq_col`: Column name containing protein sequences (default: seq)
- `-o, --output_csv`: Output CSV file path (optional)
- `--output_suffix`: Suffix for prediction column (default: _pred)

**Output:**
- CSV file with original data plus `fitness_pred` column

**Example:**
```bash
# Predict fitness for new sequences
python prottrans_predict_fitness.py -i data.csv -m results/final_model

# Use ProtAlbert model
python prottrans_predict_fitness.py -i data.csv -m results/final_model -b ProtAlbert

# Custom output file
python prottrans_predict_fitness.py -i data.csv -m results/final_model -o predictions.csv
```

---

### 4. prottrans_llh.py
Calculates pseudo-log-likelihood values for protein mutations using ProtBERT.

**Important Note:** ProtT5-XL and ProtAlbert are encoder-only models and do not support direct log-likelihood calculation. This script uses ProtBERT, which has masked language modeling capabilities.

**Usage:**
```bash
python prottrans_llh.py -i data.csv -w wt.fasta
```

**Arguments:**
- `-i, --input_csv`: Path to CSV file containing mutations (required)
- `-w, --wt_fasta`: Path to wild-type FASTA file (required)
- `-o, --output_csv`: Output CSV file path (optional)
- `--output_col`: Name for output LLH column (default: protbert_llh)
- `--n_proc`: Number of processes for parallel computation (optional)
- `--device`: Device to use (default: cuda, choices: cuda, cpu)
- `--fitness_col`: Column name containing fitness values for correlation (optional)

**Output:**
- CSV file with original data plus log-likelihood column
- Cached mutation probabilities file: `{wt_name}_mut_probs_ProtBert.csv`

**Example:**
```bash
# Calculate log-likelihood using ProtBERT
python prottrans_llh.py -i data.csv -w wt.fasta

# Custom output file
python prottrans_llh.py -i data.csv -w wt.fasta -o results.csv

# Use CPU instead of GPU
python prottrans_llh.py -i data.csv -w wt.fasta --device cpu
```

---

## Complete Workflow Examples

### Example 1: Fitness Prediction Pipeline
```bash
# Step 1: Extract embeddings
python prottrans_embedding.py -i data/proteins.csv -m ProtT5-XL

# Step 2: Train model with cross-validation
python prottrans_train_fitness.py -i data -o results -b ProtT5-XL

# Step 3: Predict fitness for new sequences
python prottrans_predict_fitness.py -i new_data.csv -m results/final_model -b ProtT5-XL
```

### Example 2: Log-likelihood Analysis
```bash
# Calculate log-likelihood for mutations
python prottrans_llh.py -i variants.csv -w wildtype.fasta
```

### Example 3: Model Comparison
```bash
# Extract embeddings with both models
python prottrans_embedding.py -i data.csv -m ProtT5-XL
python prottrans_embedding.py -i data.csv -m ProtAlbert

# Train models
python prottrans_train_fitness.py -i data -o results_t5 -b ProtT5-XL
python prottrans_train_fitness.py -i data -o results_albert -b ProtAlbert
```

---

## Input File Formats

### CSV File (for embeddings and training)
```csv
seq,log_fitness
MKLLVVV...,2.5
MLLLVVV...,1.8
```

Required columns:
- `seq`: Protein sequence
- `log_fitness`: Target fitness value (for training)

### FASTA File (for log-likelihood)
```
>wildtype
MKLLVVVLLLLVVVVVVVVVV
```

---

## Requirements

- Python 3.7+
- PyTorch
- transformers (for ProtT5-XL, ProtAlbert, ProtBERT)
- scikit-learn
- pandas
- numpy
- biopython
- loguru
- tqdm
- scipy

Install requirements:
```bash
pip install torch transformers scikit-learn pandas numpy biopython loguru tqdm scipy
```

---

## Key Differences from ESM Implementation

1. **Model Loading**: Uses Hugging Face transformers instead of ESM library
2. **Embedding Format**: Saves as single `.npy` file instead of individual `.pt` files
3. **Log-likelihood**: Uses ProtBERT for MLM (ProtT5-XL/ProtAlbert don't support LLH)
4. **Preprocessing**: Requires space-separated sequences and replaces uncommon amino acids

---

## Comparison with fitness_pred/ Directory

The existing `fitness_pred/` directory contains:
- `protrans_emb.py`: Similar embedding extraction (can be replaced)
- `train.py`: Training script with CV support (can be replaced)

The new scripts provide:
- Better documentation and error handling
- Consistent interface with ESM scripts
- Separate prediction script
- Log-likelihood calculation support
- More comprehensive CLI arguments

---

## Notes

1. **GPU Memory**: ProtT5-XL requires significant GPU memory. Use smaller batch sizes or CPU if needed.
2. **Caching**: Log-likelihood calculations cache mutation probabilities for faster subsequent runs.
3. **Embeddings**: Pre-extract embeddings once and reuse for multiple training runs.
4. **Cross-validation**: 5-fold CV is performed by default for robust evaluation.

---

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
python prottrans_embedding.py -i data.csv -b 50

# Use CPU
python prottrans_embedding.py -i data.csv --device cpu
```

### Missing Embeddings
```bash
# Make sure to run embedding extraction first
python prottrans_embedding.py -i data.csv -m ProtT5-XL

# Check that embeddings exist
ls data/ProtT5-XL/ProtT5-XL.npy
```

### Model Compatibility
Ensure the same backbone model is used for:
1. Embedding extraction
2. Training
3. Prediction

---

## Citation

If you use these scripts, please cite the ProtTrans paper:
```
Elnaggar, A., Heinzinger, M., Dallago, C. et al.
ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning.
IEEE Trans Pattern Anal Mach Intell. 2022.
```
