# ProtTrans MCP server

ProtTrans MCP server for protein modeling, extracted from the official ProtTrans tutorial.

## Overview
This ProtTrans MCP server provides comprehensive protein structure analysis tools using ProtTrans models. Here we have 4 main scripts for comprehensive protein analysis:


## Installation

```bash
# Create and activate virtual environment
mamba env create -p ./env python=3.10 pip -y
mamba activate ./env
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentencepiece 
pip install pandas loguru scikit-learn xgboost biopython sniffio
pip install --ignore-installed fastmcp
```

## Local usage
### 1. Create ProtTrans Embeddings
```shell
python scripts/prottrans_embedding.py -i example/data.csv -m ProtT5-XL
```

### 2. Train a ProtTrans Fitness Model
```shell
python scripts/prottrans_train_fitness.py -i example/ -o example/fitness_pred -b ProtT5-XL
```

### 3. Predict with a ProtTrans Fitness Model
```shell
python scripts/prottrans_predict_fitness.py -i example/data.csv -m example/fitness_pred/final_model -b ProtT5-XL
```

### 4. Calculate Log-Likelihood Scores
```shell
python scripts/prottrans_llh.py -i example/data.csv -w example/wt.fasta
```

## MCP Usage

### Install ProtTrans MCP Server
```shell
fastmcp install claude-code mcp-servers/prottrans_mcp/src/prottrans_mcp.py --python mcp-servers/prottrans_mcp/env/bin/python
fastmcp install gemini-cli mcp-servers/prottrans_mcp/src/prottrans_mcp.py --python mcp-servers/prottrans_mcp/env/bin/python
```

### Call ProtTrans MCP service
1. Train a ProtTrans-fitness model
```markdown
Can you help train a ProtTrans model for data @examples/case2.1_subtilisin/ and save it to 
@examples/case2.1_subtilisin/prot-t5_fitness using the ProtTrans mcp server with ProtT5-XL model.

Please convert the relative path to absolution path before calling the MCP servers. 
```
2. Inference ProtTrans likelihoods
```markdown
Can you help intererence ProtAlbert likelihood for data @examples/case2.1_subtilisin/data.csv with wild-type  @examples/case2.1_subtilisin/wt.fasta using the prottrans_llh_mcp api in ProtTrans mcp server. Please write the output to @examples/case2.1_subtilisin/data.csv_protalbert_llh.csv

Please convert the relative path to absolution path before calling the MCP servers. 
Please use cuda device 1 where available.
```

### Available MCP Tools

The ProtTrans MCP server provides 4 tools:

#### 1. prottrans_extract_embeddings
Extract ProtTrans embeddings from protein sequences in a CSV file.

**Parameters:**
- `csv_path` (str): Path to CSV file with protein sequences
- `model_name` (str): "ProtT5-XL" or "ProtAlbert" (default: "ProtT5-XL")
- `seq_col` (str): Column name for sequences (default: "seq")
- `id_column` (str, optional): Column for sequence IDs
- `batch_size` (int): Batch size (default: 100)
- `device` (str): "cuda" or "cpu" (default: "cuda")

**Example:**
```python
result = prottrans_extract_embeddings(
    csv_path="data/proteins.csv",
    model_name="ProtT5-XL"
)
```

#### 2. prottrans_calculate_llh
Calculate ProtBERT log-likelihood scores for mutations.

**Parameters:**
- `data_csv` (str): Path to CSV with 'seq' column
- `wt_fasta` (str): Path to wild-type FASTA file
- `n_proc` (int, optional): Number of processes
- `device` (str): "cuda" or "cpu" (default: "cuda")
- `output_col` (str): Output column name (default: "protbert_llh")

**Example:**
```python
result = prottrans_calculate_llh(
    data_csv="data/variants.csv",
    wt_fasta="data/wildtype.fasta"
)
```

#### 3. prottrans_train_fitness_model
Train regression models with 5-fold cross-validation.

**Parameters:**
- `input_dir` (str): Directory with data.csv and embeddings
- `output_dir` (str): Output directory for models
- `backbone_model` (str): "ProtT5-XL" or "ProtAlbert" (default: "ProtT5-XL")
- `head_model` (str): Regression model (default: "svr")
  - Options: svr, random_forest, knn, gbdt, ridge, lasso, elastic_net, mlp, xgboost
- `n_components` (int): PCA components (default: 60)
- `target_col` (str): Target column (default: "log_fitness")
- `seed` (int): Random seed (default: 42)

**Example:**
```python
result = prottrans_train_fitness_model(
    input_dir="data/",
    output_dir="results/",
    head_model="svr"
)
```

#### 4. prottrans_predict_fitness
Predict fitness values using a pre-trained model.

**Parameters:**
- `data_csv` (str): Path to CSV with sequences
- `model_dir` (str): Directory with trained model
- `backbone_model` (str): Model used for embeddings (default: "ProtT5-XL")
- `seq_col` (str): Sequence column (default: "seq")
- `fitness_col` (str, optional): Ground truth column for evaluation
- `output_suffix` (str): Prediction column suffix (default: "_pred")

**Example:**
```python
result = prottrans_predict_fitness(
    data_csv="data/test.csv",
    model_dir="results/final_model"
)
```

### Complete MCP Workflow

```python
# 1. Extract embeddings for training data
train_emb = prottrans_extract_embeddings(
    csv_path="data/train.csv",
    model_name="ProtT5-XL"
)

# 2. Train fitness model
training = prottrans_train_fitness_model(
    input_dir="data/",
    output_dir="results/",
    backbone_model="ProtT5-XL",
    head_model="svr"
)
print(f"CV Score: {training['cv_mean']:.3f}")

# 3. Extract embeddings for test data
test_emb = prottrans_extract_embeddings(
    csv_path="data/test.csv",
    model_name="ProtT5-XL"
)

# 4. Predict fitness
predictions = prottrans_predict_fitness(
    data_csv="data/test.csv",
    model_dir="results/final_model",
    fitness_col="log_fitness"
)
print(f"Test Spearman: {predictions['metrics']['spearman_r']:.3f}")
```

## Documentation

See [MCP_API_GUIDE.md](MCP_API_GUIDE.md) for detailed API documentation, examples, and best practices.

## Requirements

- PyTorch with CUDA support (recommended)
- Transformers (Hugging Face)
- scikit-learn, pandas, numpy
- GPU: NVIDIA GPU with 8-16GB VRAM recommended