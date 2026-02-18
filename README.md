# ProtTrans MCP Server

ProtTrans MCP server for protein fitness modeling using ProtTrans protein language models. Provides tools for embedding extraction, log-likelihood scoring, model training, and fitness prediction.

## Installation

### Option 1: Docker (Recommended)

Pull the pre-built GPU image from GHCR and register as an MCP server. No local Python setup needed.

```bash
docker pull ghcr.io/macromnex/prottrans_mcp:latest
```

Register in Claude Code:

```bash
claude mcp add prottrans -- docker run --rm -i --gpus all -v /path/to/data:/app/data ghcr.io/macromnex/prottrans_mcp:latest python src/server.py
```

> **Note:** Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU access (`--gpus all`). Mount your data directory with `-v` so tools can read/write files on the host.

### Option 2: Clone + Download Environment (Colab / Quick Start)

Clone the repo and download a pre-packaged conda environment from GitHub Releases. No compilation required.

```bash
git clone https://github.com/MacromNex/prottrans_mcp.git
cd prottrans_mcp
USE_PACKED_ENVS=1 bash quick_setup.sh
```

This downloads and extracts the pre-built CUDA environment (~2.6 GB). For Google Colab:

```python
import subprocess, os
subprocess.run(["git", "clone", "https://github.com/MacromNex/prottrans_mcp.git"])
os.chdir("prottrans_mcp")
subprocess.run(["bash", "-c", "USE_PACKED_ENVS=1 bash quick_setup.sh"])
```

Register in Claude Code:

```bash
claude mcp add prottrans -- /path/to/prottrans_mcp/env/bin/python /path/to/prottrans_mcp/src/server.py
```

### Option 3: Clone + Create Environment from Scratch

Clone the repo and build the conda environment locally. Requires conda or mamba.

```bash
git clone https://github.com/MacromNex/prottrans_mcp.git
cd prottrans_mcp
bash quick_setup.sh
```

Or install manually:

```bash
conda create -p ./env python=3.10 -y
./env/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
./env/bin/pip install transformers sentencepiece pandas loguru scikit-learn xgboost biopython sniffio
./env/bin/pip install --ignore-installed fastmcp
```

Register in Claude Code:

```bash
claude mcp add prottrans -- ./env/bin/python src/server.py
```

## Local Usage (CLI)

### 1. Create ProtTrans Embeddings
```shell
python scripts/prottrans_embedding.py -i example/data.csv -m ProtT5-XL
```

### 2. Train a Fitness Model
```shell
python scripts/prottrans_train_fitness.py -i example/ -o example/fitness_pred -b ProtT5-XL
```

### 3. Predict with a Fitness Model
```shell
python scripts/prottrans_predict_fitness.py -i example/data.csv -m example/fitness_pred/final_model -b ProtT5-XL
```

### 4. Calculate Log-Likelihood Scores
```shell
python scripts/prottrans_llh.py -i example/data.csv -w example/wt.fasta
```

## MCP Tools

The server exposes 4 tools:

### 1. prottrans_extract_embeddings

Extract ProtTrans embeddings from protein sequences in a CSV file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `csv_path` | str | *required* | Path to CSV file with protein sequences |
| `model_name` | str | `"ProtT5-XL"` | `"ProtT5-XL"` or `"ProtAlbert"` |
| `seq_col` | str | `"seq"` | Column name for sequences |
| `id_column` | str | `None` | Column for sequence IDs |
| `batch_size` | int | `100` | Batch size for processing |
| `device` | str | `"cuda"` | `"cuda"` or `"cpu"` |

### 2. prottrans_calculate_llh

Calculate ProtBERT log-likelihood scores for protein mutations.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_csv` | str | *required* | Path to CSV with `seq` column |
| `wt_fasta` | str | *required* | Path to wild-type FASTA file |
| `n_proc` | int | `None` | Number of processes (auto if None) |
| `device` | str | `"cuda"` | `"cuda"` or `"cpu"` |
| `output_col` | str | `"protbert_llh"` | Output column name |

### 3. prottrans_train_fitness_model

Train a regression model on ProtTrans embeddings with 5-fold cross-validation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_dir` | str | *required* | Directory with `data.csv` and embeddings |
| `output_dir` | str | *required* | Output directory for models and results |
| `backbone_model` | str | `"ProtT5-XL"` | `"ProtT5-XL"` or `"ProtAlbert"` |
| `head_model` | str | `"svr"` | Regression model (see below) |
| `n_components` | int | `60` | Number of PCA components |
| `target_col` | str | `"log_fitness"` | Target column in CSV |
| `seed` | int | `42` | Random seed |

Head model options: `svr`, `random_forest`, `knn`, `gbdt`, `ridge`, `lasso`, `elastic_net`, `mlp`, `xgboost`

### 4. prottrans_predict_fitness

Predict fitness values using a pre-trained model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_csv` | str | *required* | Path to CSV with sequences |
| `model_dir` | str | *required* | Directory with trained model |
| `backbone_model` | str | `"ProtT5-XL"` | Model used for embeddings |
| `seq_col` | str | `"seq"` | Sequence column name |
| `fitness_col` | str | `None` | Ground truth column for evaluation |
| `output_suffix` | str | `"_pred"` | Prediction column suffix |

## Example MCP Prompts

**Train a fitness model:**
```
Can you help train a ProtTrans model for data @example/ and save it to
@results/prot-t5_fitness using the ProtTrans mcp server with ProtT5-XL model.
Please convert the relative path to absolute path before calling the MCP servers.
Please create the embeddings if not ready.
```

**Calculate log-likelihoods:**
```
Can you help calculate ProtBERT likelihood for data @example/data.csv
with wild-type @example/wt.fasta using the prottrans MCP server.
Please convert the relative path to absolute path before calling the MCP servers.
```

## Requirements

- Python 3.10
- PyTorch 2.6.0 (CUDA 11.8 recommended, CPU supported)
- Transformers (Hugging Face)
- scikit-learn, pandas, xgboost, biopython
- GPU: NVIDIA GPU with 8-16 GB VRAM recommended
