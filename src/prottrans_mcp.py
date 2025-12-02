"""
Model Context Protocol (MCP) for prottrans_mcp

Production-ready protein fitness modeling tools using ProtTrans embeddings for protein sequence analysis.
Comprehensive suite of tools for protein fitness prediction, including embedding extraction,
log-likelihood calculation, model training, and fitness prediction.

This MCP Server contains tools for protein fitness modeling:
1. prottrans_fitness_modeling
    - prottrans_extract_embeddings: Extract per-protein embeddings from CSV files using ProtTrans models
    - prottrans_calculate_llh: Calculate ProtBERT log-likelihood for protein mutations
    - prottrans_train_fitness_model: Train regression models for fitness prediction with cross-validation
    - prottrans_predict_fitness: Predict fitness values using pre-trained models
"""

# Suppress SWIG deprecation warnings from sentencepiece
import warnings
warnings.filterwarnings("ignore", message="builtin type Swig.*", category=DeprecationWarning)

from fastmcp import FastMCP
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Import statements (alphabetical order)
from tools.prottrans_fitness_modeling import prottrans_fitness_modeling_mcp

# Server definition and mounting
mcp = FastMCP(name="prottrans_mcp")
mcp.mount(prottrans_fitness_modeling_mcp)

if __name__ == "__main__":
    mcp.run()