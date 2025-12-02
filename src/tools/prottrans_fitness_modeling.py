"""
ProtTrans fitness modeling tools.

This module provides MCP-compatible functions for protein fitness modeling using ProtTrans embeddings.
Includes tools for embedding extraction, log-likelihood calculation, model training, and fitness prediction.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from fastmcp import FastMCP

# Add scripts directory to Python path to import script functions
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import functions from scripts
from prottrans_embedding import extract_embeddings_from_csv
from prottrans_llh import calculate_protbert_llh
from prottrans_train_fitness import (
    load_data,
    perform_cross_validation,
    train_final_model,
    set_random_seed,
)
from prottrans_predict_fitness import predict_fitness

# Create the MCP instance for fitness modeling tools
prottrans_fitness_modeling_mcp = FastMCP("prottrans_fitness_modeling")


@prottrans_fitness_modeling_mcp.tool()
def prottrans_extract_embeddings(
    csv_path: str,
    model_name: str = "ProtT5-XL",
    seq_col: str = "seq",
    id_column: Optional[str] = None,
    batch_size: int = 100,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Extract ProtTrans embeddings from a CSV file containing protein sequences.

    This function:
    1. Reads a CSV file with protein sequences
    2. Extracts unique sequences from the specified column
    3. Creates a FASTA file in the same directory as the CSV
    4. Generates embeddings using ProtTrans models
    5. Saves embeddings in a model-named directory
    6. Returns paths to generated files and embedding statistics

    Args:
        csv_path: Path to CSV file containing protein sequences
        model_name: ProtTrans model name ('ProtT5-XL' or 'ProtAlbert')
        seq_col: Column name containing protein sequences (default: "seq")
        id_column: Column name containing sequence IDs. If None, generates seq_0, seq_1, etc.
        batch_size: Batch size for processing sequences (default: 100)
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Dictionary with FASTA path, embeddings directory, and metadata

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If sequence column is not found in CSV

    Example:
        result = prottrans_extract_embeddings(
            csv_path="data/proteins.csv",
            model_name="ProtT5-XL",
            seq_col="seq"
        )
    """
    try:
        result = extract_embeddings_from_csv(
            csv_path=csv_path,
            model_name=model_name,
            seq_col=seq_col,
            id_column=id_column,
            batch_size=batch_size,
            device=device,
        )
        return result
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "csv_path": csv_path,
        }


@prottrans_fitness_modeling_mcp.tool()
def prottrans_calculate_llh(
    data_csv: str,
    wt_fasta: str,
    n_proc: Optional[int] = None,
    device: str = "cuda",
    output_col: str = "protbert_llh",
) -> Dict[str, Any]:
    """
    Calculate ProtBERT log-likelihood for mutations in a CSV file.

    This function uses ProtBERT's masked language modeling to calculate pseudo-log-likelihood
    scores for protein mutations. Mutations are automatically derived by comparing sequences
    to the wild-type sequence.

    Args:
        data_csv: Path to CSV file with 'seq' column containing protein sequences
        wt_fasta: Path to wild-type FASTA file
        n_proc: Number of processes for parallel computation (None = auto)
        device: Device to use ('cuda' or 'cpu')
        output_col: Name for output column (default: 'protbert_llh')

    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - output_csv: Path to CSV file with LLH values
        - n_variants: Number of variants processed
        - metrics: Correlation metrics if fitness column is found

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If required columns are missing

    Example:
        result = prottrans_calculate_llh(
            data_csv="data/variants.csv",
            wt_fasta="data/wildtype.fasta",
            device="cuda"
        )

    Note:
        This uses ProtBERT for masked language modeling. ProtT5-XL and ProtAlbert
        are encoder-only models and do not support direct log-likelihood calculation.
    """
    try:
        # Import from script
        from prottrans_llh import calculate_protbert_llh as calc_llh
        import pandas as pd
        from pathlib import Path

        # Calculate log-likelihoods
        df_results = calc_llh(
            data_csv=data_csv,
            wt_fasta=wt_fasta,
            n_proc=n_proc,
            device=device,
            output_col=output_col,
        )

        # Determine output path
        data_path = Path(data_csv)
        output_csv = str(data_path.parent / f"{data_path.stem}_protbert_llh.csv")

        # Save results
        df_results.to_csv(output_csv, index=False)

        # Prepare response
        result = {
            "status": "success",
            "output_csv": output_csv,
            "n_variants": len(df_results),
            "columns": df_results.columns.tolist(),
        }

        # Add correlation metrics if fitness column exists
        fitness_cols = ["fitness", "log_fitness"]
        for col in fitness_cols:
            if col in df_results.columns:
                from scipy.stats import spearmanr
                import numpy as np

                llh_series = pd.to_numeric(df_results[output_col], errors="coerce")
                fitness_series = pd.to_numeric(df_results[col], errors="coerce")
                valid_mask = ~(pd.isna(llh_series) | pd.isna(fitness_series))

                if valid_mask.sum() > 2:
                    spearman_r, spearman_p = spearmanr(
                        llh_series[valid_mask], fitness_series[valid_mask]
                    )
                    result["metrics"] = {
                        "fitness_column": col,
                        "spearman_r": float(spearman_r),
                        "spearman_p": float(spearman_p),
                        "n_valid_pairs": int(valid_mask.sum()),
                    }
                break

        return result

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "data_csv": data_csv,
            "wt_fasta": wt_fasta,
        }


@prottrans_fitness_modeling_mcp.tool()
def prottrans_train_fitness_model(
    input_dir: str,
    output_dir: str,
    backbone_model: str = "ProtT5-XL",
    head_model: str = "svr",
    n_components: int = 60,
    target_col: str = "log_fitness",
    seed: int = 42,
    no_final_model: bool = False,
) -> Dict[str, Any]:
    """
    Train a regression model on ProtTrans embeddings for protein fitness prediction.

    This function performs 5-fold cross-validation for evaluation and then trains a
    final model on all data. Results include cross-validation metrics and trained models.

    Args:
        input_dir: Input directory containing data.csv and embeddings
        output_dir: Output directory for models and results
        backbone_model: ProtTrans backbone model ('ProtT5-XL' or 'ProtAlbert')
        head_model: Regression head model (svr, random_forest, knn, gbdt, ridge, lasso, elastic_net, mlp, xgboost)
        n_components: Number of PCA components (default: 60)
        target_col: Target column name in CSV (default: 'log_fitness')
        seed: Random seed for reproducibility (default: 42)
        no_final_model: Skip training final model on all data (only do CV)

    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - output_dir: Path to output directory
        - cv_results: Cross-validation results with metrics for each fold
        - cv_mean: Mean cross-validation Spearman correlation
        - cv_std: Standard deviation of cross-validation Spearman correlation
        - final_train_spearman: Final model training Spearman (if no_final_model=False)
        - model_paths: Paths to saved models

    Raises:
        FileNotFoundError: If input directory or required files don't exist
        ValueError: If target column is not found in CSV

    Example:
        result = prottrans_train_fitness_model(
            input_dir="data/proteins",
            output_dir="results",
            backbone_model="ProtT5-XL",
            head_model="svr",
            n_components=60
        )

    Available head models:
        - svr (default): Support Vector Regression
        - random_forest: Random Forest Regressor
        - knn: K-Nearest Neighbors
        - gbdt: Gradient Boosting Decision Trees
        - ridge: Ridge Regression
        - lasso: Lasso Regression
        - elastic_net: Elastic Net Regression
        - mlp: Multi-Layer Perceptron
        - xgboost: XGBoost Regressor
    """
    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        import argparse

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set random seed
        set_random_seed(seed)

        # Create args object to pass to functions
        class Args:
            pass

        args = Args()
        args.input_dir = input_dir
        args.output_dir = output_dir
        args.backbone_model = backbone_model
        args.head_model = head_model
        args.n_components = n_components
        args.target_col = target_col
        args.seed = seed
        args.no_final_model = no_final_model

        # Load data
        Xs, Ys = load_data(input_dir, backbone_model, target_col)

        # Perform cross-validation
        cv_scores, fold_results = perform_cross_validation(Xs, Ys, args)

        # Calculate CV statistics
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        # Save CV results
        df_cv_results = pd.DataFrame(fold_results)
        cv_results_path = Path(output_dir) / f"{backbone_model}_{head_model}_cv_results.csv"
        df_cv_results.to_csv(cv_results_path, index=False)

        result = {
            "status": "success",
            "output_dir": output_dir,
            "cv_results": fold_results,
            "cv_mean": float(mean_cv_score),
            "cv_std": float(std_cv_score),
            "cv_min": float(np.min(cv_scores)),
            "cv_max": float(np.max(cv_scores)),
            "cv_results_file": str(cv_results_path),
            "n_samples": len(Xs),
            "n_features": Xs.shape[1],
        }

        # Train final model unless skipped
        if not no_final_model:
            final_model, pca_model, train_spearman = train_final_model(Xs, Ys, args)
            result["final_train_spearman"] = float(train_spearman)
            result["model_paths"] = {
                "final_model_dir": str(Path(output_dir) / "final_model"),
                "pca_model": str(Path(output_dir) / "final_model" / "pca_model.joblib"),
                "head_model": str(
                    Path(output_dir) / "final_model" / f"head_model_{head_model}.joblib"
                ),
            }

        # Save summary
        summary_df = pd.DataFrame([result])
        summary_path = Path(output_dir) / "training_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        result["summary_file"] = str(summary_path)

        return result

    except Exception as e:
        import traceback

        return {
            "status": "error",
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "input_dir": input_dir,
            "output_dir": output_dir,
        }


@prottrans_fitness_modeling_mcp.tool()
def prottrans_predict_fitness(
    data_csv: str,
    model_dir: str,
    backbone_model: str = "ProtT5-XL",
    seq_col: str = "seq",
    fitness_col: Optional[str] = None,
    output_suffix: str = "_pred",
) -> Dict[str, Any]:
    """
    Predict fitness values using a pre-trained ProtTrans-based model.

    This function uses a trained model to predict fitness values for new protein sequences.
    It assumes ProtTrans embeddings have already been extracted for the sequences.

    Args:
        data_csv: Path to CSV file containing protein sequences
        model_dir: Directory containing trained model (head_model_*.joblib and pca_model.joblib)
        backbone_model: ProtTrans backbone model used for embeddings ('ProtT5-XL' or 'ProtAlbert')
        seq_col: Column name containing protein sequences (default: 'seq')
        fitness_col: Optional column name containing ground truth fitness values for evaluation
        output_suffix: Suffix for prediction column name (default: '_pred')

    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - output_csv: Path to CSV file with predictions
        - n_predictions: Number of predictions made
        - prediction_stats: Statistics about predictions (mean, std, min, max)
        - metrics: Evaluation metrics if fitness_col is provided and exists

    Raises:
        FileNotFoundError: If input files or model files don't exist
        ValueError: If required columns are missing

    Example:
        # First extract embeddings
        prottrans_extract_embeddings(
            csv_path="data/test.csv",
            model_name="ProtT5-XL"
        )

        # Then predict fitness
        result = prottrans_predict_fitness(
            data_csv="data/test.csv",
            model_dir="results/final_model",
            backbone_model="ProtT5-XL"
        )

    Workflow:
        1. Extract embeddings using prottrans_extract_embeddings
        2. Train model using prottrans_train_fitness_model
        3. Predict fitness using prottrans_predict_fitness

    Evaluation:
        If the CSV contains ground truth fitness values, use the fitness_col argument
        to compute evaluation metrics including Spearman correlation, Pearson correlation,
        R² score, MSE, RMSE, and MAE. If not specified, common column names
        ('fitness', 'log_fitness', 'score') will be auto-detected.
    """
    try:
        from pathlib import Path
        import numpy as np

        # Make predictions
        df_results, predictions, metrics = predict_fitness(
            data_csv=data_csv,
            model_dir=model_dir,
            backbone_model=backbone_model,
            seq_col=seq_col,
            fitness_col=fitness_col,
            output_suffix=output_suffix,
        )

        # Determine output path
        data_path = Path(data_csv)
        model_dir_path = Path(model_dir)
        model_dir_name = (
            model_dir_path.parent.name
            if model_dir_path.name == "final_model"
            else model_dir_path.name
        )
        output_csv = str(data_path.parent / f"{data_path.stem}_{model_dir_name}_pred.csv")

        # Save results
        df_results.to_csv(output_csv, index=False)

        result = {
            "status": "success",
            "output_csv": output_csv,
            "n_predictions": len(predictions),
            "prediction_stats": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
            },
            "columns": df_results.columns.tolist(),
        }

        # Add evaluation metrics if available
        if metrics:
            result["metrics"] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in metrics.items()
            }

        return result

    except Exception as e:
        import traceback

        return {
            "status": "error",
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "data_csv": data_csv,
            "model_dir": model_dir,
        }
