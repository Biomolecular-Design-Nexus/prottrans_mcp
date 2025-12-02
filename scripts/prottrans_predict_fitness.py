#!/usr/bin/env python3
"""
ProtTrans-based Protein Fitness Prediction Script

This script uses a pre-trained model to predict fitness values for new protein sequences.
It assumes ProtTrans embeddings have already been extracted for the sequences.

Usage:
    python prottrans_predict_fitness.py -i <data.csv> -m <model_dir> -b <model_name>

Example:
    python prottrans_predict_fitness.py -i data.csv -m results/final_model -b ProtT5-XL
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import torch
from loguru import logger
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_embeddings(data_dir, csv_file, backbone_model="ProtT5-XL", seq_col="seq"):
    """
    Load ProtTrans embeddings for sequences in the CSV file.

    Args:
        data_dir: Directory containing CSV file and embeddings
        csv_file: Name of the CSV file
        backbone_model: ProtTrans model name
        seq_col: Column name containing protein sequences

    Returns:
        embeddings: Feature matrix (embeddings)
        df_data: DataFrame with original data
    """
    csv_path = os.path.join(data_dir, csv_file)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df_data = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df_data)} samples from {csv_path}")

    # Check if sequence column exists
    if seq_col not in df_data.columns:
        raise ValueError(
            f"Sequence column '{seq_col}' not found in CSV. "
            f"Available columns: {df_data.columns.tolist()}"
        )

    # Try to load pre-computed embeddings file
    # Check two possible locations:
    # 1. csv_path.{backbone_model}.npy
    # 2. data_dir/{backbone_model}/{backbone_model}.npy
    embd_save_path = f"{csv_path}.{backbone_model}.npy"

    if os.path.exists(embd_save_path):
        logger.info(f"Loading pre-computed embeddings from {embd_save_path}")
        prot_embd = np.load(embd_save_path)
    else:
        uniq_fasta_file = os.path.join(data_dir, "sequences.fasta")
        uniq_embd_file = os.path.join(data_dir, backbone_model, f"{backbone_model}.npy")
        if not os.path.exists(uniq_fasta_file):
            raise FileNotFoundError(
                f"Fasta file not found: {uniq_fasta_file}. "
                f"Please ensure sequences.fasta is present."
            )
        if not os.path.exists(uniq_embd_file):
            raise FileNotFoundError(
                f"Embeddings file not found: {uniq_embd_file}. "
                f"Please compute embeddings first."
            )
        uniq_prot_embd = np.load(uniq_embd_file)

        # Load sequences from fasta
        seqs = []
        with open(uniq_fasta_file, "r") as f:
            for line in f:
                if line.startswith(">"):
                    continue
                seqs.append(line.strip())

        logger.info(f"Loaded {len(seqs)} sequences from fasta")

        # Map sequences to embeddings
        seq2embd = {seq: embd for seq, embd in zip(seqs, uniq_prot_embd)}

        # Now get embeddings for all sequences in CSV in order
        prot_embd = []

        for idx, seq in enumerate(df_data[seq_col]):
            if seq in seq2embd:
                prot_embd.append(seq2embd[seq])
            else:
                raise ValueError(
                    f"Could not find embeddings for sequence {seq} at index {idx} in {csv_file}"
                )

        prot_embd = np.array(prot_embd)

        # Save for future use
        logger.info(f"Saving combined embeddings to {embd_save_path}")
        np.save(embd_save_path, prot_embd)

    embeddings = prot_embd
    logger.info(f"Embeddings shape: {embeddings.shape}")

    return embeddings, df_data


def evaluate_predictions(y_true, y_pred, n_total=None, output_dir=None):
    """
    Evaluate predictions against ground truth values.

    Args:
        y_true: Ground truth fitness values
        y_pred: Predicted fitness values
        n_total: Total number of samples (for reporting valid pairs)
        output_dir: Optional directory to save evaluation results

    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Calculate metrics
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "n_valid": len(y_true),
        "n_total": n_total if n_total is not None else len(y_true),
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    # Log results
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Metrics")
    logger.info("=" * 80)
    if n_total is not None:
        logger.info(f"  Valid pairs: {len(y_true)} / {n_total}")
    logger.info(f"  Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
    logger.info(f"  Pearson correlation:  {pearson_r:.4f} (p={pearson_p:.2e})")
    logger.info(f"  R² score:             {r2:.4f}")
    logger.info(f"  MSE:                  {mse:.4f}")
    logger.info(f"  RMSE:                 {rmse:.4f}")
    logger.info(f"  MAE:                  {mae:.4f}")
    logger.info("=" * 80)

    # Save metrics if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_df = pd.DataFrame([metrics])
        metrics_path = output_dir / "evaluation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Evaluation metrics saved to: {metrics_path}")

    return metrics


def predict_fitness(
    data_csv,
    model_dir,
    backbone_model="ProtT5-XL",
    seq_col="seq",
    fitness_col=None,
    output_suffix="_pred",
):
    """
    Predict fitness values using a pre-trained model.

    Args:
        data_csv: Path to CSV file containing sequences
        model_dir: Directory containing trained model and PCA transformer
        backbone_model: ProtTrans model name (must match training)
        seq_col: Column name containing protein sequences
        fitness_col: Optional column name containing ground truth fitness values for evaluation
        output_suffix: Suffix for output column name

    Returns:
        df_results: DataFrame with original data and predictions
        predictions: Array of predicted fitness values
        metrics: Dictionary of evaluation metrics (only if fitness_col is provided and exists)
    """
    # Convert paths
    data_csv = Path(data_csv)
    model_dir = Path(model_dir)
    data_dir = data_csv.parent

    # Load the trained models
    logger.info(f"Loading models from {model_dir}")

    pca_path = model_dir / "pca_model.joblib"
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA model not found: {pca_path}")
    pca_model = joblib.load(pca_path)
    logger.info(f"Loaded PCA model with {pca_model.n_components} components")

    # Find the head model file (could be different names)
    head_model_files = list(model_dir.glob("head_model_*.joblib"))
    if not head_model_files:
        raise FileNotFoundError(f"No head model found in {model_dir}")
    head_model_path = head_model_files[0]
    head_model = joblib.load(head_model_path)
    logger.info(f"Loaded head model: {head_model_path.name}")

    # Load embeddings
    logger.info(f"Loading embeddings for {data_csv.name}")
    embeddings, df_data = load_embeddings(
        data_dir=str(data_dir),
        csv_file=data_csv.name,
        backbone_model=backbone_model,
        seq_col=seq_col,
    )

    # Apply PCA transformation
    logger.info("Applying PCA transformation...")
    embeddings_pca = pca_model.transform(embeddings)
    explained_var = np.sum(pca_model.explained_variance_ratio_)
    logger.info(
        f"PCA transformed to {embeddings_pca.shape[1]} components "
        f"(explains {explained_var:.3%} variance)"
    )

    # Make predictions
    logger.info("Making predictions...")
    predictions = head_model.predict(embeddings_pca)
    logger.info(f"Generated {len(predictions)} predictions")

    # Add predictions to dataframe
    df_results = df_data.copy()
    df_results[f"fitness{output_suffix}"] = predictions

    # Calculate statistics
    logger.info("\nPrediction Statistics:")
    logger.info(f"  Mean: {np.mean(predictions):.4f}")
    logger.info(f"  Std:  {np.std(predictions):.4f}")
    logger.info(f"  Min:  {np.min(predictions):.4f}")
    logger.info(f"  Max:  {np.max(predictions):.4f}")

    # Evaluate predictions if ground truth fitness column is available
    metrics = None
    fitness_col_found = None

    if fitness_col:
        # User specified a fitness column
        if fitness_col in df_data.columns:
            fitness_col_found = fitness_col
        else:
            logger.warning(
                f"\nSpecified fitness column '{fitness_col}' not found in data. "
                f"Available columns: {df_data.columns.tolist()}"
            )
    else:
        # Auto-detect common fitness column names
        for col in ["fitness", "log_fitness", "score"]:
            if col in df_data.columns:
                fitness_col_found = col
                logger.info(f"Auto-detected fitness column: '{col}'")
                break

    if fitness_col_found:
        logger.info(f"\nEvaluating predictions against ground truth column: '{fitness_col_found}'")

        # Handle potential NaN values
        pred_series = pd.Series(predictions)
        fitness_series = pd.to_numeric(df_data[fitness_col_found], errors="coerce")

        valid_mask = ~(pd.isna(pred_series) | pd.isna(fitness_series))
        y_true = fitness_series[valid_mask].values
        y_pred = pred_series[valid_mask].values

        if len(y_true) > 2:
            metrics = evaluate_predictions(
                y_true, y_pred, n_total=len(df_data), output_dir=model_dir
            )
        else:
            logger.warning(f"Not enough valid pairs for evaluation: {len(y_true)}")

    return df_results, predictions, metrics


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict protein fitness using pre-trained ProtTrans-based models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict fitness for new sequences
  python prottrans_predict_fitness.py -i data.csv -m results/final_model

  # Use ProtAlbert model
  python prottrans_predict_fitness.py -i data.csv -m results/final_model -b ProtAlbert

  # Custom sequence column
  python prottrans_predict_fitness.py -i data.csv -m results/final_model -s sequence

  # Custom output file
  python prottrans_predict_fitness.py -i data.csv -m results/final_model -o predictions.csv

  # Predict and evaluate against ground truth
  python prottrans_predict_fitness.py -i data.csv -m results/final_model -f log_fitness

Workflow:
  1. Extract embeddings: python prottrans_embedding.py -i data.csv -m ProtT5-XL
  2. Predict fitness: python prottrans_predict_fitness.py -i data.csv -m results/final_model
  3. (Optional) Evaluate: Add -f <fitness_column> to compute evaluation metrics

Evaluation:
  If the CSV contains ground truth fitness values, use the -f/--fitness_col argument
  to compute evaluation metrics including Spearman correlation, Pearson correlation,
  R² score, MSE, RMSE, and MAE. Metrics will be saved to evaluation_metrics.csv.
  If not specified, common column names ('fitness', 'log_fitness', 'score') will be
  auto-detected.

Requirements:
  - CSV file with protein sequences
  - Pre-extracted ProtTrans embeddings (sequences.fasta and model directory)
  - Trained model directory (containing head_model_*.joblib and pca_model.joblib)
        """,
    )

    parser.add_argument(
        "-i",
        "--input_csv",
        type=str,
        required=True,
        help="Path to CSV file containing protein sequences",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing trained model (head_model_*.joblib and pca_model.joblib)",
    )
    parser.add_argument(
        "-b",
        "--backbone_model",
        type=str,
        default="ProtT5-XL",
        choices=["ProtT5-XL", "ProtAlbert"],
        help="ProtTrans backbone model used for embeddings (default: ProtT5-XL)",
    )
    parser.add_argument(
        "-s",
        "--seq_col",
        type=str,
        default="seq",
        help="Column name containing protein sequences (default: seq)",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV file path. If not provided, uses <input_csv>_<model_name>_pred.csv",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_pred",
        help='Suffix for prediction column name (default: _pred, results in "fitness_pred")',
    )
    parser.add_argument(
        "-f",
        "--fitness_col",
        type=str,
        default=None,
        help="Column name containing ground truth fitness values for evaluation (optional)",
    )

    return parser.parse_args()


def main():
    """Main prediction pipeline."""
    args = get_args()

    # Setup logging
    logger.info("=" * 60)
    logger.info("ProtTrans Fitness Prediction")
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Backbone model: {args.backbone_model}")
    logger.info(f"Sequence column: {args.seq_col}")
    logger.info("=" * 60)

    # Make predictions
    try:
        df_results, predictions, metrics = predict_fitness(
            data_csv=args.input_csv,
            model_dir=args.model_dir,
            backbone_model=args.backbone_model,
            seq_col=args.seq_col,
            fitness_col=args.fitness_col,
            output_suffix=args.output_suffix,
        )

        # Determine output path
        if args.output_csv:
            output_path = args.output_csv
        else:
            # Extract model name from model_dir
            model_dir_name = (
                Path(args.model_dir).parent.name
                if Path(args.model_dir).name == "final_model"
                else Path(args.model_dir).name
            )
            input_path = Path(args.input_csv)
            output_path = str(
                input_path.parent / f"{input_path.stem}_{model_dir_name}_pred.csv"
            )

        # Save results
        df_results.to_csv(output_path, index=False)
        logger.info("=" * 60)
        logger.info("Prediction Complete!")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Total predictions: {len(predictions)}")

        # Display evaluation summary if available
        if metrics:
            logger.info("\nEvaluation Summary:")
            logger.info(f"  Spearman ρ: {metrics['spearman_r']:.4f}")
            logger.info(f"  Pearson r:  {metrics['pearson_r']:.4f}")
            logger.info(f"  R² score:   {metrics['r2']:.4f}")
            logger.info(f"  RMSE:       {metrics['rmse']:.4f}")

        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error("=" * 60)
        logger.error("Prediction Failed!")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    exit(main())
