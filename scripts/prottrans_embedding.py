#!/usr/bin/env python3
"""
ProtTrans Embedding Extraction Script

This script extracts ProtTrans embeddings from a CSV file containing protein sequences.
It creates a FASTA file and extracts embeddings using ProtTrans models (ProtT5-XL or ProtAlbert).

Usage:
    python prottrans_embedding.py -i <csv_path> -m <model_name> [-s <seq_col>] [-d <id_column>]

Example:
    python prottrans_embedding.py -i data/proteins.csv -m ProtT5-XL -s seq
"""

import argparse
import pandas as pd
import numpy as np
import re
import torch
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm
from loguru import logger

from transformers import T5Tokenizer, T5EncoderModel
from transformers import AlbertTokenizer, AutoModel


def load_prottrans_model(model_name: str = "ProtT5-XL", device: str = "cuda"):
    """
    Load ProtTrans model and tokenizer.

    Args:
        model_name: ProtTrans model name ('ProtT5-XL' or 'ProtAlbert')
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        tokenizer: Tokenizer for the model
        model: ProtTrans model
    """
    logger.info(f"Loading ProtTrans model: {model_name}")

    if model_name == "ProtT5-XL":
        tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
        )
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    elif model_name == "ProtAlbert":
        tokenizer = AlbertTokenizer.from_pretrained(
            "Rostlab/prot_albert", do_lower_case=False
        )
        model = AutoModel.from_pretrained("Rostlab/prot_albert")
    else:
        raise ValueError(
            f"Model {model_name} is not supported. "
            f"Available models: ProtT5-XL, ProtAlbert"
        )

    # Move model to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set precision based on device
    if device.type == "cpu":
        model.full()
    else:
        model.half()

    model.eval()
    logger.info(f"Model loaded on device: {device}")

    return tokenizer, model


def get_embeddings(model, input_ids, attention_mask, token_lens):
    """
    Extract embeddings from model.

    Args:
        model: ProtTrans model
        input_ids: Input token IDs
        attention_mask: Attention mask
        token_lens: Length of tokens for each sequence

    Returns:
        Mean embeddings for each sequence
    """
    model.eval()
    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    embedding = embedding.last_hidden_state

    mean_embs = []
    for i, tokens_len in enumerate(token_lens):
        # Skip special tokens (first and last)
        mean_emb = embedding[i, 1 : tokens_len - 1].mean(0)
        mean_embs.append(mean_emb)
    mean_embs = torch.stack(mean_embs, dim=0)
    return mean_embs


def get_embeddings_batch(
    tokenizer, model, sequences, batch_size=100, device="cuda"
):
    """
    Get embeddings for a batch of sequences.

    Args:
        tokenizer: ProtTrans tokenizer
        model: ProtTrans model
        sequences: List of protein sequences
        batch_size: Batch size for processing
        device: Device to run on

    Returns:
        Numpy array of embeddings
    """
    # Preprocess sequences: replace uncommon amino acids and add spaces
    sequences = [
        " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences
    ]
    logger.info(f"Number of sequences to embed: {len(sequences)}")

    # Tokenize all sequences
    ids = tokenizer(sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)
    token_lens = torch.tensor(ids["attention_mask"]).sum(dim=1).to(device)

    emb = []
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    remaining_samples = len(sequences)
    batch_start = 0

    for i in tqdm(range(num_batches), total=num_batches, ncols=80):
        current_batch_size = min(batch_size, remaining_samples)
        batch_end = batch_start + current_batch_size

        temp_emb = get_embeddings(
            model,
            input_ids[batch_start:batch_end],
            attention_mask[batch_start:batch_end],
            token_lens[batch_start:batch_end],
        )
        emb.append(temp_emb.cpu())

        batch_start += current_batch_size
        remaining_samples -= current_batch_size

    emb = torch.cat(emb, dim=0)
    logger.info(f"Embeddings shape: {emb.shape}")

    return emb.cpu().numpy()


def extract_embeddings_from_csv(
    csv_path: Union[str, Path],
    model_name: str = "ProtT5-XL",
    seq_col: str = "seq",
    id_column: Optional[str] = None,
    batch_size: int = 100,
    device: str = "cuda",
) -> dict:
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
        batch_size: Batch size for processing sequences
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Dictionary with FASTA path, embeddings directory, and metadata

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If sequence column is not found in CSV
    """
    # Validate CSV path
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV file
    logger.info(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from CSV")

    # Validate sequence column exists
    if seq_col not in df.columns:
        raise ValueError(
            f"Column '{seq_col}' not found in CSV. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Extract unique sequences
    if id_column and id_column in df.columns:
        # Use provided ID column
        df_unique = df[[id_column, seq_col]].drop_duplicates(subset=[seq_col])
        seq_ids = df_unique[id_column].tolist()
        sequences = df_unique[seq_col].tolist()
        logger.info(f"Using ID column '{id_column}' for sequence IDs")
    else:
        # Generate seq_0, seq_1, etc. for unique sequences
        unique_sequences = df[seq_col].unique()
        seq_ids = [f"seq_{i}" for i in range(len(unique_sequences))]
        sequences = unique_sequences.tolist()
        logger.info(f"Generated sequence IDs: seq_0 to seq_{len(sequences)-1}")

    logger.info(f"Found {len(sequences)} unique sequences")

    # Set output directory to same directory as CSV file
    output_dir = csv_path.parent

    # Create FASTA file in the same directory as CSV
    fasta_path = output_dir / "sequences.fasta"
    logger.info(f"\nCreating FASTA file: {fasta_path}")
    with open(fasta_path, "w") as f:
        for seq_id, seq in zip(seq_ids, sequences):
            f.write(f">{seq_id}\n")
            f.write(f"{seq}\n")
    logger.info(f"FASTA file created with {len(sequences)} sequences")

    # Create output directory for embeddings in same directory as CSV
    embeddings_dir = output_dir / model_name
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nEmbeddings will be saved to: {embeddings_dir}")

    try:
        # Load model
        tokenizer, model = load_prottrans_model(model_name, device)

        # Extract embeddings
        logger.info("\nExtracting embeddings...")
        logger.info("This may take a while depending on the number of sequences...\n")

        prot_emb = get_embeddings_batch(
            tokenizer, model, sequences, batch_size=batch_size, device=device
        )

        # Save embeddings as .npy file
        emb_path = embeddings_dir / f"{model_name}.npy"
        np.save(emb_path, prot_emb)
        logger.info(f"Embeddings saved to: {emb_path}")

        logger.info("=" * 80)
        logger.info("SUCCESS: Embeddings extracted successfully!")
        logger.info("=" * 80)

        # Prepare response
        response = {
            "status": "success",
            "csv_path": str(csv_path),
            "fasta_path": str(fasta_path),
            "embeddings_dir": str(embeddings_dir),
            "embeddings_file": str(emb_path),
            "model_name": model_name,
            "num_sequences": len(sequences),
            "num_unique_sequences": len(sequences),
            "embedding_dim": prot_emb.shape[1],
            "sequence_ids_sample": (
                seq_ids[:10] if len(seq_ids) > 10 else seq_ids
            ),  # Show first 10
            "total_ids": len(seq_ids),
        }

        # Print summary
        logger.info(f"\nSummary:")
        logger.info(f"  CSV file:         {csv_path}")
        logger.info(f"  FASTA file:       {fasta_path}")
        logger.info(f"  Embeddings dir:   {embeddings_dir}")
        logger.info(f"  Embeddings file:  {emb_path}")
        logger.info(f"  Model:            {model_name}")
        logger.info(f"  Total sequences:  {len(sequences)}")
        logger.info(f"  Unique sequences: {len(sequences)}")
        logger.info(f"  Embedding dim:    {prot_emb.shape[1]}")

        return response

    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR: Failed to extract embeddings")
        logger.error("=" * 80)
        logger.error(f"\nError: {e}")

        return {
            "status": "error",
            "error_message": str(e),
            "csv_path": str(csv_path),
            "fasta_path": str(fasta_path),
            "embeddings_dir": str(embeddings_dir),
        }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract ProtTrans embeddings from a CSV file containing protein sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract embeddings using default ProtT5-XL model
  python prottrans_embedding.py -i data/proteins.csv

  # Use ProtAlbert model with custom sequence column
  python prottrans_embedding.py -i data/proteins.csv -m ProtAlbert -s sequence

  # Use custom ID column
  python prottrans_embedding.py -i data/proteins.csv -d protein_id

  # Use CPU instead of GPU
  python prottrans_embedding.py -i data/proteins.csv --device cpu

Available models:
  - ProtT5-XL (default) - T5-based model with 3B parameters
  - ProtAlbert - Albert-based model with 225M parameters
        """,
    )

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to CSV file containing protein sequences",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="ProtT5-XL",
        choices=["ProtT5-XL", "ProtAlbert"],
        help="ProtTrans model name to use for embeddings extraction (default: ProtT5-XL)",
    )
    parser.add_argument(
        "-s",
        "--seq_col",
        type=str,
        default="seq",
        help="Column name containing protein sequences (default: seq)",
    )
    parser.add_argument(
        "-d",
        "--id_column",
        type=str,
        default=None,
        help="Column name containing sequence IDs. If not provided, generates seq_0, seq_1, etc.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for processing sequences (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )

    args = parser.parse_args()

    # Extract embeddings
    result = extract_embeddings_from_csv(
        csv_path=args.input_path,
        model_name=args.model,
        seq_col=args.seq_col,
        id_column=args.id_column,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Exit with appropriate code
    if result["status"] == "success":
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
