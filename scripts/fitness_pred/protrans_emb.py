import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO

import torch
import re

from loguru import logger
from transformers import T5Tokenizer, T5EncoderModel
from transformers import AlbertTokenizer, AutoModel


def load_fasta(fasta_path):
    seqs = [str(fa.seq) for fa in SeqIO.parse(fasta_path, "fasta")]
    return seqs

def load_model(model='ProtT5-XL'):

    if model == 'ProtT5-XL':
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")

    elif model == 'ProtAlbert':
        tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
        model = AutoModel.from_pretrained("Rostlab/prot_albert")

    # elif model == 'ProtBert':
    #     tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    #     model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")

    else:
        raise ValueError(f"Model {model} is not supported")

    return tokenizer, model


def get_embs(model, input_ids, attention_mask, token_lens):
    model.eval()
    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    embedding = embedding.last_hidden_state

    mean_embs = []
    for i, tokens_len in enumerate(token_lens):
        mean_emb = embedding[i, 1:tokens_len].mean(0)
        mean_embs.append(mean_emb)
    mean_embs = torch.stack(mean_embs, dim=0)
    return mean_embs


def get_emb_batch(tokenizer, model, sequences, batch_size=100, device='cuda'):
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
    print(f"number of sequences to embed: {len(sequences)}")

    ids = tokenizer(sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    token_lens = torch.tensor(ids['attention_mask']).sum(dim=1).to(device)

    emb = []
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    remaining_samples = len(sequences)
    batch_start = 0

    for i in tqdm(range(num_batches), total=num_batches, ncols=80):
        batch_size = min(batch_size, remaining_samples)
        batch_end = batch_start + batch_size

        temp_emb = get_embs(model, input_ids[batch_start:batch_end], attention_mask[batch_start:batch_end], token_lens[batch_start:batch_end])
        emb.append(temp_emb.cpu())

        batch_start += batch_size
        remaining_samples -= batch_size
    
    emb = torch.cat(emb, dim=0)
    print(emb.shape)

    return emb.cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Protein fitnes modeling via protein language models')
    parser.add_argument('-i', '--input_path', type=str, help='Input fasta path')

    parser.add_argument('-m', '--model_type', type=str, default='ProtT5-XL', help='Model type, ProtT5-XL or ProtAlbert, default=ProtT5-XL')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device, default=cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='Batch size, default=100')
    return parser.parse_args()


def main():
    args = get_args()

    logger.info(f'device: {args.device}')
    sequences = load_fasta(args.input_path)
    tokenizer, model = load_model(args.model_type)

    model.to(args.device)
    model.full() if args.device=='cpu' else model.half()

    prot_emb = get_emb_batch(tokenizer, model, sequences, batch_size=args.batch_size, device=args.device)
    logger.debug(f'Protein embeddings shape: {prot_emb.shape}')

    embd_dir = os.path.join(os.path.dirname(args.input_path), args.model_type)
    
    Path(embd_dir).mkdir(parents=True, exist_ok=True)
    embd_path = os.path.join(embd_dir, f'{args.model_type}.npy')
    np.save(embd_path, prot_emb)

    logger.info(f'Embbedding saved to path: {embd_path}')


if __name__ == "__main__":
    main()

