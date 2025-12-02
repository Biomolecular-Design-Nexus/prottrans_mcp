
``` shell
mamba activate alphavariant-env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/esm.git
# for ProtTrans
pip install transformers sentencepiece
```

## Usage

1. Prepare fasta file for the intended `data.csv` file which contain the `seq` and `log_fitness` columns.
2. Run pLM embeddings for the models to be tested.

For `ProtTrans` models:
```shell
# ProtT5-XL
python fitness_model/plm/protrans_emb.py -i example/LanM/data.fasta -m ProtT5-XL
python fitness_model/plm/train.py -i example/LanM/ -o example/LanM/ProtT5-XL -cv -b ProtT5-XL

# ProtAlbert
python fitness_model/plm/protrans_emb.py -i example/LanM/data.fasta -m ProtAlbert
python fitness_model/plm/train.py -i example/LanM/ -o example/LanM/ProtAlbert -cv -b ProtAlbert
```

For `esm` models:
```shell
# Extract embeddings
# Model options: esm2_t33_650M_UR50D, esm1v_t33_650M_UR90S_1 - esm1v_t33_650M_UR90S_5, esm2_t36_3B_UR50D
bash fitness_model/plm/run_esm_embd.sh example/LanM/data.fasta esm2_t36_3B_UR50D
```

3. Train and evaluate pLM models

```shell
# 1) 5 fold cross validation to find best backbone models
# -b options: esm2_t33_650M_UR50D, esm1v_t33_650M_UR90S_1 - esm1v_t33_650M_UR90S_5, esm2_t36_3B_UR50D, ProtAlbert, ProtT5-XL
# usually esm2_t36_3B_UR50D and ProtT5-XL performs well

python fitness_model/plm/train.py -i example/LanM/ -o example/LanM/esm2_t36_3B_UR50D -cv -b esm2_t36_3B_UR50D

# 2) 5 fold cross validation to find best head models (need to try top3 backbone model)
bash fitness_model/plm/run_train_plm_head.sh example/LanM/ head ProtT5-XL

# 3) Select best seed for selected models
bash fitness_model/plm/run_train_plm_head.sh example/LanM/ seed esm2_t36_3B_UR50D gbdt  2>&1 > seed_select3.out 
grep 'esm2_t36_3B_UR50D, gbdt' seed_select3.out
# check the results to bind the best seed for the selected model combination

# 4) Train the final model after comparison
bash fitness_model/plm/run_train_plm_head.sh example/LanM final esm2_t36_3B_UR50D gbdt 3 

```
