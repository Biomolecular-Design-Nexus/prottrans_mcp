#!/bin/bash

fastq_path=$1
model_name=$2
out_dir=$(dirname "$fastq_path")

##==== Extract the features
# check if model name is in the allowed list
esm_650M_models=("esm2_t33_650M_UR50D" "esm1v_t33_650M_UR90S_1" "esm1v_t33_650M_UR90S_2" "esm1v_t33_650M_UR90S_3" "esm1v_t33_650M_UR90S_4" "esm1v_t33_650M_UR90S_5")

if [[ " ${esm_650M_models[@]} " =~ " ${model_name} " ]]; then
    echo "Extract embeddings with Model: $model_name"
    esm-extract $model_name $fastq_path $out_dir/$model_name --repr_layers 33 --include mean

elif [[ $model_name == 'esm2_t36_3B_UR50D' ]]; then
    echo "Extract embeddings with Model: $model_name"
    esm-extract $model_name $fastq_path $out_dir/$model_name --repr_layers 36 --include mean

else
    echo "Error: Model name $model_name not recognized. Please use one of the following:"
    echo "For ESM-2 650M models: ${esm_650M_models[*]}"
    echo "For ESM-2 3B models: esm2_t36_3B_UR50D"
    exit 1
fi
