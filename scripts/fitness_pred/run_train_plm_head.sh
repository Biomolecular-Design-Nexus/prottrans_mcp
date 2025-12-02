#!/bin/bash

input_dir=$1
choice=$2
script_dir=$(dirname "$0")

#==== Select the best head model
# if choice is 'head', then select the head model
if [ "$choice" == "head" ];
then
    backbone_model=$3
    # set backbone_model to esm2_650M by default
    if [ -z "$backbone_model" ]; then
        backbone_model='esm2_t36_3B_UR50D'
    fi

    echo 'Select the best head model'
    for model_type in 'svm' 'random_forest' 'knn' 'gbdt' 'sgd' 'mlp';
    do
        python $script_dir/train.py -i $input_dir -o $input_dir/$choice -m $model_type -b $backbone_model -cv
    done
fi

#==== Select the best data split
# if choice is 'seed', then select the data split
if [ "$choice" == "seed" ]; # example: bash run_train_plm_head.sh /path/to/data seed [backbone_model] [head_model]
then
    backbone_model=$3
    head_model=$4
    # set head_model to svm and backbone_model to esm2_650M by default
    if [ -z "$head_model" ]; then
        head_model='svm'
    fi

    if [ -z "$backbone_model" ]; then
        backbone_model='esm2_t33_650M_UR50D'
    fi

    echo 'Select the best data split, i.e. selecting seed'
    for s in {1..20};
    do
        python $script_dir/train.py -i $input_dir -o $input_dir/$choice -s $s -m $head_model -b $backbone_model
    done
fi

##====### Selected final models
if [ "$choice" == "final" ];
then
    backbone=$3
    head_model=$4
    seed=$5

    echo "Obtain the final model of $input_dir with $backbone, head $head_model, seed $seed"
    mkdir -p $input_dir/$backbone-$head_model-$seed
    python $script_dir/train.py -i $input_dir -o $input_dir/$backbone-$head_model-$seed -s $seed -m $head_model -b $backbone
fi