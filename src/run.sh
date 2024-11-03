#!/bin/bash

# you can copy the config.txt contents directly into here :) 

algo=DPSGD
model=WideResNet
dataset=CIFAR10
C=0.5
C2=1.0
save_results=True
optimizer=sgd
subset_size=50000
dry_run=False
batch_size=64
epsilon=3
delta=1e-05
epochs=1
lr=0.001
cpu=True

if [ "$algo" = "DPSGD" ]; then
    source env-dpsgd/bin/activate; python main.py --model=$model --dataset=$dataset --algo=$algo --C=$C --C2=$C2 --save_results=$save_results --optimizer=$optimizer --subset_size=$subset_size --dry_run=$dry_run --batch_size=$batch_size --epsilon=$epsilon --delta=$delta --epochs=$epochs --lr=$lr --cpu=$cpu

elif [ "$algo" = "DynamicSGD" ]; then
    source env-dynamic/bin/activate; python main.py --algo=$algo --save_results=$save_results --optimizer=$optimizer --subset_size=$subset_size --dry_run=$dry_run --batch_size=$batch_size --epsilon=$epsilon --epochs=$epochs --lr=$lr --delta=$delta 
else
    echo "Unknown algorithm: $algo"
    exit 1
fi