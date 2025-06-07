#!/bin/bash

#MLP Ablations 
echo "MLP 3 layers"
python3 main.py train --exp_name GAT_MLP_3 --num_MLP_layers 3

echo "MLP 5 layers"
python3 main.py train --exp_name GAT_MLP_3 --num_MLP_layers 5

echo "MLP 1 layer"
python3 main.py train --exp_name GAT_MLP_3 --num_MLP_layers 1