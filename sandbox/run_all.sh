#!/bin/bash

BASE_FLAGS="--devices 1 --num_classes 3 --accelerator gpu --gradient_clip_val 0.5 --max_epochs 1000 --patience 50 --limit_train_batches 1.0 --tb_path $HOME/Desktop/AlgoverseResearch/u/isaacpicov/baseline/experiments/simulation/gnn_results/tb --loss_fn BCE --learn_rate 1e-3 --batch_size 64 --optimizer Adam --dataset_path action_dataset_joints_leg_sampled_150.pkl --proj_name hmr-gnn --dp_rate 0.1 --metrics_strategy weighted --oversample False --c_in 3 --c_hidden 128 --num_MLP_layers 3 --layer_name GAT --num_frames 150 --num_workers 4 --pin_memory True"

echo "=== Heads 8 layers 5 ==="
python3 main.py train --exp_name "Heads=8, layers=5" --attn_heads 8 --layers 5 --split False

echo "joint ablations"
python3 main.py train --exp_name "Heads=4, layers=5, exlcude leg, thorax, spine" --attn_heads 4 --layers 5 --split False --exlcude_groups '["leg_joints", "thorax_joints", "spine_joints"]'
python3 main.py train --exp_name "Heads=4, layers=5 exlcude thorax, spine" --attn_heads 4 --layers 5 --split False --exlcude_groups '["thorax_joints", "spine_joints"]'
python3 main.py train --exp_name "Heads=4, layers=5 exlcude leg, spine" --attn_heads 4 --layers 5 --split False --exlcude_groups '["leg_joints", "spine_joints"]'
python3 main.py train --exp_name "Heads=2, layers=5, exlcude leg, thorax, spine" --attn_heads 2 --layers 5 --split False --exlcude_groups '["leg_joints", "thorax_joints", "spine_joints"]'
python3 main.py train --exp_name "Heads=2, layers=5 exlcude thorax, spine" --attn_heads 2 --layers 5 --split False --exlcude_groups '["thorax_joints", "spine_joints"]'
python3 main.py train --exp_name "Heads=2, layers=5 exlcude leg, spine" --attn_heads 2 --layers 5 --split False --exlcude_groups '["leg_joints", "spine_joints"]'


