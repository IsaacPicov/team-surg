#!/bin/bash

BASE_FLAGS="--devices 1 --num_classes 3 --accelerator gpu --gradient_clip_val 0.5 --max_epochs 1000 --patience 50 --limit_train_batches 1.0 --tb_path $HOME/Desktop/AlgoverseResearch/u/isaacpicov/baseline/experiments/simulation/gnn_results/tb --loss_fn BCE --learn_rate 1e-3 --batch_size 64 --optimizer Adam --dataset_path action_dataset_joints_leg_sampled_150.pkl --proj_name hmr-gnn --dp_rate 0.1 --metrics_strategy weighted --oversample False --c_in 3 --c_hidden 128 --num_MLP_layers 3 --layer_name GAT --num_frames 150 --num_workers 4 --pin_memory True"

echo "=== Running One-Factor-at-a-Time Ablations ==="

# Temporal Ablation
# for TEMPORAL in True False; do
#     python3 main.py train $BASE_FLAGS --exp_name "abl_temporal_${TEMPORAL}" --has_temporal_weights $TEMPORAL --attn_heads 1 --num_layers 5 --split False
# done

# Attention Heads
for HEADS in 1 2 4 8; do
    python3 main.py train $BASE_FLAGS --exp_name "abl_heads_${HEADS}" --has_temporal_weights True --attn_heads $HEADS --num_layers 5 --split False
done

# Layer Depth
for LAYERS in 5 6 7 8 9; do
    python3 main.py train $BASE_FLAGS --exp_name "abl_layers_${LAYERS}" --has_temporal_weights True --attn_heads 1 --num_layers $LAYERS --split False
done

# Exclude Groups
GROUPS=("leg_joints" "thorax_joints" "spine_joints")
for GROUP in "${GROUPS[@]}"; do
    python3 main.py train $BASE_FLAGS --exp_name "abl_excl_${GROUP}" --has_temporal_weights True --attn_heads 1 --num_layers 5 --split False --exclude_groups "['$GROUP']"
done

# Split alone
SPLIT_1=4
SPLIT_2=2
SPLIT_RATIO=0.4
python3 main.py train $BASE_FLAGS --exp_name "abl_split_only" --has_temporal_weights True --attn_heads 1 --num_layers 7 --split True --split_1 $SPLIT_1 --split_2 $SPLIT_2 --split_ratio $SPLIT_RATIO

echo "=== Running Combined Ablations ==="

# Combination 1: More Layers + More Heads
for LAYERS in 7 9; do
  for HEADS in 4 8; do
    python3 main.py train $BASE_FLAGS --exp_name "comb_deep_${LAYERS}_heads_${HEADS}" --has_temporal_weights True --attn_heads $HEADS --num_layers $LAYERS --split False
  done
done

# Combination 2: Temporal + Split + Moderate Attention
SPLIT_1=5
SPLIT_2=2
SPLIT_RATIO=0.5
python3 main.py train $BASE_FLAGS --exp_name "comb_temp_split_heads" --has_temporal_weights True --attn_heads 4 --num_layers 7 --split True --split_1 $SPLIT_1 --split_2 $SPLIT_2 --split_ratio $SPLIT_RATIO

# Combination 3: No Temporal + Fewer Heads + Fewer Layers (baseline-lite)
python3 main.py train $BASE_FLAGS --exp_name "comb_lite_noTemp" --has_temporal_weights False --attn_heads 1 --num_layers 5 --split False

# Combination 4: Split + Exclusion
python3 main.py train $BASE_FLAGS --exp_name "comb_split_excl_thorax" --has_temporal_weights True --attn_heads 4 --num_layers 7 --split True --split_1 4 --split_2 2 --split_ratio 0.4 --exclude_groups "['thorax_joints']"

# Combination 5: Temporal Off + High Heads (stress test)
python3 main.py train $BASE_FLAGS --exp_name "comb_noTemp_highHeads" --has_temporal_weights False --attn_heads 8 --num_layers 6 --split False

echo "=== All ablation jobs queued ==="
