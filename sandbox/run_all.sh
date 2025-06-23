#!/bin/bash

# Hyperparameters
COMMON_ARGS="--attn_heads=2 --num_layers=5 --num_MLP_layers=3 --learn_rate=1e-3 --has_temporal_weights=True"

# Runs
python main.py train $COMMON_ARGS --temporal_joint_group pelvic_joints --exp_name temporal_only_pelvic
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints --exp_name temporal_only_arm
python main.py train $COMMON_ARGS --temporal_joint_group head_joints --exp_name temporal_only_head
python main.py train $COMMON_ARGS --temporal_joint_group thorax_joints --exp_name temporal_only_thorax
python main.py train $COMMON_ARGS --temporal_joint_group leg_joints --exp_name temporal_only_leg
python main.py train $COMMON_ARGS --temporal_joint_group spine_joints --exp_name temporal_only_spine
python main.py train $COMMON_ARGS --temporal_joint_group leg_joints,pelvic_joints --exp_name temporal_only_leg_pelvis
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,pelvic_joints --exp_name temporal_only_arm_pelvis
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,pelvic_joints --exp_name temporal_only_head_pelvis
python main.py train $COMMON_ARGS --temporal_joint_group pelvic_joints,thorax_joints --exp_name temporal_only_pelvis_thorax
python main.py train $COMMON_ARGS --temporal_joint_group pelvic_joints,spine_joints --exp_name temporal_only_pelvis_spine
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,head_joints --exp_name temporal_only_arm_head
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,thorax_joints --exp_name temporal_only_arm_thorax
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,leg_joints --exp_name temporal_only_arm_leg
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,spine_joints --exp_name temporal_only_arm_spine
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,thorax_joints --exp_name temporal_only_head_thorax
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,leg_joints --exp_name temporal_only_head_leg
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,spine_joints --exp_name temporal_only_head_spine
python main.py train $COMMON_ARGS --temporal_joint_group leg_joints,thorax_joints --exp_name temporal_only_leg_thorax
python main.py train $COMMON_ARGS --temporal_joint_group spine_joints,thorax_joints --exp_name temporal_only_spine_thorax
python main.py train $COMMON_ARGS --temporal_joint_group leg_joints,spine_joints --exp_name temporal_only_leg_spine
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,head_joints,pelvic_joints --exp_name temporal_only_arm_head_pelvis
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,pelvic_joints,thorax_joints --exp_name temporal_only_arm_pelvis_thorax
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,leg_joints,pelvic_joints --exp_name temporal_only_arm_leg_pelvis
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,pelvic_joints,spine_joints --exp_name temporal_only_arm_pelvis_spine
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,pelvic_joints,thorax_joints --exp_name temporal_only_head_pelvis_thorax
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,leg_joints,pelvic_joints --exp_name temporal_only_head_leg_pelvis
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,pelvic_joints,spine_joints --exp_name temporal_only_head_pelvis_spine
python main.py train $COMMON_ARGS --temporal_joint_group leg_joints,pelvic_joints,thorax_joints --exp_name temporal_only_leg_pelvis_thorax
python main.py train $COMMON_ARGS --temporal_joint_group pelvic_joints,spine_joints,thorax_joints --exp_name temporal_only_pelvis_spine_thorax
python main.py train $COMMON_ARGS --temporal_joint_group leg_joints,pelvic_joints,spine_joints --exp_name temporal_only_leg_pelvis_spine
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,head_joints,thorax_joints --exp_name temporal_only_arm_head_thorax
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,head_joints,leg_joints --exp_name temporal_only_arm_head_leg
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,head_joints,spine_joints --exp_name temporal_only_arm_head_spine
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,leg_joints,thorax_joints --exp_name temporal_only_arm_leg_thorax
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,spine_joints,thorax_joints --exp_name temporal_only_arm_spine_thorax
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,leg_joints,spine_joints --exp_name temporal_only_arm_leg_spine
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,leg_joints,thorax_joints --exp_name temporal_only_head_leg_thorax
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,spine_joints,thorax_joints --exp_name temporal_only_head_spine_thorax
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,leg_joints,spine_joints --exp_name temporal_only_head_leg_spine
python main.py train $COMMON_ARGS --temporal_joint_group leg_joints,spine_joints,thorax_joints --exp_name temporal_only_leg_spine_thorax
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,pelvic_joints,head_joints,thorax_joints --exp_name temporal_only_arm_pelvis_head_thorax
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,leg_joints,pelvic_joints,arm_joints --exp_name temporal_only_head_leg_pelvis_arm
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,head_joints,pelvic_joints,spine_joints --exp_name temporal_only_arm_head_pelvis_spine
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,leg_joints,pelvic_joints,thorax_joints --exp_name temporal_only_arm_leg_pelvis_thorax
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,pelvic_joints,spine_joints,thorax_joints --exp_name temporal_only_arm_pelvis_spine_thorax
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,leg_joints,pelvic_joints,spine_joints --exp_name temporal_only_arm_leg_pelvis_spine
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,leg_joints,pelvic_joints,thorax_joints --exp_name temporal_only_head_leg_pelvis_thorax
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,pelvic_joints,spine_joints,thorax_joints --exp_name temporal_only_head_pelvis_spine_thorax
python main.py train $COMMON_ARGS --temporal_joint_group head_joints,leg_joints,pelvic_joints,spine_joints --exp_name temporal_only_head_leg_pelvis_spine
python main.py train $COMMON_ARGS --temporal_joint_group leg_joints,pelvic_joints,spine_joints,thorax_joints --exp_name temporal_only_leg_pelvis_spine_thorax
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,head_joints,leg_joints,thorax_joints --exp_name temporal_only_arm_head_leg_thorax
python main.py train $COMMON_ARGS --temporal_joint_group arm_joints,head_joints,spine_joints,thorax_joints --exp_name temporal_only_arm_head_spine_thorax


