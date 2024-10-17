#!/bin/bash
set -x

# wandb login

export GPUS_PER_NODE=2
export CUDA_VISIBLE_DEVICES=0,1
EPOCH=1 # 1 or 2 is better, do not use large number

SAVE_PATH=commongrid_llama3-8b-no-belief-debug
MODEL_PATH=/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Meta-Llama-3-8B-Instruct-HF
SETTING=none # none, zeroth, first
DATA_PATH=/home/daiyp/CommonGrid/Open-LLaVA-NeXT/playground/llava_format_multi_doors_one_room_${SETTING}_belief_v1.json

torchrun --nnodes 1 --nproc_per_node $GPUS_PER_NODE --node_rank 0 --master_addr localhost --master_port 29504 \
    llava/train/my_train_commongrid.py \
    --lora_enable True --lora_r 64 --lora_alpha 16 --lora_dropout 0.05 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version llama3 \
    --data_path ${DATA_PATH} \
    --bf16 True \
    --group_by_modality_length True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1e5 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.08 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5120 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} \
    --setting ${SETTING}