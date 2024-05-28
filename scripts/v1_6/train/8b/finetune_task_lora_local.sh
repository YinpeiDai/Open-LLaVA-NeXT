#!/bin/bash
set -x

wandb login

export GPUS_PER_NODE=1
export NNODES=1
export BATCH_SIZE=4
export GRADIENT_ACCU_STEPS=1
export MASTER_PORT=29504
export CPUS_PER_TASK=4

export DATA_PATH=/home/daiyp/Open-LLaVA-NeXT/playground/data/llava_instruct_1k.json
export SAVE_PATH=llava-v1.6-8b_llama3-8b-debug
export BASE_LR=2e-5
export VIT_LR=2e-6


deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/daiyp/foundation_models/llama3-llava-next-8b \
    --version llava_llama_3 \
    --data_path ./playground/data/llava_instruct_1k.json \
    --image_folder ./playground/data/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type spatial_unpad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/llava-v1.6-8b_llama3-8b-debug \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 7975 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 6144 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${SAVE_PATH}'