#!/bin/bash
set -x

# wandb login

export GPUS_PER_NODE=2
export CUDA_VISIBLE_DEVICES=0,1
EPOCH=2

SAVE_PATH=llava_llama3_rvt_alltask_lora_debug_ep${EPOCH}
MODEL_PATH=/data/daiyp/foundation_models/llama3-llava-next-8b
# MODEL_PATH=/scratch/chaijy_root/chaijy2/daiyp/llama3-llava-next-8b



torchrun --nnodes 1 --nproc_per_node $GPUS_PER_NODE --node_rank 0 --master_addr localhost --master_port 29504 \
    llava/train/my_train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version llava_llama_3_rvt \
    --data_path /home/daiyp/Open-LLaVA-NeXT/playground/rvt_llava_data/all_tasks.json \
    --image_folder /home/daiyp/Open-LLaVA-NeXT \
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
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
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
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} \
    --predict_failure_label True
