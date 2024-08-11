#!/bin/bash


#SBATCH --job-name=llava-llama3-rvt-lora-realrobot    # name
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16            # number of cores per tasks
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --mem-per-gpu=40G       
#SBATCH --time=5-00:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.log      # output file name
#SBATCH --mail-user=daiyp@umich.edu
#SBATCH --mail-type=BEGIN,END

source /home/daiyp/.bashrc
cd /home/daiyp/Open-LLaVA-NeXT
source ./scripts/setup_greatlakes.bash


export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9905
export EPOCH=3

echo "GPU_PER_NODE="$GPUS_PER_NODE
echo "MASTER_PORT="$MASTER_PORT
echo "EPCH="$EPOCH
echo "MASTER_ADDR="$MASTER_ADDR

export SAVE_PATH=llava_llama3_rvt_realrobot_ep${EPOCH}_bs64_nodup_v2
export MODEL_PATH=/scratch/chaijy_root/chaijy2/daiyp/llama3-llava-next-8b

set -x

srun --jobid $SLURM_JOBID bash -c 'torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    llava/train/my_train_realrobot.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version llava_llama_3_rvt \
    --data_path /home/daiyp/Open-LLaVA-NeXT/playground/rvt_llava_data_real_robot/all_tasks.json \
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
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 1e5 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} \
    --predict_failure_label True'