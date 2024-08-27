#!/bin/bash

#SBATCH --job-name=accessibility_llama3    # name
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4            # number of cores per tasks
#SBATCH --gres=gpu:2                 # number of gpus
#SBATCH --mem-per-gpu=40G       
#SBATCH --time=5-00:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.log      # output file name
#SBATCH --mail-user=daiyp@umich.edu
#SBATCH --mail-type=BEGIN,END

source /home/daiyp/.bashrc
cd /home/daiyp/Open-LLaVA-NeXT
source ./scripts/setup_greatlakes.bash


export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9902


echo "MASTER_ADDR="$MASTER_ADDR
/bin/hostname

# hyper parameters
# lora_r lora_alpha epoch batch_size
#  64    16        2        64
#  64    16        1        64
#  64    16        2        32
#  64    16        1        32
#  32    8         2        64
#  32    8         1        64
#  32    8         2        32
#  32    8         1        32

# remember to change job name
export LORA_R=32
export EPOCH=1
export ACCU=16
bs=$((ACCU * 4))
export LORA_ALPHA=$((LORA_R / 4))
echo "LORA_R="$LORA_R
echo "LORA_ALPHA="$LORA_ALPHA
export SAVE_PATH=accessibility_llama3-8b-accessibility-lora${LORA_R}_ep${EPOCH}_bs${bs}
export MODEL_PATH=/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Meta-Llama-3-8B-Instruct-HF
export DATA_PATH=/home/daiyp/Open-LLaVA-NeXT/playground/accessibility_data/sample_train_llava_format.json

set -x

srun --jobid $SLURM_JOBID bash -c 'torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    llava/train/my_train_accessibility.py \
    --lora_enable True --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA}  \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version llama3 \
    --data_path $DATA_PATH \
    --bf16 True \
    --group_by_modality_length True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${ACCU} \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 1e5 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.08 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1280 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH}'