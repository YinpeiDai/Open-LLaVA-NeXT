#!/bin/bash

#SBATCH --job-name=commongrid_llama3    # name
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=4                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --gres=gpu:2                 # number of gpus
#SBATCH --mem-per-gpu=40G       
#SBATCH --time=5-00:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.log      # output file name
#SBATCH --mail-user=daiyp@umich.edu
#SBATCH --mail-type=BEGIN,END

source /home/daiyp/.bashrc # change your own path
cd /home/daiyp/Commongrid/Open-LLaVA-NeXT # change your own path
micromamba activate llava  # change your own env
module load cuda/12.1.1

export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9902
export EPOCH=2
export BELIEF_SETTING=no_belief # no_belief, zeroth_belief, zeroth_and_first_belief


echo "MASTER_ADDR="$MASTER_ADDR
/bin/hostname

export SAVE_PATH=commongrid_llama3_ep${EPOCH}_bs64_${BELIEF_SETTING}_debug
export MODEL_PATH=/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Meta-Llama-3-8B-Instruct-HF

set -x

srun --jobid $SLURM_JOBID bash -c 'torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    llava/train/my_train_commongrid.py \
    --lora_enable True --lora_r 64 --lora_alpha 16  \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version llama3 \
    --data_path playground/commongrid/dataset/SFT/meta/llava_format_${BELIEF_SETTING}.json \
    --bf16 True \
    --group_by_modality_length True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
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
    --model_max_length 5120 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${SAVE_PATH} \
    --setting ${BELIEF_SETTING} '