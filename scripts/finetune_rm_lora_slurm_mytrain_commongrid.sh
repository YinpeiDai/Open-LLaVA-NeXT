#!/bin/bash

#SBATCH --job-name=rm_commongrid_llama3_none_v3    # name
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=4                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --gres=gpu:2                 # number of gpus
#SBATCH --mem-per-gpu=40G       
#SBATCH --time=2-00:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.log      # output file name
#SBATCH --mail-user=roihn@umich.edu
#SBATCH --mail-type=BEGIN,END

source /home/roihn/.bashrc
cd /gpfs/accounts/chaijy_root/chaijy2/roihn/CommonGrid/Open-LLaVA-NeXT # change your own path
# micromamba activate commongrid  # change your own env
# module load python3.10-anaconda
# source activate base
conda activate /home/roihn/miniconda3/envs/grid
module load cuda/12.1.1

export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9902
export EPOCH=1


echo "MASTER_ADDR="$MASTER_ADDR
/bin/hostname

export BELIEF_SETTING=none # none, zeroth, first
# export DATA_RATIO=0.25
export SAVE_PATH=commongrid_llama3_ep1_bs64_rm_${BELIEF_SETTING}_pickandopen_3k_debug_v3 # change the save path yourself
export MODEL_PATH=/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Meta-Llama-3-8B-Instruct-HF
export DATA_PATH=playground_replicated/dataset/RM/llava_format_${BELIEF_SETTING}_belief_v3_51.json
# export DATA_PATH=playground_replicated/dataset/SFT/llava_format_pickandopen_${BELIEF_SETTING}_belief_v2_${DATA_RATIO}.json

set -x

srun --jobid $SLURM_JOBID bash -c 'torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    llava/train/my_train_commongrid_rm.py \
    --lora_enable True --lora_r 64 --lora_alpha 16  \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version llama3 \
    --data_path ${DATA_PATH} \
    --bf16 True \
    --group_by_modality_length True \
    --output_dir mycheckpoint_replicated/${SAVE_PATH} \
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
