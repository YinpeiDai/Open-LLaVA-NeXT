#!/bin/bash

#SBATCH --job-name=eval_accessibility_llama3    # name
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=2            # number of cores per tasks
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --mem-per-gpu=40G       
#SBATCH --time=10-00:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.log      # output file name
#SBATCH --mail-user=daiyp@umich.edu
#SBATCH --mail-type=BEGIN,END

source /home/daiyp/.bashrc
cd /home/daiyp/Open-LLaVA-NeXT
source ./scripts/setup_greatlakes.bash


/bin/hostname


export MODEL_NAME=target_accessibility_llama3-8b-lora32_alpha16_ep5_bs64_lr3e-5

srun --jobid $SLURM_JOBID bash -c 'python scripts/evaluate_accessibility.py --model-path checkpoints/accessibility/${MODEL_NAME} --dirname playground/accessibility_data/test_eval --key target_text --test-files test1 test2'

