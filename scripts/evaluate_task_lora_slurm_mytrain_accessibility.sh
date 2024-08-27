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

export MODEL_NAME=accessibility_llama3-8b-accessibility-lora32_alpha8_ep3_bs64

srun --jobid $SLURM_JOBID bash -c 'python scripts/evaluate_accessibility.py --model-path checkpoints/${MODEL_NAME} --test-files sample_test; python scripts/metric_accessibility.py --model-name ${MODEL_NAME} --test-files sample_test'

