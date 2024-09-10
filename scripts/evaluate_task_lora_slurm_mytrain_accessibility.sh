#!/bin/bash

#SBATCH --job-name=eval_testeval_accessibility_llama3    # name
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=2            # number of cores per tasks
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --mem-per-gpu=40G       
#SBATCH --time=3-00:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.log      # output file name
#SBATCH --mail-user=daiyp@umich.edu
#SBATCH --mail-type=BEGIN,END

source /home/daiyp/.bashrc
cd /home/daiyp/Open-LLaVA-NeXT
source ./scripts/setup_greatlakes.bash


/bin/hostname

set -x


export MODEL_NAME=new-target_accessibility_llama3-8b-lora32_alpha16_ep10_bs64_lr3e-5

srun --jobid $SLURM_JOBID bash -c 'python scripts/evaluate_accessibility.py --model-path checkpoints/accessibility/${MODEL_NAME} --dirname playground/accessibility_data/test_eval --key target_text --test-files test1 test2'


# filtered_review-North_Dakota_processed filtered_review-Vermont_processed filtered_review-Alaska_processed filtered_review-Wyoming_processed filtered_review-South_Dakota_processed filtered_review-Delaware_processed filtered_review-Rhode_Island_processed filtered_review-Montana_processed filtered_review-West_Virginia_processed filtered_review-Maine_processed filtered_review-District_of_Columbia_processed filtered_review-New_Hampshire_processed filtered_review-Mississippi_processed filtered_review-Nebraska_processed filtered_review-Hawaii_processed filtered_review-Idaho_processed filtered_review-Iowa_processed filtered_review-Arkansas_processed filtered_review-Connecticut_processed filtered_review-Kansas_processed filtered_review-New_Mexico_processed filtered_review-Louisiana_processed filtered_review-Kentucky_processed filtered_review-Alabama_processed filtered_review-Oklahoma_processed filtered_review-Nevada_processed filtered_review-Minnesota_processed filtered_review-Wisconsin_processed filtered_review-Utah_processed filtered_review-Maryland_processed filtered_review-Massachusetts_processed filtered_review-Indiana_processed


# filtered_review-South_Carolina_processed filtered_review-Missouri_processed filtered_review-New_Jersey_processed filtered_review-Oregon_processed filtered_review-Tennessee_processed filtered_review-Virginia_processed filtered_review-Michigan_processed filtered_review-Washington_processed filtered_review-Colorado_processed filtered_review-Illinois_processed

# filtered_review-Arizona_processed filtered_review-Ohio_processed filtered_review-Georgia_processed filtered_review-Pennsylvania_processed filtered_review-North_Carolina_processed filtered_review-New_York_processed

# filtered_review-Texas_processed filtered_review-Florida_processed filtered_review-California_processed