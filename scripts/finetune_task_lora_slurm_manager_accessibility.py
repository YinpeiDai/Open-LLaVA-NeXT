import argparse
import os
import subprocess

class SlurmManager:
    def __init__(self):
        self.dirname = os.path.dirname(os.path.realpath(__file__))
        parser = self.make_args()
        args = parser.parse_args()
        self.args = args    
        self.execute()
        
        
    
    def make_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--job-name", type=str, required=True)                
        parser.add_argument("--account", type=str, default="chaijy2")
        parser.add_argument("--partition", type=str, default="spgpu")
        parser.add_argument("--nodes", type=int, help="if hetero setting, set as -1", default=2)
        parser.add_argument("--ntasks-per-node", type=int, default=1)
        parser.add_argument("--cpus-per-task", type=int, default=4)
        parser.add_argument("--gres", type=str, default="gpu:2")
        parser.add_argument("--mem-per-gpu", type=str, default="20G")
                
        parser.add_argument("--time", type=str, default="01:00:00")
        parser.add_argument("--output", type=str, default=f"logs/%x-%j.log")
        parser.add_argument("--mail-user", type=str, default="daiyp@umich.edu")
        parser.add_argument("--mail-type", type=str, default="BEGIN,END")

        parser.add_argument("--port", type=int, default=9902)
        parser.add_argument("--lora-r", type=int, default=32)
        parser.add_argument("--lora-alpha", type=int, default=8)
        parser.add_argument("--epoch", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--data-path", type=str, default="/nfs/turbo/coe-chaijy-unreplicated/llama3-8b-accessibility")
        
        return parser
        

    def make_script(self):
        script = f"""#!/bin/bash
        
#SBATCH --job-name=train-{self.args.job_name}
#SBATCH --account={self.args.account}
#SBATCH --partition={self.args.partition}
#SBATCH --time={self.args.time}  
#SBATCH --nodes={self.args.nodes}
#SBATCH --ntasks-per-node={self.args.ntasks_per_node}
#SBATCH --gres={self.args.gres}
#SBATCH --cpus-per-task={self.args.cpus_per_task}  
#SBATCH --mem-per-gpu={self.args.mem_per_gpu}
#SBATCH --output={self.args.output}
#SBATCH --mail-user={self.args.mail_user}
#SBATCH --mail-type={self.args.mail_type}



source /home/daiyp/.bashrc
cd /home/daiyp/Open-LLaVA-NeXT
source ./scripts/setup_greatlakes.bash

export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export MASTER_PORT={self.args.port}
export LORA_R={self.args.lora_r}
export LORA_ALPHA={self.args.lora_alpha}
export EPOCH={self.args.epoch}
export ACCU={self.args.batch_size//4}
export bs={self.args.batch_size}
export LR={self.args.lr}
export SAVE_PATH={self.args.job_name}
echo "SAVE_PATH="$SAVE_PATH
export MODEL_PATH=/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Meta-Llama-3-8B-Instruct-HF
export DATA_PATH={self.args.data_path}

srun --jobid $SLURM_JOBID bash -c 'torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    llava/train/my_train_accessibility.py \
    --lora_enable True --lora_r $LORA_R --lora_alpha $LORA_ALPHA  \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --version llama3 \
    --data_path $DATA_PATH \
    --bf16 True \
    --group_by_modality_length True \
    --output_dir /home/daiyp/Open-LLaVA-NeXT/checkpoints/accessibility/$SAVE_PATH \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $ACCU \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 1e5 \
    --save_total_limit 1 \
    --learning_rate $LR \
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
    --run_name $SAVE_PATH'

"""
        self.script = script
    
            
    def execute(self):
        self.make_script()
        # print(self.script)
        subprocess.run(["sbatch"], input=self.script, text=True)
    

if __name__ == "__main__":
    slurm = SlurmManager()