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
        parser.add_argument("--nodes", type=int, help="if hetero setting, set as -1", default=1)
        parser.add_argument("--ntasks-per-node", type=int, default=1)
        parser.add_argument("--cpus-per-task", type=int, default=2)
        parser.add_argument("--gres", type=str, default="gpu:1")
        parser.add_argument("--mem-per-gpu", type=str, default="20G")
                
        parser.add_argument("--time", type=str, default="01:00:00")
        parser.add_argument("--output", type=str, default=f"logs/%x-%j.log")
        parser.add_argument("--mail-user", type=str, default="daiyp@umich.edu")
        parser.add_argument("--mail-type", type=str, default="BEGIN,END")

        parser.add_argument("--test-file", type=str, required=True)
        
        return parser
        

    def make_script(self):
        script = f"""#!/bin/bash
        
#SBATCH --job-name=eval-{self.args.job_name}
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


export MODEL_NAME={self.args.job_name}
export TEST_FILE={self.args.test_file}
srun --jobid $SLURM_JOBID bash -c 'python scripts/evaluate_accessibility.py --model-path /home/daiyp/Open-LLaVA-NeXT/checkpoints/$MODEL_NAME --test-files $TEST_FILE --dirname /home/daiyp/Open-LLaVA-NeXT/playground/accessibility_data; python scripts/metric_accessibility.py --model-name $MODEL_NAME --test-files $TEST_FILE'
"""
        self.script = script
    
            
    def execute(self):
        self.make_script()
        # print(self.script)
        subprocess.run(["sbatch"], input=self.script, text=True)
    

if __name__ == "__main__":
    slurm = SlurmManager()