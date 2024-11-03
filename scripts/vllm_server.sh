
module load cuda/12.1.1
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Llama-3.1-70B-Instruct --dtype auto --enforce-eager --gpu-memory-utilization 0.95  --max-model-len 512 --tensor-parallel-size 4