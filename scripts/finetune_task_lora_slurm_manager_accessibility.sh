# change DATA_PATH

port=9902

for LORA_R in 32
do
    for LORA_ALPHA in 8
    do
        for EPOCH in 2 3
        do
            for bs in 32 64
            do
                export MODEL_NAME=accessibility_llama3-8b-accessibility-lora${LORA_R}_alpha${LORA_ALPHA}_ep${EPOCH}_bs${bs}
                python scripts/finetune_task_lora_slurm_manager_accessibility.py --job-name ${MODEL_NAME} --port ${port} --lora-r ${LORA_R} --lora-alpha ${LORA_ALPHA} --epoch ${EPOCH} --batch-size ${bs}

                port=$((port+1))
            done
        done
    done
done