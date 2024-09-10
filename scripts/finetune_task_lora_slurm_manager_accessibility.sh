# change DATA_PATH
# login3

port=10001

for LORA_R in 32
do
    LORA_ALPHA=$((LORA_R/2))
    for EPOCH in 5
    do
        for bs in 64
        do
            for lr in 1e-5 2e-5 3e-5
            do
                MODEL_NAME=original_accessibility_llama3-8b-lora${LORA_R}_alpha${LORA_ALPHA}_ep${EPOCH}_bs${bs}_lr${lr}
                python scripts/finetune_task_lora_slurm_manager_accessibility.py --job-name ${MODEL_NAME} --port ${port} --lora-r ${LORA_R} --lora-alpha ${LORA_ALPHA} --epoch ${EPOCH} --batch-size ${bs} --lr ${lr} --data-path /home/daiyp/Open-LLaVA-NeXT/playground/accessibility_data/original_train_llava_format.json
                port=$((port+1))
                # check if the file exists, if not then sleep 10 seconds
                file_name=/home/daiyp/Open-LLaVA-NeXT/checkpoints/accessibility/${MODEL_NAME}/adapter_model.safetensors
                while [ ! -f $file_name ]
                do
                    sleep 30
                done
            done 
        done
    done
done