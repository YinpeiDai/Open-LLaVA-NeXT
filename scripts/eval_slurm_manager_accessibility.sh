for LORA_R in 32
do
    for LORA_ALPHA in 8
    do
        for EPOCH in 1 2 3
        do
            for bs in 64
            do
                export MODEL_NAME=accessibility_llama3-8b-accessibility-lora${LORA_R}_alpha${LORA_ALPHA}_ep${EPOCH}_bs${bs}
                python scripts/eval_slurm_manager_accessibility.py --job-name ${MODEL_NAME} --test-file sample_test
            done
        done
    done
done