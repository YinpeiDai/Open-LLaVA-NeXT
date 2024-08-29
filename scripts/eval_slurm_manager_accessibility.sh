for LORA_R in 8 16 32
do
    LORA_ALPHA=$((LORA_R/2))
    for EPOCH in 3 4 5
    do
        for bs in 64
        do
            for lr in 1e-5 2e-5 3e-5
            do
                export MODEL_NAME=target_accessibility_llama3-8b-lora${LORA_R}_alpha${LORA_ALPHA}_ep${EPOCH}_bs${bs}_lr${lr}
                python scripts/eval_slurm_manager_accessibility.py --job-name ${MODEL_NAME} --test-file test --key target_text
            done
        done
    done
done