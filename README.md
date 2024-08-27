# Open-LLaVA-NeXT

## New notes

```
# download open-llava-next submodule for llama/llava training 
git submodule update --init --recursive

# install (`module load cuda/12.1.1` if you install on GreatLakes)
cd Open-LLaVA-NeXT
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install tensorboardX

# check if llava-next is installed
pip list | grep llava

# check whether branch  is `commongrid`
git branch

# link data and checkpoint (you can choose your own path)
ln -s /nfs/turbo/coe-chaijy-unreplicated/daiyp/llava_rvt_checkpoints/commongrid_ckpt/ checkpoints
mkdir playground && cd playground
ln -s /nfs/turbo/coe-chaijy-unreplicated/roihn/commongrid commongrid
```


### Deploy

For server code, refer to `CommonGrid/Open-LLaVA-NeXT/deploy/llama3_server.py`
```
cd Open-LLaVA-NeXT
CUDA_VISIBLE_DEVICES=0 python deploy/llama3_server.py --model-path <lora_model_save_path> -model-base /nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Meta-Llama-3-8B-Instruct-HF --model-name llama3_lora --port <port_num>
```

Current trained ckpts for <lora_model_save_path>:
1. `/home/daiyp/CommonGrid/Open-LLaVA-NeXT/checkpoints/commongrid_llama3_ep2_bs64_no_belief` 
2. `/home/daiyp/CommonGrid/Open-LLaVA-NeXT/checkpoints/commongrid_llama3_ep2_bs64_zeroth_belief`
3. `/home/daiyp/CommonGrid/Open-LLaVA-NeXT/checkpoints/commongrid_llama3_ep2_bs64_zeroth_and_first_belief` 

port number can be manually specified, ip can be seen with `hostname -I` (the first address)


For client code, refer to `CommonGrid/Open-LLaVA-NeXT/deploy/llama3_client*.py`


### Training
#### Cook data for llava format
```
cd Open-LLaVA-NeXT
python gather_commongrid_data
```

#### Local debug with 1 GPU
```
cd Open-LLaVA-NeXT
./scripts/finetune_task_lora_local_mytrain_commongrid.sh 
```

#### GreatLakes multi-GPU training
remember set `module load cuda/12.1.1` inside the script
```
cd Open-LLaVA-NeXT
sbatch ./scripts/finetune_task_lora_slurm_mytrain_commongrid.sh
```
You will see `logs` in the directory

Notes:
1. `per_device_train_batch_size` is better to set as 1, use `gradient_accumulation_steps` and multiple GPUs to enlarge the total batch size
2. Deepspeed zero2 is better then zero3 for current experiments

### Evaluation

#### Local debug with 1 GPU
```
cd ArCHer
sbatch ./scripts/eval_llama3_localmodel.py
```
#### GreatLakes 

```
cd ArCHer
sbatch ./scripts/eval_llama3_localmodel_slurm.sh
```
`logs` is under the `ArCHer` dir







## 💡 Highlights

An open-source implementation of **LLaVA-NeXT** series for facilitating the large multi-modal model community.

- 🔥 All training data and checkpoints at each stage are open-sourced, friendly for research usage.
- 🔥 Able to reproduce the results of **[LLaVA-NeXT](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)**.
- 🔥 Based on the **[LLaVA](https://github.com/haotian-liu/LLaVA)** codebase with minimal modification, easy to follow.

## 🤖 Model Zoo

See more details in [ModelZoo.md](docs/ModelZoo.md).

| Name | ViT | LLM | Weights | MME | SEED | SQA | MMB | MMB-CN | TextVQA | GQA |
|---|---|---|---|---|---|---|---|---|---|---|
| llava-next-vicuna-7b | CLIP-L-336 | Vicuna-7B | [SFT](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) | 1519 | 70.2 | 70.1 | 67.4 | 60.6 | 64.9 | 64.2 |
| open-llava-next-vicuna-7b| CLIP-L-336 | Vicuna-7B | [PT](https://huggingface.co/Lin-Chen/open-llava-next-vicuna-7b/tree/main/pretrain), [SFT](https://huggingface.co/Lin-Chen/open-llava-next-vicuna-7b) | 1540 | 71.1 | 70.7 | 68.5 | 60.7 | 67.2 | 64.3 |
| open-llava-next-llama3-8b| CLIP-L-336 | LLaMA3-8B | [PT](https://huggingface.co/Lin-Chen/open-llava-next-llama3-8b), [SFT](https://huggingface.co/Lin-Chen/open-llava-next-llama3-8b) | 1552 | 74.4 | 77.3 | 74.4 | 70.4 | 69.8 | 65.9 |


## 👨‍💻 ToDo

- [x] Reproduce LLaVA-Next-LLaMA3-8B
- [ ] Reproduce LLaVA-Next-Nous-Yi-34B
- [ ] Support SigLIP and more LLMs with various scales (Need the help from community!)
- [ ] Integrate [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for convenient evaluation

## 🔧 Install

1. Clone this repository and navigate to Open-LLaVA-NeXT folder
```bash
git clone https://github.com/xiaoachen98/Open-LLaVA-NeXT.git
cd Open-LLaVA-NeXT
```

2. Install Package
```Shell
conda create -n llava-next python=3.10 -y
conda activate llava-next
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

1. Install additional packages for training
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data Preparation

You should follow this instruction **[Data.md](docs/Data.md)** to manage the training datasets.

## Training Overview

Open-LLaVA-NeXT training consists of two stages: (1) feature alignment stage: use 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage:  finetune the entire model with 1M **completely open source** data. Detailed data statics is provided in [Visual Instruction Tuning](https://github.com/xiaoachen98/Open-LLaVA-NeXT?tab=readme-ov-file#visual-instruction-tuning). We take the Vicuna-v1.5-7B variant as example to present the training  and evaluation details.

Open-LLaVA-NeXT series are trained on A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. And utilizing DeepSpeed ZeRO-3 can further reduce the memory requirements. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Hyperparameters
We use a same set of hyperparameters as LLaVA in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Projector lr | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-LLaVA-NeXT-7B | 256 | 1e-3 | 1 | 4096 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size |  LLM lr |  Projector lr |  Vision Tower lr | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Open-LLaVA-NeXT-7B | 128 | 2e-5 | 2e-5 | 2e-6 | 1 | 4096 | 0 |

### Pretrain

Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Pretrain takes around 5 hours for Open-LLaVA-NeXT-7B on 16 x A100 (80G).

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](scripts/v1_6/train/7b/pretrain.sh).

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.

### Visual Instruction Tuning

1. Prepare data
You should follow the instructions for data preparation in [Data](docs/Data.md).
2. Prepare MLP projectors
You may download our pretrained projectors in [Model Zoo](docs/ModelZoo.md), or specify your own MLP projector after pre-training.
3. Start training
Visual instruction tuning takes around 20 hours for Open-LLaVA-NeXT-7B on 16x A100 (80G).

Training script with DeepSpeed ZeRO-2: [`finetune.sh`](scripts/v1_6/train/7b/finetune.sh).

New options to note:

- `--unfreeze_mm_vision_tower True`: finetune vision tower.
- `--mm_vision_tower_lr 2e-6`: learning rate of vision tower.
- `--image_aspect_ratio anyres`: Process an image with variable resolutions.
- `--mm_patch_merge_type spatial_unpad`: This unpads a PyTorch tensor of a padded and resized image, and by inserting learnable newline vectors into image tokens, the model becomes aware of two-dimensional spatial information. This is used to process image token.

## Evaluation

See [Evaluation.md](docs/Evaluation.md).

## ❤️ Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their brilliant contributions to the community! We just can't wait to use LLaVA-NeXT.
- [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V): Thanks for their code about finetuning the vision tower.
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): the amazing open-sourced suit for evaluating various LMMs!




## Ready for greaklakes
```
cd <Open-LLaVA-NeXT-dir>
micromanba to this env
pip install tensorboardX
git pull
git checkout dev
source setup_greatlakes.bash
ln -s /nfs/turbo/coe-chaijy-unreplicated/daiyp/augmented_data_heuristic  augmented_data_heuristic
python gather_rvt_llava_data.py
./scripts/v1_6/train/8b/finetune_task_lora_slurm_mytrain.sh
```


### GPU usage test

|stage|GPU|bs-per-GPU|peak mem|time|
|--|--|--|--|--|
|zero2|2|1|38.4|2.63s/it|
|zero2|2|2|44.1|5.1s/it|
|zero2|4|2|44.4|5.41s/it|
|zero3|2|1|42.4|3.23s/it|
|zero3|2|2|43.0|5.68s/it|
|zero3|4|2|43.1|6.98s/it|
|zero3-offload|4|2|41.5|8.47s/it|
|zero3-offload|4|4|45.3|12.73s/it|


zero2 8gpu-bs16 5.5s, 8gpu-bs16-acc2 11s
zero3 8gpu-bs16 7  
zero3-offload 4gpu-bs16 12.73, 8gpu-bs32 13s

solution: better to use zero2