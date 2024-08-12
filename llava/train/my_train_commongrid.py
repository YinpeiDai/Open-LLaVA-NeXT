
import copy
import json
import os
import pathlib
from typing import Dict

import tokenizers
import torch
import transformers
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.model import *
from llava.train.llava_trainer import LLaVATrainer



local_rank = None

from llava.train.train import (
    ModelArguments, DataArguments, TrainingArguments, get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    find_all_linear_names, safe_save_model_for_hf_trainer, DataCollatorForSupervisedDataset,
    format_bytes, preprocess_llama3
)

from llava.conversation import Conversation, SeparatorStyle
system_prompt = """
This is a 12x12 2D grid world where two agents collobratively accomplish tasks. 
We use the following symbols to represent the cell:
{
    '<_>': 'empty cell',
    '<agent0>': 'agent0',
    '<agent1>': 'agent1',
    '<W>': 'wall cell',
    '<X>': 'unseen cell'
}
Other objects will be given in the inputs in the form of '<object+id>'

You can take actions:
1. forward/backward/left/right: move the agent in the corresponding direction.
2. pick: pick up the object in the current cell, only is valid when agent is facing to the object.
3. exchange: exchange the object with the other agent, only is valid when the agent is facing to the other agent.

You will be specified with either agent0 or agent1 and given:
1. The task description.
2. A 3x3 local grid observed by the specified agent.
3. Possible actions that you can take.
4. Description for objects and agent states when they are observed.

Then you need to predict the correction action the specified agent should take to accomplish the task with another agent together. The complete map is not given to you, you should infer from the input information.

Be careful about the movement and the orientation of the agent in the observed grid. For example, if the specific agent is facing down, the action 'move forward' will move the agent to the cell below it, not the cell above it; if the agent is facing left, the action 'move forward' will move the agent to the cell on its left, not the cell on its right.
"""

CONV_COMMONGRID_LLAMA3_TEMPLATE = Conversation(
    system="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"+system_prompt,
    roles=("<|start_header_id|>user<|end_header_id|>\n\n",
           "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama3",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
)


def rank0_print(*args):
    if local_rank in [0, -1, None]:
        print(*args)


class MyLazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(MyLazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = self.get_len(sample)
            length_list.append(cur_len)
        return length_list
    
    def get_len(self, item): # TODO
        length = []
        for conv in item["conversations"]:
            value = conv["value"]
            value = value.replace("<", " < ")
            value = value.replace(">", " > ")
            value = value.replace("\n", " ")
            length.append(len(value.split()))
        return sum(length)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = self.get_len(sample)
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(
            sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_llama3(
            sources,
            self.tokenizer,
            has_image=False)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        return data_dict



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MyLazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))
    model_max_length_args = {}

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    if config.max_position_embeddings < training_args.model_max_length:
        rank0_print(
            f'Set the max_position_embeddings from {config.max_position_embeddings} to {training_args.model_max_length}')
        model_max_length_args.update(
            {'max_position_embeddings': training_args.model_max_length})

    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (
            torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, training_args.lora_qv_proj_only),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    
    if tokenizer.pad_token is None:
        rank0_print("Adding pad token as <|reserved_special_token_5|>")
        tokenizer.pad_token = "<|reserved_special_token_5|>"
        tokenizer.pad_token_id = 128010

    conversation_lib.default_conversation = CONV_COMMONGRID_LLAMA3_TEMPLATE

    model.config.pad_token_id = tokenizer.pad_token_id

    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    rank0_print(f"Total parameters: {format_bytes(total_params)}")
    rank0_print(f"Trainable parameters: {format_bytes(trainable_params)}")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(
                training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
