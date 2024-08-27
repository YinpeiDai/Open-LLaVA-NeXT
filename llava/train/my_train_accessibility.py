
import copy
from dataclasses import dataclass, field
import json
import os
import pathlib
from typing import Dict, Optional

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


system_prompt = """You are a language model trained to identify the sentiment for parking based on the following Google Maps comment. Your task is to analyze the text of each review and assign an appropriate label based on the sentiment or relevance of the content. The possible labels are:
- negative: The review expresses a negative sentiment or criticism about the parking situation.
- neutral: The review provides a factual or mixed description without strong positive or negative sentiment.
- positive: The review expresses a positive sentiment or praise about the parking situation.
- unrelated: The review is not related to parking or the given comment does not describe any attitude toward parking accessible facility
For each input, respond with the label that best describes the sentiment or relevance of the review."""


def rank0_print(*args):
    if local_rank in [0, -1, None]:
        print(*args)


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    IGNORE_INDEX = -100

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):        
        conv = Conversation(
            system="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"+system_prompt,
            roles=("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|start_header_id|>assistant<|end_header_id|>\n\n"),
            version="llama3",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|eot_id|>",
        ).copy()

        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids    
    targets = input_ids.clone()

    # Mask targets # conv.sep = <|eot_id|>
    sep = conv.sep + conv.roles[1] # '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    is_open_llava_next_llama3 = tokenizer.pad_token == "<pad>"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep) # split with '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)

            if is_open_llava_next_llama3 and i > 0:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len + 1 # add <|eot_id|>
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

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
            self.tokenizer)
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
