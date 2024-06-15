"""
Adapt llava training for rvt data training
"""

import copy
import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import tokenizers
import torch
import transformers
from packaging import version
from PIL import Image
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                             IMAGE_TOKEN_INDEX)
from llava.mm_utils import process_anyres_image, tokenizer_image_token
from llava.model import *
from llava.train.llava_trainer import LLaVATrainer

from peft.peft_model import PeftModel
from llava.utils import RLBENCH_TASKS


local_rank = None

from llava.train.train import (
    ModelArguments, DataArguments, TrainingArguments, get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    find_all_linear_names, safe_save_model_for_hf_trainer, preprocess_multimodal, DataCollatorForSupervisedDataset,
    format_bytes
)

def rank0_print(*args):
    if local_rank in [0, -1, None]:
        print(*args)


@dataclass
class MyDataArguments(DataArguments):
    predict_failure_label: bool = field(
        default=False, metadata={"help": "Predict the failure label."}
    )
    predict_heuristic: bool = field(
        default=False, metadata={"help": "Predict the heuristic language."}
    )
    is_resume: bool = field(
        default=False, metadata={"help": "Resume from the checkpoint."}
    )



def return_sentence(sentence, args):
    TEMP_0_nolabel = "<image>\nThe task goal is: {task_goal}. This is the first step and the robot is about to start the task. Based on the visual observation and the context, what's the next instruction for the robot arm?"
    TEMP_nolabel = "<image>\nThe task goal is: {task_goal}. In the previous step, the robot arm was given the following instruction: \"{previous_instruction}\". {robot_delta_state} Based on the visual observation and the context, what's the next instruction for the robot arm?"
    TEMP_0_label = "<image>\nThe task goal is: {task_goal}. This is the first step and the robot is about to start the task. Based on the visual observation and the context, how does the robot fulfil that previous instruction and what's the next instruction for the robot arm?"
    TEMP_label = "<image>\nThe task goal is: {task_goal}. In the previous step, the robot arm was given the following instruction: \"{previous_instruction}\". {robot_delta_state} Based on the visual observation and the context, how does the robot fulfil that previous instruction and what's the next instruction for the robot arm?"
    if sentence["from"] == "human":
        if "robot_delta_state" not in sentence: # first step
            if args.predict_failure_label:
                return TEMP_0_label.format(task_goal=sentence["task_goal"])
            else:
                return TEMP_0_nolabel.format(task_goal=sentence["task_goal"])
        else:
            if args.predict_failure_label:
                return TEMP_label.format(task_goal=sentence["task_goal"], previous_instruction=sentence["previous_instruction"], robot_delta_state=sentence["robot_delta_state"])
            else:
                return TEMP_nolabel.format(task_goal=sentence["task_goal"], previous_instruction=sentence["previous_instruction"], robot_delta_state=sentence["robot_delta_state"])
    else:
        if args.predict_failure_label:
            if args.predict_heuristic:
                return sentence["heuristic_instruction"]
            else:
                return sentence["gpt_instruction"]
        else:
            if args.predict_heuristic:
                return sentence["heuristic_instruction_no_label"]
            else:
                return sentence["gpt_instruction_no_label"]

def preprocess_llama3_rvt(
    sources,
    args,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    task_name: Optional[str] = None
) -> Dict:
    if task_name is None:
        assert conversation_lib.default_conversation.version == "llama3_rvt"
    else:
        assert task_name in RLBENCH_TASKS
        assert conversation_lib.default_conversation.version == f"llama3_rvt_{task_name}"
    conv = conversation_lib.default_conversation.copy()        
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, return_sentence(sentence, args))
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

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

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer))
            else:
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
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split())
                               for conv in sample['conversations']) + img_tokens)
            length_list.append(3200)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # cur_len = sum(len(conv['value'].split())
            #               for conv in sample['conversations'])
            # cur_len = cur_len if 'image' in sample else -cur_len
            cur_len = 3200
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(
            sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(
                image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255)
                                      for x in processor.image_mean))
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')[ #(640, 480)
                    'pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "anyres":
                image_size = image.size
                image = process_anyres_image(       # torch.Size([5, 3, 336, 336])
                    image, processor, self.data_args.image_grid_pinpoints)
            else:
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')[
                    'pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_llama3_rvt(
            sources,
            self.data_args,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(
                3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        return data_dict  # final length is 2928 + len(input_ids) - 1



class MyLLaVATrainer(LLaVATrainer):
    
    def _save_checkpoint(self, model, trial, metrics=None):
        super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
        # if self.args.lora_enable:
        #     state_dict = get_peft_state_maybe_zero_3(
        #         model.named_parameters(), self.args.lora_bias
        #     )
        #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        #         model.named_parameters()
        #     )
        #     outdir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(self.state.global_step), "lora_model_output")
        #     os.makedirs(outdir, exist_ok=True)
        #     if self.args.local_rank == 0 or self.args.local_rank == -1:
        #         if isinstance(model.config, dict):
        #             with open(os.path.join(outdir, 'config.json'), 'w') as f:
        #                 json.dump(model.config, f)
        #         else:
        #             model.config.save_pretrained(outdir)
        #         model.save_pretrained(
        #             outdir, state_dict=state_dict)
        #         torch.save(non_lora_state_dict, os.path.join(
        #             outdir, 'non_lora_trainables.bin'))



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


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, MyDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    model_max_length_args = {}
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    if config.max_position_embeddings < training_args.model_max_length:
        rank0_print(
            f'Set the max_position_embeddings from {config.max_position_embeddings} to {training_args.model_max_length}')
        model_max_length_args.update(
            {'max_position_embeddings': training_args.model_max_length})

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **model_max_length_args
    )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

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
        # rank0_print("Adding pad token as '<pad>'")
        # smart_tokenizer_and_embedding_resize(
        #     special_tokens_dict=dict(pad_token="<pad>"),
        #     tokenizer=tokenizer,
        #     model=model,
        # )
        rank0_print("Adding pad token as <|reserved_special_token_5|>")
        tokenizer.pad_token = "<|reserved_special_token_5|>" # set this as pad_token for llava-next-llama3-model
        tokenizer.pad_token_id = 128010
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version]
    else:
        raise ValueError(f"Unsupported version: {model_args.version}")

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_aspect_ratio == 'anyres':
            base_size = vision_tower.config.image_size
            grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [
                [g[0]*base_size, g[1]*base_size] for g in grids]
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower

        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.get_model().parameters())
        trainable_params = sum(
            p.numel() for p in model.get_model().parameters() if p.requires_grad)

        rank0_print(f"Total parameters: {format_bytes(total_params)}")
        rank0_print(f"Trainable parameters: {format_bytes(trainable_params)}")


        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        model.config.pad_token_id = tokenizer.pad_token_id


    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = MyLLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and data_args.is_resume:
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
