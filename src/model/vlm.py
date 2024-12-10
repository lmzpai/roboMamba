from functools import partial
import os
import time

import torch
from torch import nn
from torch.distributed.fsdp.wrap import _or_policy, transformer_auto_wrap_policy, _module_wrap_policy, lambda_auto_wrap_policy
from transformers.models.mamba.modeling_mamba import MambaBlock
from timm.models.vision_transformer import Block, VisionTransformer
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

from model.my_mamba.modeling_mamba import MyMambaBlock
from model.llm import MyMambaLLM, MambaLLM
from model.vision import Vision
from model.my_mamba.modeling_mamba import GenerationMixin

class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.GELU(),
            nn.Linear(input_size * 4, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size)
        )

    def forward(self, x):
        return self.projector(x)

class LinearVLM(nn.Module):
    def __init__(self, vision_encoder, llm, projector='None'):
        super(LinearVLM, self).__init__()
        self.vision:Vision = vision_encoder
        self.llm = llm
        self.projector:Projector = Projector(self.vision.hidden_size, self.llm.hidden_size) 
        if isinstance(self.llm, MyMambaLLM):
            self.config = self.llm.mamba.config

    def lora(self):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["in_proj", "dt_proj", "x_proj", "out_proj"],
        )
        self.llm.mamba = get_peft_model(self.llm.mamba, peft_config)

    def encode(self, vision, text, multi_modal=None):
        text_encoded = self.llm.embed(text)
        if multi_modal is None:
            multi_modal = torch.count_nonzero(vision, dim=[-1, -2, -3]) > 0     
        vision_encoded = self.vision(vision)
        vision_encoded_result = self.projector(vision_encoded)[multi_modal]
        shape = list(text_encoded.shape)
        shape[1] += vision_encoded_result.shape[1]
        text_result = torch.zeros(shape,dtype=vision.dtype,device=vision_encoded_result.device)
        text_result[multi_modal] = torch.cat([vision_encoded_result, text_encoded[multi_modal]], dim=1)
        if not multi_modal.all():
            text_result[~multi_modal, :text_encoded.shape[1]] = text_encoded[~multi_modal]
        return text_result

    def forward(self, vision, text, multi_modal=None, labels=None, cache_params=None, position_ids=None, **kwargs):
        if vision is None:
            return self.llm.mamba(input_ids=text, **kwargs)
        text_encoded = self.encode(vision, text, multi_modal=multi_modal)
        if isinstance(self.llm, MyMambaLLM):
            res = self.llm(text_encoded, None, labels, cache_params=cache_params, position_ids=position_ids, **kwargs)
        else:
            res = self.llm(text_encoded, None, labels, cache_params=cache_params, **kwargs)
        return res

    @property
    def tokenizer(self):
        return self.llm.tokenizer

    @property
    def transform(self):
        return self.vision.transform

    @property
    def fsdp_wrapping_policy(self):
        return partial(_or_policy,
            policies=[
                partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
                partial(transformer_auto_wrap_policy, transformer_layer_cls={MambaBlock}),
                partial(transformer_auto_wrap_policy, transformer_layer_cls={MyMambaBlock}),
                partial(_module_wrap_policy, module_classes={Projector}),
            ]
        )
    
    @torch.inference_mode()
    def generate_batch(self, vision, text, multi_modal=None,  **kwargs):
        if isinstance(self.llm, MyMambaLLM):
            res = GenerationMixin.generate(self, pixel_values=vision, input_ids=text, cg=True, **kwargs)
            res = res[:,text.shape[1]:]
        else:
            mamba_result = self.encode(vision, text, multi_modal=multi_modal)
            res = self.llm.generate(mamba_result, **kwargs)
        res = self.tokenizer.batch_decode(res , skip_special_tokens=True)

        return res

    def allocate_inference_cache(self, *args, **kwargs):
        return self.llm.mamba.allocate_inference_cache(*args, **kwargs)




