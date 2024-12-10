from functools import partial
import os
import time

import torch
from torch import nn
import torch.nn.init as init
from torch.distributed.fsdp.wrap import _or_policy, transformer_auto_wrap_policy, _module_wrap_policy, lambda_auto_wrap_policy
import torch.nn.functional as F
from transformers.models.mamba.modeling_mamba import MambaBlock
from timm.models.vision_transformer import Block
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

from .llm import MyMambaLLM, MambaLLM
from .vision import Vision
from .vlm import Projector

class SpecialMLP(nn.Module):
    def __init__(self,inp,oup):
        super(SpecialMLP, self).__init__()
        self.fc1 = nn.Linear(inp, inp//2) 
        self.fc2 = nn.Linear(inp//2, inp//4)  
        self.fc3 = nn.Linear(inp//4, oup, bias=False)
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x) 
        x = torch.squeeze(x, 1)
        return x
    
class LinearManip(nn.Module):
    def __init__(self, vision_encoder, llm, head_type='two_mlp', ):
        super(LinearManip, self).__init__()
        self.vision:Vision = vision_encoder
        self.llm = llm
        self.head_type = head_type
        self.projector = Projector(self.vision.hidden_size, self.llm.hidden_size)
        if head_type == 'mlp':
            self.action_head = SpecialMLP(llm.hidden_size, 2+6)
        elif head_type == 'two_mlp':
            self.action_head1 = SpecialMLP(llm.hidden_size, 2)
            self.action_head2 = SpecialMLP(llm.hidden_size, 6)
            for m in self.action_head1.modules():  
                if isinstance(m, nn.Linear):  
                    init.xavier_uniform_(m.weight)  
                    if m.bias is not None:  
                        init.constant_(m.bias, 0)
            for m in self.action_head2.modules():  
                if isinstance(m, nn.Linear):  
                    init.xavier_uniform_(m.weight)  
                    if m.bias is not None:  
                        init.constant_(m.bias, 0)

        elif head_type == 'ssm+mlp':
            self.ssm1 = MambaBlock(llm.mamba.config, -1)
            self.ssm2 = MambaBlock(llm.mamba.config, -1)
            self.action_head1 = SpecialMLP(llm.hidden_size, 2)
            self.action_head2 = SpecialMLP(llm.hidden_size, 6)
        else:
            assert False, f"Unknown head type: {head_type}"
        self.pool = nn.AdaptiveAvgPool1d(1)        
        if isinstance(self.llm, (MyMambaLLM, MambaLLM)):
            ori_weight = self.llm.mamba.lm_head.weight
            self.llm.mamba.lm_head = nn.Identity()
            self.llm.mamba.lm_head.weight = nn.Parameter(torch.zeros_like(ori_weight))

    def forward(self, vision, text, labels=None, cache_params=None, state=None):
        text_encoded = self.llm.embed(text)
        vision_encoded = self.vision(vision)

        vision_encoded = self.projector(vision_encoded)
        text_result = torch.cat([vision_encoded, text_encoded], dim=1)
        res = self.llm(text_result, return_dict=False, labels=labels, cache_params=cache_params, attention_mask=None).logits

        if self.head_type == 'ssm+mlp':
            res1 = self.ssm1(res)
            res1 = self.pool(res1.permute(0, 2, 1)).squeeze(-1)
            res2 = self.ssm2(res)
            res2 = self.pool(res2.permute(0, 2, 1)).squeeze(-1)
        else:
            res = (res[:,vision_encoded.shape[1]]+res[:,-1])/2
            
        if self.head_type == 'mlp':
            res = self.action_head(res)
        elif self.head_type == 'two_mlp':
            res1 = self.action_head1(res)
            res2 = self.action_head2(res)
            res = torch.cat([res1, res2], dim=1)
        elif self.head_type == 'ssm+mlp':
            res1 = self.action_head1(res1)
            res2 = self.action_head2(res2)
            res = torch.cat([res1, res2], dim=1)
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
                partial(transformer_auto_wrap_policy, transformer_layer_cls={MambaBlock})
            ]
        )
# 6D-Rot loss
# input sz bszx6
def loss_6d_rot(pred_6d, gt_6d):

    def bgs(d6s):
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
        a2 = d6s[:, :, 1]
        b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

    # batch geodesic loss for rotation matrices
    def bgdR(Rgts, Rps):
        Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
        Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
        return torch.acos(theta)
    
    pred_Rs = bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    gt_Rs = bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    theta = bgdR(gt_Rs, pred_Rs)
    return theta.mean()