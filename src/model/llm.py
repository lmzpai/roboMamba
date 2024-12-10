
from transformers import MambaForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from safetensors.torch import load_file
    
from .my_mamba.modeling_mamba import MyMambaForCausalLM


mamba_dict = {
    'mamba-2.8b': 'state-spaces/mamba-2.8b-hf',
    'mamba-1.4b': 'state-spaces/mamba-1.4b-hf',
    'mamba-790m': 'state-spaces/mamba-790m-hf',
    'mamba-370m': 'state-spaces/mamba-370m-hf',
    'mamba-zephyr': 'xiuyul/mamba-2.8b-zephyr'
}

    
class MyMambaLLM(nn.Module):
    def __init__(self, mamba_type):
        super(MyMambaLLM, self).__init__()
        assert mamba_type in mamba_dict, "Unknown mamba type {}".format(mamba_type)
        self.mamba_type = mamba_dict[mamba_type]
        self.mamba = MyMambaForCausalLM.from_pretrained(self.mamba_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.mamba_type)
        self.hidden_size = self.mamba.config.hidden_size
        # assert False, "MyMambaLLM is not supported"
    
    def embed(self, input_ids):
        return self.mamba.backbone.embeddings(input_ids)
    
    def forward(self, inputs_embeds, attention_mask, labels, cache_params=None, return_dict=True, **kwargs):
        return self.mamba(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, retain_dict=return_dict, **kwargs)
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
    
    def generate(self, inputs_embeds, **kwargs):
        return self.mamba.generate(inputs_embeds=inputs_embeds, use_cache=True, **kwargs)

class MambaLLM(nn.Module):
    def __init__(self, mamba_type):
        super(MambaLLM, self).__init__()
        assert mamba_type in mamba_dict, "Unknown mamba type {}".format(mamba_type)
        self.mamba_type = mamba_dict[mamba_type]
        self.mamba = MambaForCausalLM.from_pretrained(self.mamba_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.mamba_type)
        self.hidden_size = self.mamba.config.hidden_size
        
    def embed(self, input_ids):
        return self.mamba.backbone.embeddings(input_ids)

    def forward(self, inputs_embeds, attention_mask, labels, cache_params=None,return_dict=True):
        return self.mamba(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, cache_params=cache_params, retain_dict=return_dict)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def generate(self, inputs_embeds, **kwargs):
        return self.mamba.generate(inputs_embeds=inputs_embeds, use_cache=True, **kwargs)


