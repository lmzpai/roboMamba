import os

from model.vision import Vision
from model.llm import MyMambaLLM, MambaLLM
from model.vlm import LinearVLM, Projector
from model.manip import LinearManip

def create_model(vision_encoder, llm_type, types='VLM'):
    vision = Vision(vision_encoder)
    
    if llm_type.startswith('mamba-'):
        llm = MyMambaLLM(llm_type)
    else:
        assert False, f"{llm_type} is not supported"

    if types == 'VLM':
        return LinearVLM(vision, llm)
    if types == 'MANIP':
        return LinearManip(vision, llm)
    if types == 'METAWORLD':
        return LinearManip(vision, llm, head_type='metaworld')
    raise NotImplementedError
