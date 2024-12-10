import timm
from torch import nn
from torchvision.transforms import Compose, Resize

vision_encoders = {
    'SIGLIP256': 'vit_large_patch16_siglip_256',
    'SIGLIP384': 'vit_large_patch16_siglip_384',
    'CLIP224': 'vit_large_patch14_clip_224.openai',  
    'CLIP336': 'vit_large_patch14_clip_336.openai'
}

img_size ={
    'SIGLIP256': 256,
    'SIGLIP384': 384,
    'CLIP224': 224,
    'CLIP336': 336
}


class Vision(nn.Module):
    def __init__(self, encoder_type):
        super(Vision, self).__init__()
        assert encoder_type in vision_encoders.keys(), 'Unknown encoder {}'.format(encoder_type)
        pretrain_cfg_overlay = None
        if encoder_type.startswith("CLIP"):
            self.model = timm.create_model(vision_encoders[encoder_type], 
                                        pretrained=True, 
                                        num_classes=0, 
                                        pretrained_cfg_overlay=pretrain_cfg_overlay,
                                        act_layer="quick_gelu")
        else:
            self.model = timm.create_model(vision_encoders[encoder_type], 
                                        pretrained=True, 
                                        num_classes=0, 
                                        pretrained_cfg_overlay=pretrain_cfg_overlay)


        data_config = timm.data.resolve_model_data_config(self.model)
        data_config['input_size'] = (3, img_size[encoder_type], img_size[encoder_type])

        image_transform = timm.data.create_transform(**data_config, is_training=False).transforms
        self.transform = Compose(
            [
                Resize(img_size[encoder_type], interpolation=image_transform[0].interpolation),
                *image_transform[1:]
            ]
        )
        self.hidden_size = self.model.embed_dim
        self.patch_size = self.model.patch_embed.num_patches

    def forward(self, x):
        x = self.model.get_intermediate_layers(x, n={len(self.model.blocks) - 2})[0]
        return x