from .registry import model_entrypoints
from .registry import is_model
from .sam_prompt_encoder_modified import *
from .transformer_encoder_fpn import *
from .encoder_deform import *
from .vitdet import *
from .sam_prompt_encoder import *
from .transformer_encoder_fpn_workspace import *
from .transformer_encoder_fpn_ import *
from .transformer_encoder_fpn_test import *
def build_encoder(config, *args, **kwargs):
    model_name = config['MODEL']['ENCODER']['NAME']
    if model_name == 'noencoder':
        return None
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, *args, **kwargs)

def build_s_encoder(config, *args, **kwargs):
    model_name = config['MODEL']['ENCODER']['s_NAME']
    if model_name == 'noencoder':
        return None
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, *args, **kwargs)