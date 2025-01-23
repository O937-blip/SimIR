from .registry import model_entrypoints
from .registry import is_model


def build_decoder(config, *args, **kwargs):
    model_name = config['MODEL']['DECODER']['NAME']
    if model_name == 'nodecoder':
        return None
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, *args, **kwargs)

def build_s_decoder(config, *args, **kwargs):
    model_name = config['MODEL']['DECODER']['s_NAME']
    if model_name == 'nodecoder':
        return None
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, *args, **kwargs)