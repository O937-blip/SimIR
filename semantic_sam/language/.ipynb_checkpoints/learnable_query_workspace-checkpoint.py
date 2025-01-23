import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from .registry import register_model
from ..utils import configurable
from .LangEncoder import build_tokenizer, build_lang_encoder
from utils.prompt_engineering import prompt_engineering, get_prompt_templates

class LEARNQ(nn.Module):

    @configurable
    def __init__(
            self,
            num_queries: int,
            hidden_dim: int
    ):
        super().__init__()
        # seg
        self.query_feat_decoder = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed_decoder = nn.Embedding(num_queries, hidden_dim)
        self.query_feat_fpn = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed_fpn = nn.Embedding(num_queries, hidden_dim)
        self.query_feat_encoder = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed_encoder = nn.Embedding(num_queries, hidden_dim)

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        dec_cfg = cfg['MODEL']['DECODER']
        ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
        ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
        return ret

    def update_parameter(self, query_feat, query_embed):
        self.query_feat.weight = query_feat
        self.query_embed.weight =query_embed

    def get_encoder(self):
        return self.query_feat_encoder, self.query_embed_encoder
    def get_fpn(self):
        return self.query_feat_fpn, self.query_embed_fpn
    def get_decoder(self):
        return self.query_feat_decoder, self.query_embed_decoder

@register_model
def get_language_model(cfg, **kwargs):
    return LEARNQ(cfg)