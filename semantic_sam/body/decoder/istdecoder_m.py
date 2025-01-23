# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All at Once
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
from torchvision.transforms.functional import to_pil_image
import logging
import sys
from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Optional, Tuple, Type
from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init
from ...transformerforsam_m import TwoWayTransformer
from .registry import register_decoder
from .modules import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from ...utils import configurable
from ...modules import PositionEmbeddingSine
from ...commonforsam import LayerNorm2d
from ...backbone.repvit11 import Conv2d_BN
class SEEMDecoder(nn.Module):

    @configurable
    def __init__(
            self,
            lang_encoder: nn.Module,
            in_channels,
            mask_classification=True,
            *,
            hidden_dim: int,
            dim_proj: int,
            num_queries: int,
            contxt_len: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            task_switch: dict,
            enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_tokens = nn.Embedding(1, 256)
        self.pe_layer = PositionEmbeddingRandom(256 // 2)
        self.transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            )

        self.output_upscaling = nn.Sequential(
            nn.Conv2d(256, 256 // 4, kernel_size=1, stride=1),
            LayerNorm2d(256 // 4),
            nn.GELU(),
            nn.Conv2d(256 // 4, 256 // 8, 3, 1, 1),
            nn.GELU(),
        )
        self.output_hypernetworks_mlps = MLP(256, 256, 256 // 8, 3)
        self.lang = lang_encoder
        self.query_embed = nn.Embedding(1, 256)
        self.query_feat = nn.Embedding(1, 256)
        self.sparse_embed = nn.Embedding(8, 256)
    @classmethod
    def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
        ret = {}

        ret["lang_encoder"] = lang_encoder
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
        ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
        ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
        ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']

        # Transformer parameters:
        ret["nheads"] = dec_cfg['NHEADS']
        ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert dec_cfg['DEC_LAYERS'] >= 1
        ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
        ret["pre_norm"] = dec_cfg['PRE_NORM']
        ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
        ret["mask_dim"] = enc_cfg['MASK_DIM']
        ret["task_switch"] = extra['task_switch']

        # attn data struct

        return ret


    def forward(self, x, mask_features,dense_prompt_embeddings, sparse_query_fpn, target_queries=None, target_vlp=None, task='seg', extra={}):
        # x is a list of multi-scale feature

        sparse_prompt_embeddings = sparse_query_fpn.permute(1,0,2)
        sparse_embed = self.sparse_embed.weight.unsqueeze(0).expand(mask_features.size(0), -1, -1)
        
        output_tokens = self.query_feat.weight.unsqueeze(0).expand(mask_features.size(0), -1, -1)
        output_embed = self.query_embed.weight.unsqueeze(0).expand(mask_features.size(0), -1, -1)
        
        
        output_tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        output_embed = torch.cat((output_embed, sparse_embed), dim=1)
        
        src = mask_features + dense_prompt_embeddings
        pos_src = self.pe_layer([src.size(-2),src.size(-1)]).unsqueeze(0).expand(src.size(0), -1, -1,-1)
        b, c, h, w = src.shape
        output_tokens, src = self.transformer(src, pos_src, output_tokens,output_embed)
        src = src.transpose(1, 2).view(b, c, h, w).contiguous()
        out = self.output_upscaling(src)
        output_tokens = self.output_hypernetworks_mlps(output_tokens[:,0,:])
        output_tokens = output_tokens.unsqueeze(1)
        b, c, h, w = out.shape
        masks = (output_tokens @ out.view(b, c, h * w)).view(b, -1, h, w).contiguous()
        return masks


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


@register_decoder
def get_seem_interface(cfg, in_channels, lang_encoder, mask_classification, extra):
    return SEEMDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
