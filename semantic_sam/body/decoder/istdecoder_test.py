# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All at Once
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import logging
import sys
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init

from .registry import register_decoder
from .modules import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from ...utils import configurable
from ...modules import PositionEmbeddingSine


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
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.contxt_len = contxt_len
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()


        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        '''self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)'''
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.task_switch = task_switch
        self.query_index = {}
        # output FFNs
        self.lang_encoder = lang_encoder
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.n = 0

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

    def prepare_features(self, x, num_feature_levels, pe_layer, input_proj, level_embed):
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        for i in range(num_feature_levels):
            size_list.append(x[i].shape[-2:])

            pos.append(pe_layer(x[i], None).flatten(2))
            src.append(input_proj[i](x[i]).flatten(2) + level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        return src, pos, size_list

    def forward(self, x, mask_features,image, mask=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels;
        del mask
        src, pos, size_list = self.prepare_features(x, self.num_feature_levels, self.pe_layer, self.input_proj,
                                                    self.level_embed)

        _, bs, _ = src[0].shape
        query_feat,query_embed = self.lang_encoder.get()
        # QxNxC
        query_embed = query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        results = []
        # prediction heads on learnable query features
        results.append(self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0]))

        return results

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, layer_id=-1):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        outputs_bbox = [None for i in range(len(outputs_mask))]


        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).float()
        attn_mask = attn_mask.detach()

        results = {
            "predictions_mask": outputs_mask,
            "predictions_bbox": outputs_bbox,
            "predictions_maskemb": mask_embed,
            "attn_mask": attn_mask,
        }
        return results

    def grad(self, inputs):

        x_result = self.sobel_x_conv(inputs)
        y_result = self.sobel_y_conv(inputs)
        out = torch.sqrt(x_result ** 2 + y_result ** 2)
        return out


@register_decoder
def get_seem_interface(cfg, in_channels, lang_encoder, mask_classification, extra):
    return SEEMDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
