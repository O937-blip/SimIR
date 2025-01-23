# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import sys

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, DeformConv, ShapeSpec, get_norm

from .registry import register_encoder
from ..transformer_blocks import TransformerEncoder, TransformerEncoderLayer, _get_clones, _get_activation_fn
from ...modules import PositionEmbeddingSine
from ...utils import configurable
from ..decoder.modules import SelfAttentionLayer, CrossAttentionLayer,FFNLayer,MLP


# This is a modified FPN decoder.
class BasePixelDecoder(nn.Module):
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            conv_dim: int,
            mask_dim: int,
            mask_on: bool,
            norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_on = mask_on
        if self.mask_on:
            self.mask_dim = mask_dim
            self.mask_features = Conv2d(
                conv_dim,
                mask_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        enc_cfg = cfg['MODEL']['ENCODER']
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in enc_cfg['IN_FEATURES']
        }
        ret["conv_dim"] = enc_cfg['CONVS_DIM']
        ret["mask_dim"] = enc_cfg['MASK_DIM']
        ret["norm"] = enc_cfg['NORM']
        return ret

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1

        mask_features = self.mask_features(y) if self.mask_on else None
        return mask_features, None, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)


class TransformerEncoderOnly(nn.Module):
    def __init__(
            self,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


# This is a modified FPN decoder with extra Transformer encoder that processes the lowest-resolution feature map.
class TransformerEncoderPixelDecoder(BasePixelDecoder):
    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            transformer_dropout: float,
            transformer_nheads: int,
            transformer_dim_feedforward: int,
            transformer_enc_layers: int,
            transformer_pre_norm: bool,
            conv_dim: int,
            mask_dim: int,
            mask_on: int,
            norm: Optional[Union[str, Callable]] = None,
            lang:nn.Module

    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm, mask_on=mask_on)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)
        '''self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )'''
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv
        self.n = 0

        N_steps = 256 // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.level_embed_proj = nn.Embedding(4, 256)
        #self.level_embed_broad = nn.Embedding(3, 256)
        self.transformer_cross_attention_layers_proj = nn.ModuleList()
        self.transformer_cross_attention_layers_broad = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.transformer_self_attention_layers = nn.ModuleList()
        for i in range(4):
            self.transformer_self_attention_layers.append(SelfAttentionLayer(
                    d_model=256,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                ))
        for i in range(4):
            self.transformer_cross_attention_layers_proj.append(CrossAttentionLayer(
            d_model=256,
            nhead=8,
            dropout=0.0,
            normalize_before=False,
        ))
        for i in range(4):
            self.transformer_cross_attention_layers_broad.append(CrossAttentionLayer(
            d_model=256,
            nhead=8,
            dropout=0.0,
            normalize_before=False,
        ))
        for i in range(4):
            self.transformer_ffn_layers.append(FFNLayer(
                    d_model=256,
                    dim_feedforward=2048,
                    dropout=0.0,
                    normalize_before=False,
                ))
        self.lang = lang
        self.query_feat = nn.Embedding(4, 256)
        # learnable query p.e.
        self.query_embed = nn.Embedding(4, 256)
        self.decoder_norm = nn.LayerNorm(256)
        self.mask_embed = MLP(256, 256, mask_dim, 3)
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec],lang):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        ret = super().from_config(cfg, input_shape)
        ret["transformer_dropout"] = dec_cfg['DROPOUT']
        ret["transformer_nheads"] = dec_cfg['NHEADS']
        ret["transformer_dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
        ret["transformer_enc_layers"] = enc_cfg['TRANSFORMER_ENC_LAYERS']  # a separate config
        ret["transformer_pre_norm"] = dec_cfg['PRE_NORM']

        ret['mask_on'] = cfg['MODEL']['DECODER']['MASK']
        ret['lang'] = lang
        return ret

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        query_feat, query_embed = self.lang.get()
        # QxNxC

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                '''b, c, h, w = transformer.shape
                import matplotlib.pyplot as plt

                for j in range(b):
                        tensor = torch.mean(transformer[j, :, :, :],dim=0)
                        tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                        tensor = tensor.cpu().detach().numpy()
                        plt.imshow(tensor, cmap='hot', interpolation='nearest')
                        plt.colorbar()
                        # 保存图像为PNG文件

                        plt.savefig(f'/home/zmj/zc/irr/visual/hv2/before/before_{self.n}_{num_cur_levels}.png')
                        plt.close()'''
                b, c, h, w = transformer.shape
                pos = (self.pe_layer(transformer, None).flatten(2))
                src = transformer.flatten(2) + self.level_embed_proj.weight[0][None, :, None]
                pos = pos.permute(2, 0, 1)
                src = src.permute(2, 0, 1)
                _, bs, _ = src.shape
                query_embed_t = query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
                query_feat_t = query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

                query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
                query_embed = torch.cat([query_embed,query_embed_t],dim=0)
                query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
                query_feat = torch.cat([query_feat, query_feat_t], dim=0)
                query_feat, avg_attn = self.transformer_cross_attention_layers_proj[num_cur_levels](
                    query_feat, src,
                    pos=pos, query_pos=query_embed
                )
                '''import matplotlib.pyplot as plt
                b, q, n = avg_attn.shape
                map = avg_attn.reshape(b, q, int(n ** 0.5), int(n ** 0.5))

                for j in range(b):
                    for k in range(q):
                        tensor = map[j, k, :, :]
                        tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                        tensor = tensor.cpu().detach().numpy()
                        plt.imshow(tensor, cmap='hot', interpolation='nearest')
                        plt.colorbar()
                        # 保存图像为PNG文件

                        plt.savefig(f'/home/zmj/zc/irr/visual/hv2/attn_proj/heatmap_{self.n}_{num_cur_levels}_{k}.png')
                        plt.close()'''

                query_feat = self.transformer_ffn_layers[num_cur_levels](query_feat)

                query_feat = self.transformer_self_attention_layers[num_cur_levels](
                    query_feat,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed)
                # flatten NxCxHxW to HWxNxC
                output, avg_attn = self.transformer_cross_attention_layers_broad[num_cur_levels](
                    src, query_feat,
                    pos=query_embed, query_pos=pos
                )
                transformer = output.permute(1, 2, 0).reshape(b, c, h, w)
                '''b, c, h, w = transformer.shape
                import matplotlib.pyplot as plt

                for j in range(b):
                        tensor = torch.mean(transformer[j, :, :, :],dim=0)
                        tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                        tensor = tensor.cpu().detach().numpy()
                        plt.imshow(tensor, cmap='hot', interpolation='nearest')
                        plt.colorbar()
                        # 保存图像为PNG文件

                        plt.savefig(f'/home/zmj/zc/irr/visual/hv2/after/after_{self.n}_{num_cur_levels}.png')
                        plt.close()'''
                y = output_conv(transformer)
                '''b, c, h, w = y.shape
                import matplotlib.pyplot as plt

                for j in range(b):
                    tensor = torch.mean(y[j, :, :, :], dim=0)
                    tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                    tensor = tensor.cpu().detach().numpy()
                    plt.imshow(tensor, cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    # 保存图像为PNG文件

                    plt.savefig(f'/home/zmj/zc/irr/visual/momov14/y/y_{self.n}_{num_cur_levels}.png')
                    plt.close()'''
                '''b, c, h, w = y.shape
                import matplotlib.pyplot as plt
                for j in range(b):
                    tensor = torch.mean(y[j, :, :, :], dim=0)
                    tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                    tensor = tensor.cpu().detach().numpy()
                    plt.imshow(tensor, cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    # 保存图像为PNG文件

                    plt.savefig(f'/home/zmj/zc/irr/visual/momov13/y/y_{self.n}_{num_cur_levels}.png')
                    plt.close()'''
                # save intermediate feature as input to Transformer decoder
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                '''b, c, h, w = cur_fpn.shape
                import matplotlib.pyplot as plt
                for j in range(b):
                        tensor = torch.mean(cur_fpn[j, :, :, :],dim=0)
                        tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                        tensor = tensor.cpu().detach().numpy()
                        plt.imshow(tensor, cmap='hot', interpolation='nearest')
                        plt.colorbar()
                        # 保存图像为PNG文件

                        plt.savefig(f'/home/zmj/zc/irr/visual/hv2/before/before_{self.n}_{num_cur_levels}.png')
                        plt.close()'''
                b, c, h, w = cur_fpn.shape
                pos = (self.pe_layer(cur_fpn, None).flatten(2))
                src = cur_fpn.flatten(2) + self.level_embed_proj.weight[num_cur_levels][None, :, None]
                pos = pos.permute(2, 0, 1)
                src = src.permute(2, 0, 1)
                _, bs, _ = src.shape
                query_feat, avg_attn = self.transformer_cross_attention_layers_proj[num_cur_levels](
                    query_feat, src,
                    pos=pos, query_pos=query_embed
                )
                '''import matplotlib.pyplot as plt
                b, q, n = avg_attn.shape
                map = avg_attn.reshape(b, q, int(n ** 0.5), int(n ** 0.5))

                for j in range(b):
                    for k in range(q):
                        tensor = map[j, k, :, :]
                        tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                        tensor = tensor.cpu().detach().numpy()
                        plt.imshow(tensor, cmap='hot', interpolation='nearest')
                        plt.colorbar()
                        # 保存图像为PNG文件

                        plt.savefig(f'/home/zmj/zc/irr/visual/hv2/attn_proj/heatmap_{self.n}_{num_cur_levels}_{k}.png')
                        plt.close()'''
                query_feat = self.transformer_ffn_layers[num_cur_levels](query_feat)

                query_feat = self.transformer_self_attention_layers[num_cur_levels](
                    query_feat,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed)
                # flatten NxCxHxW to HWxNxC
                output, avg_attn = self.transformer_cross_attention_layers_broad[num_cur_levels](
                    src, query_feat,
                    pos=query_embed, query_pos=pos
                )


                cur_fpn = output.permute(1, 2, 0).reshape(b, c, h, w)
                # Following FPN implementation, we use nearest upsampling here
                '''b, c, h, w = cur_fpn.shape
                import matplotlib.pyplot as plt
                for j in range(b):
                        tensor = torch.mean(cur_fpn[j, :, :, :],dim=0)
                        tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                        tensor = tensor.cpu().detach().numpy()
                        plt.imshow(tensor, cmap='hot', interpolation='nearest')
                        plt.colorbar()
                        # 保存图像为PNG文件

                        plt.savefig(f'/home/zmj/zc/irr/visual/hv2/after/after_{self.n}_{num_cur_levels}.png')
                        plt.close()'''
                '''b, c, h, w = int.shape
                import matplotlib.pyplot as plt
                for j in range(b):
                    tensor = torch.mean(int[j, :, :, :], dim=0)
                    tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                    tensor = tensor.cpu().detach().numpy()
                    plt.imshow(tensor, cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    # 保存图像为PNG文件

                    plt.savefig(f'/home/zmj/zc/irr/visual/momov10/int/int_{self.n}_{num_cur_levels}.png')
                    plt.close()'''

                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                '''b, c, h, w = y.shape
                import matplotlib.pyplot as plt
                for j in range(b):
                    tensor = torch.mean(y[j, :, :, :], dim=0)
                    tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                    tensor = tensor.cpu().detach().numpy()
                    plt.imshow(tensor, cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    # 保存图像为PNG文件

                    plt.savefig(f'/home/zmj/zc/1/visual/default/y/y_{self.n}_{idx}.png')
                    plt.close()'''
                y = output_conv(y)
                '''b, c, h, w = y.shape
                import matplotlib.pyplot as plt
                for j in range(b):
                    tensor = torch.mean(y[j, :, :, :], dim=0)
                    tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
                    tensor = tensor.cpu().detach().numpy()
                    plt.imshow(tensor, cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    # 保存图像为PNG文件

                    plt.savefig(f'/home/zmj/zc/irr/visual/momov14/y/y_{self.n}_{num_cur_levels}.png')
                    plt.close()'''
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        mask_features = self.mask_features(y) if self.mask_on else None

        decoder_output = self.decoder_norm(query_feat[0,:,:].unsqueeze(0))
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        #self.n = self.n+1
        return mask_features, outputs_mask, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)


@register_encoder
def get_transformer_encoder_fpn(cfg, input_shape,lang):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    model = TransformerEncoderPixelDecoder(cfg, input_shape,lang)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model