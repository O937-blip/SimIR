# ------------------------------------------------------------------------
# Copyright (c) MicroSoft, Inc. and its affiliates.
# Modified from DINO https://github.com/IDEA-Research/MaskDINO by Feng Li.
# ------------------------------------------------------------------------
import logging
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .registry import register_body
from .encoder import build_encoder,build_s_encoder
from .decoder import build_decoder,build_s_decoder
from ..utils import configurable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
class ISTDHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
            pixel_decoder2: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_predictor: nn.Module,
            transformer_predictor2: nn.Module,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.ignore_value = ignore_value
        self.loss_weight = loss_weight
        self.n = 0
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

        self.pixel_decoder2 = pixel_decoder2
        self.predictor2 = transformer_predictor2

        self.num_classes = num_classes
        # store processed features
        self.processed_features = None



    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec],lang, extra: dict):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']
        transformer_predictor_in_channels = enc_cfg['CONVS_DIM']

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in enc_cfg['IN_FEATURES']
            },
            "num_classes": enc_cfg.get('NUM_CLASSES', None),
            "pixel_decoder": build_encoder(cfg, input_shape,lang),
            "pixel_decoder2": build_s_encoder(cfg, input_shape),
            "loss_weight": enc_cfg['LOSS_WEIGHT'],
            "transformer_predictor": build_decoder(
                cfg,
                transformer_predictor_in_channels,
                lang,
                mask_classification=True,
                extra=extra,
            ),
            "transformer_predictor2": build_s_decoder(
                cfg,
                transformer_predictor_in_channels,
                lang,
                mask_classification=True,
                extra=extra,
            ),
        }

    def forward(self, features,outputs_mask,sparse_query_encoder, mask=None, targets=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        return self.layers(features, outputs_mask,sparse_query_encoder,mask, targets=targets, target_queries=target_queries, target_vlp=target_vlp, task=task, extra=extra)

    def layers(self, features,outputs_mask,sparse_query_encoder, mask=None,targets=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        mask_features, mask, multi_scale_features,sparse_query_fpn = self.pixel_decoder.forward_features(
            features,sparse_query_encoder)
        output = {'predictions_mask': mask}
        sparse_embeddings, dense_embeddings = self.pixel_decoder2(
            points=None,
            boxes=None,
            masks=mask,
        )
        low_res_masks = self.predictor(multi_scale_features, mask_features, dense_embeddings, sparse_query_fpn,
                                       target_queries=target_queries, target_vlp=target_vlp, task=task, extra=extra)
        output2 = {'predictions_mask': low_res_masks}
        output3 = {'predictions_mask': outputs_mask}
        predictions = []
        predictions.append(output3)
        predictions.append(output)
        predictions.append(output2)
        return predictions


@register_body
def get_interactive_maskdino_head(cfg, input_shape,lang, extra):
    return ISTDHead(cfg, input_shape,lang, extra)