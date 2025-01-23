import sys
from ..body.decoder.modules import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
import torch.nn as nn
from ..modules import PositionEmbeddingSine
from detectron2.modeling import ShapeSpec
import matplotlib.pyplot as plt


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


from timm.models.layers import SqueezeExcite

import torch


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


from timm.models.vision_transformer import trunc_normal_


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepViT(nn.Module):
    def __init__(self, cfgs, lang, num_classes=1000, distillation=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.lang = lang
        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                                          Conv2d_BN(input_channel // 2, input_channel, 3, 1, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": 256,
            "res3": 256,
            "res4": 256,
            "res5": 256,
        }
        self.transformer_cross_attention_layers_proj = nn.ModuleList()
        self.transformer_cross_attention_layers_broad = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.transformer_self_attention_layers = nn.ModuleList()
        self.proj1 = torch.nn.Sequential(Conv2d_BN(64, 256, 1, 1, 0), torch.nn.GELU())
        self.proj2 = torch.nn.Sequential(Conv2d_BN(128, 256, 1, 1, 0), torch.nn.GELU())
        self.proj3 = torch.nn.Sequential(Conv2d_BN(256, 256, 1, 1, 0), torch.nn.GELU())
        self.proj4 = torch.nn.Sequential(Conv2d_BN(512, 256, 1, 1, 0), torch.nn.GELU())
        self.proj_256 = torch.nn.Sequential(Conv2d_BN(256, 64, 1, 1, 0), torch.nn.GELU())
        self.proj_128 = torch.nn.Sequential(Conv2d_BN(256, 128, 1, 1, 0), torch.nn.GELU())
        self.proj_64 = torch.nn.Sequential(Conv2d_BN(256, 256, 1, 1, 0), torch.nn.GELU())
        self.n = 0

        N_steps = 256 // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.transformer_cross_attention_layers_proj = nn.ModuleList()
        self.transformer_cross_attention_layers_broad = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.transformer_self_attention_layers = nn.ModuleList()
        for i in range(3):
            self.transformer_self_attention_layers.append(SelfAttentionLayer(
                d_model=256,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            ))
        for i in range(3):
            self.transformer_cross_attention_layers_proj.append(CrossAttentionLayer(
                d_model=256,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            ))
        for i in range(3):
            self.transformer_cross_attention_layers_broad.append(CrossAttentionLayer(
                d_model=256,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            ))
        for i in range(3):
            self.transformer_ffn_layers.append(FFNLayer(
                d_model=256,
                dim_feedforward=2048,
                dropout=0.0,
                normalize_before=False,
            ))
        self.level_embed_proj = nn.Embedding(4, 256)
        self.query_feat_encoder = nn.Embedding(3, 256)
        self.query_embed_encoder = nn.Embedding(3, 256)
        
        self.decoder_norm = nn.LayerNorm(256)
        self.mask_embed = MLP(256, 256, 32, 3)
        self.query_embed_encoder = nn.Embedding(3, 256)
        self.mask_features = nn.Conv2d(
                256,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        self.km_embed = torch.nn.Sequential(Conv2d_BN(1, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                                          Conv2d_BN(input_channel // 2, input_channel, 3, 1, 1))
        self.spatial_attn = spatialAttention()
    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def forward(self, x,km):
        km = self.km_embed(km)
        # x = self.features(x)
        outputs = {}
        for i, f in enumerate(self.features):
            num_cur_levels = 0
            x = f(x)
            if i ==0:
                x = x + km
            if i == 3:
                outputs['res2'] = self.proj1(x)
                b, c, h, w = outputs['res2'].shape
                pos = (self.pe_layer(outputs['res2'], None).flatten(2))
                src = outputs['res2'].flatten(2) + self.level_embed_proj.weight[num_cur_levels][None, :, None]
                pos = pos.permute(2, 0, 1)
                src = src.permute(2, 0, 1)
                _, bs, _ = src.shape
                query_feat = self.query_feat_encoder.weight.unsqueeze(1).repeat(1, bs, 1)
                query_embed = self.query_embed_encoder.weight.unsqueeze(1).repeat(1, bs, 1)

                query_feat, avg_attn = self.transformer_cross_attention_layers_proj[num_cur_levels](
                    query_feat, src,
                    pos=pos, query_pos=query_embed
                )

                query_feat = self.transformer_ffn_layers[num_cur_levels](query_feat)

                query_feat = self.transformer_self_attention_layers[num_cur_levels](
                    query_feat,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed)
                output, avg_attn = self.transformer_cross_attention_layers_broad[num_cur_levels](
                    src, query_feat,
                    pos=query_embed, query_pos=pos
                )
                output = output.permute(1, 2, 0).reshape(b, c, h, w)
                x = x + self.proj_256(output)
                # flatten NxCxHxW to HWxNxC
                num_cur_levels = num_cur_levels + 1
            if i == 7:
                outputs['res3'] = self.proj2(x)
                b, c, h, w = outputs['res3'].shape
                pos = (self.pe_layer(outputs['res3'], None).flatten(2))
                src = outputs['res3'].flatten(2) + self.level_embed_proj.weight[num_cur_levels][None, :, None]
                pos = pos.permute(2, 0, 1)
                src = src.permute(2, 0, 1)

                query_feat, avg_attn = self.transformer_cross_attention_layers_proj[num_cur_levels](
                    query_feat, src,
                    pos=pos, query_pos=query_embed
                )

                query_feat = self.transformer_ffn_layers[num_cur_levels](query_feat)

                query_feat = self.transformer_self_attention_layers[num_cur_levels](
                    query_feat,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed)
                output, avg_attn = self.transformer_cross_attention_layers_broad[num_cur_levels](
                    src, query_feat,
                    pos=query_embed, query_pos=pos
                )
                output = output.permute(1, 2, 0).reshape(b, c, h, w)
                x = x + self.proj_128(output)
                num_cur_levels = num_cur_levels + 1
            if i == 21:
                outputs['res4'] = self.proj3(x)
                b, c, h, w = outputs['res4'].shape
                pos = (self.pe_layer(outputs['res4'], None).flatten(2))
                src = outputs['res4'].flatten(2) + self.level_embed_proj.weight[num_cur_levels][None, :, None]
                pos = pos.permute(2, 0, 1)
                src = src.permute(2, 0, 1)
                query_feat, avg_attn = self.transformer_cross_attention_layers_proj[num_cur_levels](
                    query_feat, src,
                    pos=pos, query_pos=query_embed
                )

                query_feat = self.transformer_ffn_layers[num_cur_levels](query_feat)

                query_feat = self.transformer_self_attention_layers[num_cur_levels](
                    query_feat,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed)
                output, avg_attn = self.transformer_cross_attention_layers_broad[num_cur_levels](
                    src, query_feat,
                    pos=query_embed, query_pos=pos
                )
                output = output.permute(1, 2, 0).reshape(b, c, h, w)
                x = x + self.proj_64(output)
                num_cur_levels = num_cur_levels + 1
            if i == 24:
                outputs['res5'] = self.proj4(x)
                
        mask_features = self.mask_features(outputs['res4'])
        decoder_output = self.decoder_norm(query_feat[-1,:,:].unsqueeze(0))
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                
        return outputs,outputs_mask,query_feat



class spatialAttention(nn.Module):
    def __init__(self , kernel_size = 7):
        super(spatialAttention, self).__init__()
        assert kernel_size in (3 , 7 ), " kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        # avg 和 max 两个描述，叠加 共两个通道。
        self.conv1 = nn.Conv2d(2 , 1 , kernel_size , padding = padding , bias = False)#保持卷积前后H、W不变
        self.sigmoid = nn.Sigmoid()
    def forward(self , x,out):
        # egg：input: 1 , 3 * 2 * 2  avg_out :
        avg_out = torch.mean(x , dim = 1 , keepdim = True)#通道维度的平均池化
        # 注意 torch.max(x ,dim = 1) 返回最大值和所在索引，是两个值  keepdim = True 保持维度不变（求max的这个维度变为1），不然这个维度没有了
        max_out ,_ = torch.max(x , dim =1 ,keepdim=True)#通道维度的最大池化
        # print(avg_out.shape)
        # print(max_out.shape)
        x = torch.cat([avg_out , max_out] , dim =1)
        # print(x.shape)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x * out

from .registry import register_backbone


@register_backbone
def repvit_m1_1(cfg, lang, pretrained=False, num_classes=1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, lang, num_classes=num_classes, distillation=distillation)
