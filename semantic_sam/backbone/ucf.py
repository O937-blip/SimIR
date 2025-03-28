
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            # self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        # print(x_l.shape)
        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg

class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, padding_mode='zeros',dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=False, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 inline=False, outline=False,**conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.inline = inline
        self.outline= outline
    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.outline:
            out = torch.cat(out, dim=1)
        return out

class CDC_conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size=3, stride=1,
                padding=1, dilation=1, theta=0.7, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                            stride = stride, dilation=dilation, bias=bias, padding_mode=padding_mode)
        self.theta = theta

    def forward(self, x):
        norm_out = self.conv(x)
        if (self.theta - 0.0) < 1e-6:
            return norm_out
        else:
            # [c_out, c_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            diff_out = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                dilation=1, padding=0)
            out = norm_out - self.theta * diff_out
            return out

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, theta_1=0, theta_2=0.7, theta_r=0., norm=nn.BatchNorm2d,
                 padding_mode='zeros'):
        super().__init__()
        self.conv_block = nn.Sequential(
            CDC_conv(in_c, out_c, kernel_size=3, padding=1, theta=theta_1, padding_mode=padding_mode,
            stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
            nn.ReLU(inplace=True),
            CDC_conv(out_c, out_c, kernel_size=3, padding=1, theta=theta_2, padding_mode=padding_mode,
            stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.residual_block = nn.Sequential(
            CDC_conv(in_c, out_c, kernel_size=3, padding=1, theta=theta_r, padding_mode=padding_mode,
            stride=stride, bias=False if norm == nn.BatchNorm2d else True),
            norm(out_c),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_out = self.conv_block(x)
        residual_out = self.residual_block(x)
        out = self.relu(conv_out + residual_out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.squeeze = nn.Conv2d(in_c, out_c, kernel_size=1)

        self.up_block = nn.Sequential(
            nn.Conv2d(out_c*2, out_c, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )
    def forward(self, x, lateral):
        size = lateral.shape[-2:]
        x = self.squeeze(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        out = self.up_block(torch.cat((x, lateral), dim=1))
        return out


class UCFNet(nn.Module):
    def __init__(self, in_c=3, out_c=1,
                base_dim=32,
                theta_0=0.7, theta_1=0, theta_2=0.7, theta_r=0,
                maxpool='pool', norm='bn', padding_mode='reflect',
                n_blocks=7,
                 ):
        super(UCFNet, self).__init__()
        if norm == 'bn':
            self.norm = nn.BatchNorm2d
        if maxpool == 'pool':
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stride = 1
        self.conv1 = nn.Sequential(
            CDC_conv(in_c, base_dim, bias=False, theta=theta_0),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            CDC_conv(base_dim, base_dim, bias=False, theta=theta_0),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(base_dim, base_dim * 2, norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=self.stride)
        self.layer2 = ResidualBlock(base_dim * 2, base_dim * 4, norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=self.stride)
        self.layer3 = ResidualBlock(base_dim * 4, base_dim * 8, norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=self.stride)
        self.layer4 = ResidualBlock(base_dim * 8, base_dim * 16, norm=self.norm, padding_mode='zeros',
                                    theta_1=theta_1, theta_2=theta_2, theta_r=theta_r, stride=self.stride)
        self.up3 = UpsampleBlock(base_dim * 16, base_dim * 8)
        self.up2 = UpsampleBlock(base_dim * 8, base_dim * 4)
        self.up1 = UpsampleBlock(base_dim * 4, base_dim * 2)
        self.up0 = UpsampleBlock(base_dim * 2, base_dim)
        self.last_conv = nn.Conv2d(base_dim, out_c, kernel_size=1, stride=1)
        ffc_blocks = nn.ModuleList()
        resnet_conv_kwargs = {'ratio_gin': 0.75, 'ratio_gout': 0.75}
        for i in range(n_blocks):
            if i == 0:
                cur_resblock = FFCResnetBlock(base_dim * 16, padding_type='reflect', activation_layer=nn.ReLU,
                                          norm_layer=nn.BatchNorm2d, inline=True, outline=False,**resnet_conv_kwargs)
            elif i == n_blocks-1:
                cur_resblock = FFCResnetBlock(base_dim * 16, padding_type='reflect', activation_layer=nn.ReLU,
                                          norm_layer=nn.BatchNorm2d, inline=False, outline=True, **resnet_conv_kwargs)
            else:
                cur_resblock = FFCResnetBlock(base_dim * 16, padding_type='reflect', activation_layer=nn.ReLU,
                                          norm_layer=nn.BatchNorm2d, inline=False, outline=False, **resnet_conv_kwargs)
            ffc_blocks.append(cur_resblock)
        self.ffc_blocks = nn.Sequential(*ffc_blocks)
    def forward(self, x):
        out_0 = self.conv1(x)
        out_0 = self.conv2(out_0)
        out_1 = self.layer1(self.maxpool(out_0))
        out_2 = self.layer2(self.maxpool(out_1))
        out_3 = self.layer3(self.maxpool(out_2))
        out_4 = self.layer4(self.maxpool(out_3))
        out_da = self.ffc_blocks(out_4)
        up_3 = self.up3(out_da, out_3)
        up_2 = self.up2(up_3, out_2)
        up_1 = self.up1(up_2, out_1)
        up_0 = self.up0(up_1, out_0)
        out = self.last_conv(up_0)
        return out



from .registry import register_backbone


@register_backbone
def ucf(cfg,lang, **kwargs):
    return  UCFNet(theta_r=0, theta_0=0.7, theta_1=0, theta_2=0.7, n_blocks=7)
