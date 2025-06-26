import torch
from torch import nn
from torch.nn import functional as F

from model.layers import InvConv2dLU,InvConv2d,ZeroConv2d
from model.layers.modulation import SelfConditionedModulation
from model.layers.activation_norm import ActNorm
from model.layers.CBAM import CBAM,ChannelAttention,SpatialAttention
# from model.layers.attention import SpatialAttention

class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super(AffineCoupling,self).__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            out_b = (in_b + t) * s

        else:
            net_out = self.net(in_a)
            out_b = in_b - net_out

        return torch.cat([in_a, out_b], 1)

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b + net_out

        return torch.cat([out_a, in_b], 1)

class Flow(nn.Module):
    def __init__(self, in_channel, use_coupling=True, affine=True, conv_lu=True):
        super(Flow,self).__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
            
        self.use_coupling = use_coupling
        if self.use_coupling:
            self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        
        input = self.actnorm(input)
        input = self.invconv(input)
        #print('input: ',input.shape)
        if self.use_coupling:
            input = self.coupling(input)
        return input

    def reverse(self, input):
        if self.use_coupling:
            input = self.coupling.reverse(input)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, squeeze=4, affine=True, conv_lu=True):
        super(Block,self).__init__()

        # print('the in channel:',in_channel)
        squeeze_dim = in_channel * squeeze
        self.flows = nn.ModuleList()
        self.modulate = SelfConditionedModulation(squeeze_dim, return_delta=True)

        self.spatial_attn = SpatialAttention()
        self.channel_attn = ChannelAttention(squeeze_dim, reduction=squeeze)

        self.last_attention = None
        self.last_delta_mu = None
        self.last_delta_std = None

        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))
        
    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        
        for flow in self.flows:
            out = flow(out)

        return out

    def reverse(self, output, alpha=True):
        input = output

        if alpha:
            # Channel & spatial attention
            ch_att = self.channel_attn(input)
            input_attn = input * ch_att

            sp_att = self.spatial_attn(input_attn)
            input_attn = input_attn * sp_att

            self.last_attention = sp_att.detach().cpu()

            # Save attention mask as soft gate
            soft_gate = sp_att  # shape: [B, 1, H, W]
        else:
            input_attn = input
            soft_gate = torch.zeros_like(input[:, :1, :, :])  # dummy gate (no modulation)

        # Apply modulation
        mod_input, d_mean, d_var = self.modulate(input_attn,input)

        self.last_delta_mu = d_mean
        self.last_delta_std = d_var

        # Soft-gating: combine original and modulated input
        input = input * (1 - soft_gate) + mod_input * soft_gate

        # Decode through flow
        for flow in reversed(self.flows):
            input = flow.reverse(input)

        # 5. Un-squeeze back to spatial dimensions
        b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3).contiguous()
        unsqueezed = unsqueezed.view(b_size, n_channel // 4, height * 2, width * 2)

        return unsqueezed

class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True, alpha=True):
        super(Glow,self).__init__()
        
        self.alpha = alpha
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4
        
        self.blocks.append(Block(n_channel, n_flow, affine=affine))
        
        
    def forward(self, input, forward=True):
        if forward:
            return self._forward_set(input)
        else:
            return self._reverse_set(input)

    def _forward_set(self, input):
        z = input
        
        for block in self.blocks:
            z = block(z)
        return z

    def _reverse_set(self, z):
        out = z
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out,alpha=self.alpha)
        return out
