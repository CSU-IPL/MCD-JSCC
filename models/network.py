import torch
import torch.nn as nn
import torch.nn.functional as F


from modules import PatchEmbed
from .encoder import WITT_Encoder
from .decoder import WITT_Decoder


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)

        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)


    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class PointWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PointWiseConv, self).__init__()
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.point_conv(input)
        return out



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class Transmitter(nn.Module):
    def __init__(self, encoder_kwargs, args, embed_dims=64):
        super(Transmitter, self).__init__()
        self.args = args

        patch_embed_dim = encoder_kwargs.get("embed_dims", [embed_dims])[0]
        self.patch_embed = PatchEmbed(img_size=args.img_size, patch_size=2, in_chans=3, embed_dim=patch_embed_dim)
        self.Main_Extractor = WITT_Encoder(**encoder_kwargs)

    def forward(self, x, cr):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        return self.Main_Extractor(x, cr)


class Receiver(nn.Module):
    def __init__(self, decoder_kwargs, args):
        super(Receiver, self).__init__()
        self.args = args
        self.Main_Extractor = WITT_Decoder(**decoder_kwargs)

    def forward(self, x, cr, hca_group_tokens=None):
        return self.Main_Extractor(x, cr, hca_group_tokens=hca_group_tokens)
