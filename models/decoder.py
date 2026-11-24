
from modules import *
import torch
from .encoder import AdaptiveModulator

class ConvBlock(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_dim, 2*in_dim),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Linear(2 *in_dim, in_dim),
            nn.Dropout(0.),
        )

    def forward(self, x):
        x = x + self.conv(x)

        return x

class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2 * in_dim),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Linear(2 * in_dim, in_dim),
            nn.Dropout(0.),
        )

        self.channel_attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, in_dim),
            nn.Sigmoid()
        )

        self.feature_map = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, in_dim),
            nn.GELU(),
        )

    def forward(self, x):

        residual = x

        transformed = self.mlp(x)

        squeezed = x.mean(dim=1)
        squeezed = self.feature_map(squeezed) + squeezed

        attention_input = squeezed

        attention_weights = self.channel_attention(attention_input).unsqueeze(1)

        output = residual + transformed * attention_weights

        return output


class BasicLayer(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # The decoder's adaptive compute is handled by the DeepSeek-style block before
        # these layers. Keep this stage as shape/channel recovery only.
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, shape_adjust=True"

    def flops(self):
        flops = 0
        if self.upsample is not None:
            flops += self.upsample.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)

class Adapter(nn.Module):
    def __init__(self, layer_num, in_dims, hidden_dim, input_resolution, num_layers):
        super(Adapter, self).__init__()
        self.layer_num = layer_num
        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution
        self.num_layers = num_layers

        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.pm_list = nn.ModuleList()

        self.sm_list.append(nn.Linear(in_dims, self.hidden_dim))
        for i in range(self.layer_num):
            if i == self.layer_num - 1:
                outdim = self.in_dims
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.pm_list.append(AdaptiveModulator(self.hidden_dim))

            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, para, snr):
        B, L = x.shape[0], x.shape[1]
        (W, H) = self.input_resolution
        device = x.device

        para_cuda = torch.tensor(para, dtype=torch.float).to(device)
        para_batch = para_cuda.unsqueeze(0).expand(B, -1)

        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x.detach())
            else:
                temp = self.sm_list[i](temp)

            pm = self.pm_list[i](para_batch).unsqueeze(1).expand(-1, L, -1)
            bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)

            temp = temp * bm * pm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x = x * mod_val + x

        return x

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)


class CR_Adapter(nn.Module):
    def __init__(self, layer_num, in_dims, hidden_dim, input_resolution, num_layers):
        super(CR_Adapter, self).__init__()
        self.layer_num = layer_num
        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution
        self.num_layers = num_layers

        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()

        self.sm_list.append(nn.Linear(in_dims, self.hidden_dim))
        for i in range(self.layer_num):
            if i == self.layer_num - 1:
                outdim = self.in_dims
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))

            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cr):
        B, L = x.shape[0], x.shape[1]
        (W, H) = self.input_resolution
        device = x.device

        cr_cuda = torch.tensor(cr, dtype=torch.float).to(device)
        cr_batch = cr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x.detach())
            else:
                temp = self.sm_list[i](temp)

            bm = self.bm_list[i](cr_batch).unsqueeze(1).expand(-1, L, -1)

            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x = x * mod_val + x

        return x

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)

class WITT_Decoder(nn.Module):
    def __init__(self, img_size, embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 bottleneck_dim=16, deepseek_num_heads=None,
                 image_recovery_depth=3,
                 hca_enabled=False, hca_tile_tokens=None, hca_group_tokens=4,
                 hca_group_token_counts=None,
                 hca_eval_group_tokens=None,
                 hca_selector_hidden_ratio=2.0,
                 decoder_coarse_group_size=None):
        super().__init__()

        self.num_layers = len(depths)
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.H = img_size[0]
        self.W = img_size[1]

        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))
        self.input_resolution = self.patches_resolution
        num_patches = self.H // 4 * self.W // 4
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging)
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())

        self.head_list = nn.Linear(C, embed_dims[0])
        self.image_recovery_depth = max(0, int(image_recovery_depth))
        self.image_recovery = nn.ModuleList()
        for _ in range(self.image_recovery_depth):
            self.image_recovery.append(
                DeepSeekRecoveryBlock(
                    dim=embed_dims[0],
                    input_resolution=self.input_resolution,
                    num_heads=deepseek_num_heads or num_heads[0],
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    hca_enabled=hca_enabled,
                    hca_tile_tokens=hca_tile_tokens,
                    hca_group_tokens=hca_group_tokens,
                    hca_group_token_counts=hca_group_token_counts,
                    hca_eval_group_tokens=hca_eval_group_tokens,
                    hca_selector_hidden_ratio=hca_selector_hidden_ratio,
                    coarse_group_size=decoder_coarse_group_size,
                )
            )
        self.apply(self._init_weights)
        self.hidden_dim = int(self.embed_dims[0] * 1.5)

        self.CR_layer_num = 7
        self.CR_Adapter = CR_Adapter(self.CR_layer_num, self.embed_dims[0],
                                      self.hidden_dim, self.input_resolution, self.num_layers)

    def forward(self, x, cr, hca_group_tokens=None):
        x = self.head_list(x)
        x = self.CR_Adapter(x, cr)
        for recovery_blk in self.image_recovery:
            x = recovery_blk(x, hca_group_tokens=hca_group_tokens)
        for layer in self.layers:
            x = layer(x)
        B, L, N = x.shape
        return x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self, hca_group_tokens=None):
        flops = 0
        flops += self.recovery_flops(hca_group_tokens=hca_group_tokens)
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops

    def recovery_flops(self, hca_group_tokens=None):
        flops = 0
        for recovery_blk in self.image_recovery:
            flops += recovery_blk.flops(hca_group_tokens=hca_group_tokens)
        return flops

    def hca_group_token_choices(self):
        for recovery_blk in self.image_recovery:
            hca_attention = getattr(recovery_blk, "hca_attention", None)
            if hca_attention is not None:
                return hca_attention.group_token_choices()
        return None

    def hca_eval_group_tokens(self):
        for recovery_blk in self.image_recovery:
            hca_attention = getattr(recovery_blk, "hca_attention", None)
            if hca_attention is not None:
                return hca_attention.default_eval_group_tokens()
        return None

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        self.CR_Adapter.update_resolution(H, W)
        for recovery_blk in self.image_recovery:
            recovery_blk.update_resolution(H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))


def create_decoder(**kwargs):
    model = WITT_Decoder(**kwargs)
    return model
