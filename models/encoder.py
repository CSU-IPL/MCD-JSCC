

from modules import *

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,

                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        H, W = self.input_resolution

        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"

        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows = self.attn(x_windows,
                                 add_token=False,
                                 mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.to(self.norm1.weight.device)
        else:
            pass


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


class ConvSwinBlock(nn.Module):
    def __init__(self, channel, swin_block):
        super().__init__()
        self.conv = ChannelAttentionBlock(channel)
        self.swin = swin_block

    def forward(self, x):

        x = self.conv(x)
        x = self.swin(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.depth = depth

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, out_dim={self.out_dim}, input_resolution={self.input_resolution}, shape_adjust=True"

    def flops(self):
        flops = 0
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        if self.downsample is not None:
            self.downsample.input_resolution = (H, W)


# SNR->1->M->M
class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.GELU(),
            nn.Linear(M, M),
            nn.GELU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)


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


class WITT_Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 bottleneck_dim=16, deepseek_num_heads=None,
                 image_recovery_depth=0,
                 encoder_hca_enabled=False,
                 encoder_hca_tile_tokens=None,
                 encoder_hca_group_tokens=4,
                 encoder_hca_group_token_counts=None,
                 encoder_hca_eval_group_tokens=None,
                 encoder_hca_selector_hidden_ratio=2.0,
                 encoder_coarse_group_size=None,
                 encoder_csa_group_size=None):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.C = C
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)

        self.hidden_dim = int((self.embed_dims[len(embed_dims)-1]))
        self.layer_num = 7
        self.CR_layer_num = 7

        # build layers
        self.layers = nn.ModuleList()
        current_resolution = self.patches_resolution
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else int(embed_dims[0]),
                               out_dim=int(embed_dims[i_layer]),
                               input_resolution=current_resolution,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer != 0 else None)
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
            if i_layer != 0:
                current_resolution = (current_resolution[0] // 2, current_resolution[1] // 2)

        self.H, self.W = current_resolution
        self.input_resolution = current_resolution

        self.CR_Adapter = CR_Adapter(self.CR_layer_num, self.embed_dims[len(embed_dims) - 1],
                                     self.hidden_dim, self.input_resolution, self.num_layers)

        self.norm = norm_layer(embed_dims[-1])
        self.recovery_insert_layer = 1 if self.num_layers > 1 else self.num_layers
        self.recovery_resolution = self.patches_resolution
        self.recovery_dim = embed_dims[0]
        self.image_recovery_depth = max(0, int(image_recovery_depth))
        self.image_recovery = nn.ModuleList()
        for _ in range(self.image_recovery_depth):
            self.image_recovery.append(
                DeepSeekRecoveryBlock(
                    dim=self.recovery_dim,
                    input_resolution=self.recovery_resolution,
                    num_heads=deepseek_num_heads or num_heads[0],
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    hca_enabled=encoder_hca_enabled,
                    hca_tile_tokens=encoder_hca_tile_tokens,
                    hca_group_tokens=encoder_hca_group_tokens,
                    hca_group_token_counts=encoder_hca_group_token_counts,
                    hca_eval_group_tokens=encoder_hca_eval_group_tokens,
                    hca_selector_hidden_ratio=encoder_hca_selector_hidden_ratio,
                    csa_group_size=encoder_csa_group_size,
                    coarse_group_size=encoder_coarse_group_size,
                )
            )
        self.head_list = nn.Linear(embed_dims[-1], C)
        self.apply(self._init_weights)

    def forward_image_recovery(self, x, hca_group_tokens=None):
        for recovery_blk in self.image_recovery:
            x = recovery_blk(x, hca_group_tokens=hca_group_tokens)
        return x

    def forward(self, x, cr, hca_group_tokens=None):
        for i_layer, layer in enumerate(self.layers):
            if i_layer == self.recovery_insert_layer:
                x = self.forward_image_recovery(x, hca_group_tokens=hca_group_tokens)
            x = layer(x)
        if self.recovery_insert_layer == len(self.layers):
            x = self.forward_image_recovery(x, hca_group_tokens=hca_group_tokens)
        x = self.norm(x)
        x = self.CR_Adapter(x, cr)
        return self.head_list(x)

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
        for i, layer in enumerate(self.layers):
            if i == self.recovery_insert_layer:
                for recovery_blk in self.image_recovery:
                    flops += recovery_blk.flops(hca_group_tokens=hca_group_tokens)
            flops += layer.flops()
        if self.recovery_insert_layer == len(self.layers):
            for recovery_blk in self.image_recovery:
                flops += recovery_blk.flops(hca_group_tokens=hca_group_tokens)
        flops += self.H * self.W * self.embed_dims[-1]
        flops += self.H * self.W * self.embed_dims[-1] * self.C
        return flops

    def update_resolution(self, H, W):
        current_H = H // self.patch_size
        current_W = W // self.patch_size
        self.patches_resolution = (current_H, current_W)
        self.recovery_resolution = (current_H, current_W)
        for recovery_blk in self.image_recovery:
            recovery_blk.update_resolution(current_H, current_W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(current_H, current_W)
            if layer.downsample is not None:
                current_H = current_H // 2
                current_W = current_W // 2

        self.H = current_H
        self.W = current_W
        self.input_resolution = (current_H, current_W)
        self.CR_Adapter.update_resolution(current_H, current_W)


def create_encoder(**kwargs):
    model = WITT_Encoder(**kwargs)
    return model
