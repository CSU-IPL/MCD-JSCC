import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
from bisect import bisect
import torch.nn.functional as F
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight).to(dtype)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, add_token=True, token_num=0, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (N+1)x(N+1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if add_token:
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(
                0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            if add_token:
                # padding mask matrix
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).to(x.device)

            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)
        # self.proj = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H, W = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        # print(self.input_resolution)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H*W//4, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        H, W = input_resolution
        self.patch_merging1 = PatchMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_merging2 = PatchMerging((H // 2, W // 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_merging1(x, H, W)
        x = self.patch_merging2(x, H//2, W//2)
        return x


class PatchReverseMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)
        # self.proj = nn.ConvTranspose2d(dim // 4, 3, 3, stride=1, padding=1)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.norm(x)
        x = self.increment(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = nn.PixelShuffle(2)(x)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        # print(x.shape)
        x = x.flatten(2).permute(0, 2, 1)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class PatchReverseMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        self.use_conv = use_conv
        self.input_resolution = input_resolution
        self.dim = dim
        H, W = input_resolution
        self.patch_reverse_merging1 = PatchReverseMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_reverse_merging2 = PatchReverseMerging((H * 2, W * 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_reverse_merging1(x, H, W)
        x = self.patch_reverse_merging2(x, H*2, W*2)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.proj(x)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class DeepSeekSwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekGQAAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_kv_heads=None):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = int(num_kv_heads or max(1, num_heads // 2))
        if num_heads % self.num_kv_heads != 0:
            self.num_kv_heads = num_heads
        self.head_dim = dim // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _apply_1d_rope(self, q, k):
        rope_dim = (self.head_dim // 2) * 2
        if rope_dim == 0:
            return q, k
        seq_len = q.shape[-2]
        pos = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2, device=q.device, dtype=torch.float32) / rope_dim))
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1).to(dtype=q.dtype)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        q_rope, q_pass = q[..., :rope_dim], q[..., rope_dim:]
        k_rope, k_pass = k[..., :rope_dim], k[..., rope_dim:]
        q = torch.cat((q_rope * cos + self._rotate_half(q_rope) * sin, q_pass), dim=-1)
        k = torch.cat((k_rope * cos + self._rotate_half(k_rope) * sin, k_pass), dim=-1)
        return q, k

    def _apply_1d_rope_cross(self, q, k):
        rope_dim = (self.head_dim // 2) * 2
        if rope_dim == 0:
            return q, k
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2, device=q.device, dtype=torch.float32) / rope_dim))

        q_pos = torch.arange(q_len, device=q.device, dtype=torch.float32)
        q_freqs = torch.einsum("i,j->ij", q_pos, inv_freq)
        q_emb = torch.repeat_interleave(q_freqs, 2, dim=-1).to(dtype=q.dtype)
        q_cos = q_emb.cos()[None, None, :, :]
        q_sin = q_emb.sin()[None, None, :, :]

        k_pos = torch.arange(k_len, device=k.device, dtype=torch.float32)
        k_freqs = torch.einsum("i,j->ij", k_pos, inv_freq)
        k_emb = torch.repeat_interleave(k_freqs, 2, dim=-1).to(dtype=k.dtype)
        k_cos = k_emb.cos()[None, None, :, :]
        k_sin = k_emb.sin()[None, None, :, :]

        q_rope, q_pass = q[..., :rope_dim], q[..., rope_dim:]
        k_rope, k_pass = k[..., :rope_dim], k[..., rope_dim:]
        q = torch.cat((q_rope * q_cos + self._rotate_half(q_rope) * q_sin, q_pass), dim=-1)
        k = torch.cat((k_rope * k_cos + self._rotate_half(k_rope) * k_sin, k_pass), dim=-1)
        return q, k

    def _apply_2d_rope(self, q, k, input_resolution):
        H, W = input_resolution
        rope_dim = (self.head_dim // 4) * 4
        if rope_dim == 0 or H * W != q.shape[-2]:
            return self._apply_1d_rope(q, k)
        axis_dim = rope_dim // 2
        yy, xx = torch.meshgrid(
            torch.arange(H, device=q.device, dtype=torch.float32),
            torch.arange(W, device=q.device, dtype=torch.float32),
            indexing="ij",
        )
        yy = yy.flatten()
        xx = xx.flatten()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, axis_dim, 2, device=q.device, dtype=torch.float32) / axis_dim))
        y_freqs = torch.einsum("i,j->ij", yy, inv_freq)
        x_freqs = torch.einsum("i,j->ij", xx, inv_freq)
        emb = torch.cat(
            (torch.repeat_interleave(y_freqs, 2, dim=-1),
             torch.repeat_interleave(x_freqs, 2, dim=-1)),
            dim=-1,
        ).to(dtype=q.dtype)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        q_rope, q_pass = q[..., :rope_dim], q[..., rope_dim:]
        k_rope, k_pass = k[..., :rope_dim], k[..., rope_dim:]
        q = torch.cat((q_rope * cos + self._rotate_half(q_rope) * sin, q_pass), dim=-1)
        k = torch.cat((k_rope * cos + self._rotate_half(k_rope) * sin, k_pass), dim=-1)
        return q, k

    def forward(self, x, input_resolution=None, kv_tokens=None, attn_mask=None):
        B, L, C = x.shape
        kv_source = x if kv_tokens is None else kv_tokens
        kv_len = kv_source.shape[1]
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_source).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_source).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if kv_tokens is not None:
            # Applying 1D RoPE cross attention in this project
            q, k = self._apply_1d_rope_cross(q, k)
        elif input_resolution is None:
            q, k = self._apply_1d_rope(q, k)
        else:
            q, k = self._apply_2d_rope(q, k, input_resolution)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(B, L, C)
        return self.o_proj(x)


class SpatialTiledTokenFusionAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, num_kv_heads=None,
                 tile_tokens=None, group_tokens=4, group_token_counts=None, eval_group_tokens=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.tile_tokens = None if tile_tokens is None else max(1, int(tile_tokens))
        self.group_tokens = max(1, int(group_tokens))
        self.group_token_counts = self._normalize_group_token_counts(group_token_counts)
        self.eval_group_tokens = None if eval_group_tokens is None else max(1, int(eval_group_tokens))
        max_group_tokens = max(self.group_token_choices())
        self.max_fusion_group_size = max(1, (self.tile_token_count() + max_group_tokens - 1) // max_group_tokens)
        self.merge_logits = nn.Parameter(torch.zeros(self.tile_token_count(), self.max_fusion_group_size))
        self.attn = DeepSeekGQAAttention(dim, num_heads=num_heads, num_kv_heads=num_kv_heads)

    @staticmethod
    def _normalize_group_token_counts(group_token_counts):
        if group_token_counts is None:
            return None
        counts = []
        for count in group_token_counts:
            count = max(1, int(count))
            if count not in counts:
                counts.append(count)
        return tuple(counts)

    def group_token_choices(self):
        if self.group_token_counts is None:
            return (self.group_tokens,)
        return self.group_token_counts

    def default_eval_group_tokens(self):
        if self.eval_group_tokens is not None:
            return self.eval_group_tokens
        if self.group_token_counts:
            return max(self.group_token_counts)
        return self.group_tokens

    def tile_token_count(self):
        H, W = self.input_resolution
        if self.tile_tokens is None:
            return min(self.group_tokens, max(1, H * W))
        return min(self.tile_tokens, max(1, H * W))

    def _tile_shape(self):
        H, W = self.input_resolution
        tile_tokens = self.tile_token_count()
        root = int(tile_tokens ** 0.5)
        for tile_h in range(min(root, H), 0, -1):
            if tile_tokens % tile_h == 0:
                tile_w = tile_tokens // tile_h
                if tile_w <= W:
                    return tile_h, tile_w
        tile_h = min(H, tile_tokens)
        tile_w = max(1, min(W, (tile_tokens + tile_h - 1) // tile_h))
        return tile_h, tile_w

    def num_groups(self):
        H, W = self.input_resolution
        tile_h, tile_w = self._tile_shape()
        grid_h = (H + tile_h - 1) // tile_h
        grid_w = (W + tile_w - 1) // tile_w
        return max(1, grid_h * grid_w)

    def _fusion_shape(self, token_count, group_tokens=None):
        group_tokens = max(1, int(self.group_tokens if group_tokens is None else group_tokens))
        group_tokens = min(group_tokens, max(1, token_count))
        fusion_group_size = max(1, (token_count + group_tokens - 1) // group_tokens)
        fusion_groups = max(1, (token_count + fusion_group_size - 1) // fusion_group_size)
        return fusion_groups, fusion_group_size

    def _merge_logits(self, fusion_groups, fusion_group_size):
        logits = self.merge_logits[:fusion_groups, :fusion_group_size]
        if logits.shape == (fusion_groups, fusion_group_size):
            return logits
        logits = self.merge_logits.unsqueeze(0).unsqueeze(0)
        logits = F.interpolate(logits, size=(fusion_groups, fusion_group_size), mode="bilinear", align_corners=False)
        return logits.squeeze(0).squeeze(0)

    def _fuse_tokens(self, x, valid, group_tokens=None):
        B, L, C = x.shape
        fusion_groups, fusion_group_size = self._fusion_shape(L, group_tokens=group_tokens)
        padded_len = fusion_groups * fusion_group_size
        if padded_len > L:
            x = F.pad(x, (0, 0, 0, padded_len - L))
            valid = F.pad(valid, (0, padded_len - L), value=False)

        grouped = x.view(B, fusion_groups, fusion_group_size, C)
        valid = valid.view(B, fusion_groups, fusion_group_size)
        logits = self._merge_logits(fusion_groups, fusion_group_size).to(dtype=x.dtype, device=x.device)
        logits = logits.view(1, fusion_groups, fusion_group_size).expand(B, -1, -1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        all_invalid = ~valid.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(all_invalid, 0)
        weights = F.softmax(logits, dim=-1).masked_fill(~valid, 0)
        fused = (grouped * weights.unsqueeze(-1)).sum(dim=2)
        fused_valid = valid.any(dim=-1)
        return fused, fused_valid

    def forward(self, x, group_tokens=None):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"

        tile_h, tile_w = self._tile_shape()
        pad_h = (tile_h - H % tile_h) % tile_h
        pad_w = (tile_w - W % tile_w) % tile_w
        padded_H = H + pad_h
        padded_W = W + pad_w

        x_2d = x.view(B, H, W, C)
        if pad_h or pad_w:
            x_2d = F.pad(x_2d.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)

        grid_h = padded_H // tile_h
        grid_w = padded_W // tile_w
        grouped = x_2d.view(B, grid_h, tile_h, grid_w, tile_w, C)
        grouped = grouped.permute(0, 1, 3, 2, 4, 5).contiguous()
        tile_count = grid_h * grid_w
        tile_token_count = tile_h * tile_w

        # 每个片内单独计算
        grouped = grouped.view(B * tile_count, tile_token_count, C)

        valid = torch.ones(B * tile_count, tile_token_count, dtype=torch.bool, device=x.device)
        if pad_h or pad_w:
            valid_2d = torch.zeros(padded_H, padded_W, dtype=torch.bool, device=x.device)
            valid_2d[:H, :W] = True
            valid = valid_2d.view(grid_h, tile_h, grid_w, tile_w)
            valid = valid.permute(0, 2, 1, 3).contiguous().view(tile_count, tile_token_count)
            valid = valid.unsqueeze(0).expand(B, -1, -1).reshape(B * tile_count, tile_token_count)

        selected_group_tokens = max(1, int(self.group_tokens if group_tokens is None else group_tokens))

        # 判断现在要加权生成的tokens数目是不是超过了每个片本身
        if selected_group_tokens >= tile_token_count:
            attn_mask = valid[:, None, None, :] if not valid.all() else None
            grouped = self.attn(grouped, input_resolution=(tile_h, tile_w), attn_mask=attn_mask)
        else:
            # valid是一个标记，让填充的部分不参与计算
            fused_tokens, fused_valid = self._fuse_tokens(grouped, valid, group_tokens=selected_group_tokens)
            # 使用分组Attention
            attn_mask = fused_valid[:, None, None, :]
            grouped = self.attn(grouped, kv_tokens=fused_tokens, attn_mask=attn_mask)

        grouped = grouped.view(B, grid_h, grid_w, tile_h, tile_w, C)
        x_2d = grouped.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, padded_H, padded_W, C)
        return x_2d[:, :H, :W, :].contiguous().view(B, L, C)

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)

    def flops(self, group_tokens=None):
        H, W = self.input_resolution
        L = H * W
        tile_h, tile_w = self._tile_shape()
        tile_token_count = tile_h * tile_w
        tile_count = self.num_groups()
        padded_tokens = tile_count * tile_token_count
        selected_group_tokens = max(1, int(self.group_tokens if group_tokens is None else group_tokens))
        if selected_group_tokens >= tile_token_count:
            attn_k = tile_token_count
            flops = 3 * padded_tokens * self.dim * self.dim
        else:
            fusion_groups, fusion_group_size = self._fusion_shape(tile_token_count, group_tokens=selected_group_tokens)
            attn_k = fusion_groups
            flops = tile_count * fusion_groups * fusion_group_size * self.dim
            flops += tile_count * tile_token_count * self.dim * self.dim
            flops += 2 * tile_count * attn_k * self.dim * self.dim
        flops += 2 * tile_count * tile_token_count * attn_k * self.dim
        flops += padded_tokens * self.dim * self.dim
        return flops + max(0, padded_tokens - L) * self.dim


class GroupedTokenFusion(nn.Module):
    def __init__(self, dim, group_size=8):
        super().__init__()
        self.dim = dim
        self.group_size = max(1, int(group_size))
        self.merge_logits = nn.Parameter(torch.zeros(self.group_size))

    def _group_shape(self, token_count):
        token_count = max(1, int(token_count))
        groups = max(1, (token_count + self.group_size - 1) // self.group_size)
        return groups, self.group_size

    def num_groups(self, token_count):
        groups, _ = self._group_shape(token_count)
        return groups

    def forward(self, x):
        B, L, C = x.shape
        groups, group_size = self._group_shape(L)
        print(groups, group_size)
        padded_len = groups * group_size
        if padded_len > L:
            x = F.pad(x, (0, 0, 0, padded_len - L))

        grouped = x.view(B, groups, group_size, C)
        token_ids = torch.arange(padded_len, device=x.device).view(groups, group_size)
        valid = token_ids < L
        logits = self.merge_logits.to(dtype=x.dtype, device=x.device).view(1, group_size).expand(groups, -1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=-1)
        return (grouped * weights.view(1, groups, group_size, 1)).sum(dim=2)

    def flops(self, token_count):
        groups, group_size = self._group_shape(token_count)
        return groups * group_size * self.dim


class DeepSeekRecoveryBlock(nn.Module):
    """DeepSeek-style image token block for spatial feature recovery.

    The block borrows RMSNorm, GQA and SwiGLU from modern large-model blocks,
    but uses non-causal 2D RoPE so spatial image tokens can attend globally.
    """

    def __init__(self, dim, input_resolution, num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 hca_enabled=False, hca_tile_tokens=None, hca_group_tokens=4,
                 hca_group_token_counts=None,
                 hca_eval_group_tokens=None,
                 hca_selector_hidden_ratio=2.0,
                 csa_group_size=None,
                 coarse_group_size=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.num_kv_heads = max(1, num_heads // 2)
        self.hca_enabled = bool(hca_enabled)
        self.csa_group_size = None if csa_group_size is None else max(1, int(csa_group_size))
        self.coarse_group_size = None if coarse_group_size is None else max(1, int(coarse_group_size))

        hidden_dim = int(dim * mlp_ratio)
        self.attn_norm = RMSNorm(dim)
        self.attn = DeepSeekGQAAttention(dim, num_heads=num_heads, num_kv_heads=self.num_kv_heads)
        self.hca_attention = None
        self.hca_selector = None
        self.csa_fusion = None
        self.coarse_fusion = None
        self.coarse_attn = None
        self.branch_logits = None
        if self.hca_enabled:
            self.hca_attention = SpatialTiledTokenFusionAttention(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                tile_tokens=hca_tile_tokens,
                group_tokens=hca_group_tokens,
                group_token_counts=hca_group_token_counts,
                eval_group_tokens=hca_eval_group_tokens,
            )
        elif self.csa_group_size is not None and self.csa_group_size > 1:
            self.csa_fusion = GroupedTokenFusion(dim=dim, group_size=self.csa_group_size)
        if self.coarse_group_size is not None and self.coarse_group_size > 1:
            self.coarse_fusion = GroupedTokenFusion(dim=dim, group_size=self.coarse_group_size)
            self.coarse_attn = DeepSeekGQAAttention(dim, num_heads=num_heads, num_kv_heads=self.num_kv_heads)
            self.branch_logits = nn.Parameter(torch.zeros(2))
        self.ffn_norm = RMSNorm(dim)
        self.ffn = DeepSeekSwiGLU(dim, hidden_dim)
        self.recover_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, hca_group_tokens=None):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"

        attn_input = self.attn_norm(x)
        if self.hca_attention is not None:
            attn_out = self.hca_attention(attn_input, group_tokens=hca_group_tokens)
        else:
            kv_tokens = None
            if self.csa_fusion is not None:
                kv_tokens = self.csa_fusion(attn_input)
            attn_out = self.attn(attn_input, input_resolution=(H, W), kv_tokens=kv_tokens)
        if self.coarse_fusion is not None and self.hca_attention is None:
            coarse_tokens = self.coarse_fusion(attn_input)
            coarse_out = self.coarse_attn(attn_input, input_resolution=(H, W), kv_tokens=coarse_tokens)
            branch_weights = F.softmax(self.branch_logits.to(dtype=x.dtype), dim=0)
            attn_out = branch_weights[0] * attn_out + branch_weights[1] * coarse_out
        hidden = x + attn_out
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return x + self.recover_scale * (hidden - x)

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        if self.hca_attention is not None:
            self.hca_attention.update_resolution(H, W)

    def flops(self, hca_group_tokens=None):
        H, W = self.input_resolution
        L = H * W
        hidden_dim = self.ffn.gate_proj.out_features
        if self.hca_attention is not None:
            flops = self.hca_attention.flops(hca_group_tokens)
        else:
            if self.csa_fusion is not None:
                attn_k = self.csa_fusion.num_groups(L)
                flops = self.csa_fusion.flops(L)
                flops += L * self.dim * self.dim
                flops += 2 * attn_k * self.dim * self.dim
            else:
                attn_k = L
                flops = 0
                flops += 3 * L * self.dim * self.dim
            flops += L * self.dim * self.dim
            flops += 2 * L * attn_k * self.dim
            if self.coarse_fusion is not None:
                coarse_k = self.coarse_fusion.num_groups(L)
                flops += self.coarse_fusion.flops(L)
                flops += L * self.dim * self.dim
                flops += 2 * coarse_k * self.dim * self.dim
                flops += L * self.dim * self.dim
                flops += 2 * L * coarse_k * self.dim
                flops += 2 * L * self.dim
        flops += 3 * L * self.dim * hidden_dim
        return flops
