import torch, os
from copy import deepcopy
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import torch.nn as nn

from distortion import *

class config():
    seed = 1024
    pass_channel = True
    CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if CUDA else "cpu")
    norm = False
    out_dim = 32

    # Rician K factor (linear scale): LoS-to-scattered power ratio.
    rician_k_factor = 10.0

    data_root = os.environ.get('DATASET_ROOT', 'data')
    trainset = 'DIV2K'
    testset = 'Kodak'

    train_img_size = 128
    train_batch_size = 8
    test_batch_size = 1
    image_dims = (3, train_img_size, train_img_size)
    downsample = 2
    s_encoder_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
        embed_dims=[128, 256], depths=[2, 4], num_heads=[2, 4], C=out_dim,
        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
        deepseek_num_heads=4, image_recovery_depth=4,
        encoder_hca_enabled=True,
        encoder_hca_tile_tokens=256,
        encoder_hca_group_tokens=256,
        encoder_hca_group_token_counts=None,
        encoder_hca_eval_group_tokens=256,
        encoder_coarse_group_size=64,
        encoder_csa_group_size=None,
    )
    s_decoder_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=out_dim,
        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
        deepseek_num_heads=8, image_recovery_depth=4,
        hca_enabled=True, hca_tile_tokens=512, hca_group_tokens=128,
        hca_group_token_counts=(64, 128, 256, 512),
        hca_eval_group_tokens=512,
        hca_selector_hidden_ratio=2.0,
        decoder_coarse_group_size=64,
    )
