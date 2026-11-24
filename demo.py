"""Run a forward-pass smoke check with randomly initialized base components."""

import argparse
from copy import deepcopy
from types import SimpleNamespace

import torch

from channel import Channel
from config import config
from models import Receiver, Transmitter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', choices=['none', 'awgn', 'rayleigh', 'rician'], default='awgn')
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--rate', type=float, default=0.33, help='Conditioning value for the base adapters')
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.img_size <= 0 or args.img_size % 32:
        parser.error('--img-size must be a positive multiple of 32')
    if not 0 < args.rate <= 1:
        parser.error('--rate must be in (0, 1]')

    torch.manual_seed(config.seed)
    device = torch.device(args.device)
    encoder_kwargs = deepcopy(config.s_encoder_kwargs)
    decoder_kwargs = deepcopy(config.s_decoder_kwargs)
    encoder_kwargs['img_size'] = (args.img_size, args.img_size)
    decoder_kwargs['img_size'] = (args.img_size, args.img_size)
    transmitter = Transmitter(encoder_kwargs, args).to(device).eval()
    receiver = Receiver(decoder_kwargs, args).to(device).eval()
    channel = Channel(SimpleNamespace(device=device, rician_k_factor=config.rician_k_factor)).to(device)

    with torch.no_grad():
        image = torch.rand(1, 3, args.img_size, args.img_size, device=device)
        code = transmitter(image, args.rate)
        received = channel(code, args.snr, channel_type=args.channel)
        reconstruction = receiver(received, args.rate)
    if reconstruction.shape != image.shape or not torch.isfinite(reconstruction).all():
        raise RuntimeError('Forward pass produced an invalid reconstruction')
    print(f'Input: {tuple(image.shape)}, code: {tuple(code.shape)}, output: {tuple(reconstruction.shape)}')
    print('Forward pass passed with random weights. This example does not measure reconstruction quality.')


if __name__ == '__main__':
    main()
