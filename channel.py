import math

import torch
import torch.nn as nn


class Channel(nn.Module):
    """Complex baseband channel models used by the semantic communication models.

    ``rician`` implements ``y = |h| * x + n`` with
    ``h = sqrt(K/(K+1)) + sqrt(1/(K+1)) g``, where ``g`` is unit-power
    circular complex Gaussian noise and ``K`` is the Rician K factor. Only
    the non-negative amplitude ``|h|`` is applied, so there is no phase shift.
    """

    RICIAN_TYPES = (3, "rician")

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.chan_type = None
        self.rician_k_factor = float(getattr(config, "rician_k_factor", 10.0))

    @staticmethod
    def _complex_gaussian(shape, std, device, dtype=torch.float32):
        real = torch.normal(0.0, std, size=shape, device=device, dtype=dtype)
        imag = torch.normal(0.0, std, size=shape, device=device, dtype=dtype)
        return torch.complex(real, imag)

    def _noise_for(self, channel_in, std, noise_state=None):
        if noise_state is None:
            return self._complex_gaussian(
                channel_in.shape, std, channel_in.device, channel_in.real.dtype,
            )
        noise = torch.as_tensor(noise_state, device=channel_in.device)
        if noise.numel() != channel_in.numel():
            raise ValueError(
                "Shared channel noise has incompatible size: "
                f"got {noise.numel()}, expected {channel_in.numel()}.",
            )
        return noise.reshape(channel_in.shape).to(dtype=channel_in.dtype).detach()

    def gaussian_noise_layer(self, input_layer, std, name=None, noise_state=None):
        noise = self._noise_for(input_layer, std, noise_state)
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        noise = self._complex_gaussian(
            input_layer.shape, std, input_layer.device, input_layer.real.dtype,
        )
        h = self._sample_rayleigh_state(input_layer)
        return input_layer * h + noise

    @staticmethod
    def _to_complex_symbols(input_layer):
        flat_input = input_layer.reshape(-1)
        if flat_input.numel() % 2:
            raise ValueError("Channel input must contain an even number of real values.")
        half_length = flat_input.numel() // 2
        return flat_input[:half_length] + flat_input[half_length:] * 1j

    @staticmethod
    def _is_rayleigh(channel_type):
        return channel_type == 2 or channel_type == "rayleigh"

    @staticmethod
    def _is_rician(channel_type):
        return channel_type in Channel.RICIAN_TYPES

    def _sample_rayleigh_state(self, channel_in):
        """Sample the repository's real, amplitude-only Rayleigh gain."""
        return torch.sqrt(
            torch.normal(0.0, 1.0, size=channel_in.shape, device=channel_in.device) ** 2
            + torch.normal(0.0, 1.0, size=channel_in.shape, device=channel_in.device) ** 2
        ) / math.sqrt(2.0)

    def _sample_rician_state(self, channel_in):
        """Sample independent non-negative Rician fading gains per symbol."""
        k_factor = max(self.rician_k_factor, 0.0)
        scatter = self._complex_gaussian(
            channel_in.shape,
            1.0 / math.sqrt(2.0),
            channel_in.device,
            channel_in.real.dtype,
        )
        los_amplitude = math.sqrt(k_factor / (k_factor + 1.0))
        scatter_amplitude = math.sqrt(1.0 / (k_factor + 1.0))
        return torch.abs(los_amplitude + scatter_amplitude * scatter)

    def sample_channel_state(self, input_layer, channel_type, chan_param):
        """Sample a reusable channel realization with noise and fading gains.

        Rayleigh and Rician have one non-negative real gain per complex symbol.
        Pass the returned state to forward() to reuse the same realization.
        """
        channel_in = self._to_complex_symbols(input_layer)
        sigma = math.sqrt(1.0 / (2.0 * 10.0 ** (chan_param / 10.0)))
        fading = None
        if self._is_rayleigh(channel_type):
            fading = self._sample_rayleigh_state(channel_in).detach()
        elif self._is_rician(channel_type):
            fading = self._sample_rician_state(channel_in).detach()
        return {
            "fading": fading,
            "noise": self._noise_for(channel_in, sigma).detach(),
        }

    @staticmethod
    def _split_channel_state(channel_state):
        """Accept both a full shared state and the legacy raw fading scalar."""
        if isinstance(channel_state, dict):
            return channel_state.get("fading"), channel_state.get("noise")
        return channel_state, None

    def _rayleigh_state_for(self, channel_in, channel_state=None):
        if channel_state is None:
            return self._sample_rayleigh_state(channel_in)
        h = torch.as_tensor(channel_state, device=channel_in.device)
        if h.numel() not in (1, channel_in.numel()):
            raise ValueError(
                "Shared Rayleigh fading state has incompatible size: "
                f"got {h.numel()}, expected 1 or {channel_in.numel()}.",
            )
        return torch.abs(h).to(dtype=channel_in.real.dtype).detach()

    def _rician_state_for(self, channel_in, channel_state=None):
        if channel_state is None:
            return self._sample_rician_state(channel_in)
        h = torch.as_tensor(channel_state, device=channel_in.device)
        if h.numel() not in (1, channel_in.numel()):
            raise ValueError(
                "Shared Rician fading state has incompatible size: "
                f"got {h.numel()}, expected 1 or {channel_in.numel()}.",
            )
        return torch.abs(h).to(dtype=channel_in.real.dtype).detach()

    @staticmethod
    def complex_normalize(x, power):
        pwr = torch.mean(x ** 2) * 2
        out = math.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    def forward(self, input, chan_param, channel_type=None, avg_pwr=False, channel_state=None):
        self.chan_type = channel_type
        if avg_pwr:
            channel_tx = input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)

        input_shape = channel_tx.shape
        channel_in = self._to_complex_symbols(channel_tx)
        channel_output = self.complex_forward(channel_in, chan_param, channel_state)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)

        if self.chan_type == 1 or self.chan_type == "awgn":
            # Keep the legacy AWGN straight-through treatment unchanged.
            noise = (channel_output - channel_tx).detach()
            channel_tx = channel_tx + noise
            return channel_tx * torch.sqrt(avg_pwr * 2) if avg_pwr else channel_tx * torch.sqrt(pwr)
        if (self.chan_type == 0 or self.chan_type == "none" or self.chan_type == 2
                or self.chan_type == "rayleigh" or self._is_rician(self.chan_type)):
            return channel_output * torch.sqrt(avg_pwr * 2) if avg_pwr else channel_output * torch.sqrt(pwr)
        raise ValueError(f"Unsupported channel type: {self.chan_type!r}")

    def complex_forward(self, channel_in, chan_param, channel_state=None):
        if self.chan_type == 0 or self.chan_type == "none":
            return channel_in

        sigma = math.sqrt(1.0 / (2.0 * 10.0 ** (chan_param / 10.0)))
        fading_state, noise_state = self._split_channel_state(channel_state)
        if self.chan_type == 1 or self.chan_type == "awgn":
            return self.gaussian_noise_layer(
                channel_in, std=sigma, name="awgn_chan_noise", noise_state=noise_state,
            )
        if self._is_rayleigh(self.chan_type):
            noise = self._noise_for(channel_in, sigma, noise_state)
            h = self._rayleigh_state_for(channel_in, fading_state)
            return channel_in * h + noise
        if self._is_rician(self.chan_type):
            noise = self._noise_for(channel_in, sigma, noise_state)
            h = self._rician_state_for(channel_in, fading_state)
            return channel_in * h + noise
        raise ValueError(f"Unsupported channel type: {self.chan_type!r}")

    def noiseless_forward(self, channel_in):
        channel_tx, _ = self.complex_normalize(channel_in, power=1)
        return channel_tx
