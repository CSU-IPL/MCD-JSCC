# Rician channel

`channel_type="rician"` implements the complex-baseband channel

\[
    y = h x + n, \qquad n \sim \mathcal{CN}(0, \sigma_n^2 I),
\]

using amplitude-only fading. For each complex symbol, it samples the complex
Rician coefficient

\[
    h_c = \sqrt{\frac{K}{K+1}} + \sqrt{\frac{1}{K+1}}g,
    \qquad g \sim \mathcal{CN}(0,1),
\]

and applies only its non-negative amplitude, `h = |h_c|`; it does not rotate
the received signal phase. Here, `K` is the LoS-to-scattered-power ratio. The default is
`rician_k_factor = 10.0` (linear scale) in `config.py`. Setting `K = 0`
reduces the fading-amplitude distribution to Rayleigh; a larger K gives a
stronger LoS component.

## Running the base components

The partial release includes a forward-pass example using random weights:

```sh
python demo.py --channel rician --snr 10
```

`Channel.sample_channel_state()` returns reusable noise and fading gains.
Pass that state as `channel_state` to reuse the same channel realization.
AWGN shares only noise; Rayleigh and Rician also share non-negative gains.

Noise follows the existing convention:

```python
sigma = sqrt(1 / (2 * 10**(snr_db / 10)))
```
