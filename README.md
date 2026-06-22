# pypoly

Polyphase analysis/synthesis channelizer primitives in Python, recreating a slice of the
functionality in MATLAB's DSP System Toolbox (`dsp.Channelizer` / polyphase filter banks).

## Status

| Component | Status |
|---|---|
| `design_prototype_filter` | Working. Designs a Kaiser-windowed lowpass prototype, with a sanity-checking `UserWarning` for cutoffs that are too wide to ever separate channels (see [Oversampling](#oversampling-decimation) below). |
| `PolyphaseAnalysisChannelizer` | Working and tested. Splits a signal into `num_channels` frequency channels via a polyphase filter bank + FFT. Supports both critical sampling and integer-ratio oversampling. |
| `PolyphaseSynthesisChannelizer` | **Broken.** Does not correctly invert `PolyphaseAnalysisChannelizer` -- round-trip reconstruction error is ~0.97 (normalized), i.e. the output is essentially uncorrelated with the original input, not just imperfect. See the `TODO` on the class in [`src/pypoly/channelizer.py`](src/pypoly/channelizer.py) and [`notebooks/02_analysis_synthesis_reconstruction.ipynb`](notebooks/02_analysis_synthesis_reconstruction.ipynb) for a reproduction. Root cause: the synthesis kernel's output commutator doesn't mirror the analysis kernel's input commutator direction, so decimation images aren't cancelled. Not yet fixed -- deprioritized relative to the analysis side. |

If you only need to split a signal into channels (not reconstruct it), `PolyphaseAnalysisChannelizer`
is ready to use. If you need the full analysis -> synthesis round trip, that part is not yet correct.

## Install

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs `pypoly` itself (editable) plus its runtime dependencies (`numpy`, `scipy`, `numba`).
The notebooks additionally need the `dev` dependency group (`matplotlib`, `ipykernel`, `ipympl`),
which `uv sync` installs by default as well.

## Usage

### Basic (critically sampled) channelizer

```python
import numpy as np
from pypoly import PolyphaseAnalysisChannelizer

analysis = PolyphaseAnalysisChannelizer.from_design(num_channels=8, taps_per_channel=16)

samples = np.random.randn(4096) + 1j * np.random.randn(4096)
channels = analysis.process(samples)   # shape: (8, 512) -- one row per channel

# A tone at input frequency k0 * fs / num_channels lands in output channel k0.
```

### Oversampling (`decimation`)

Critical sampling (the default) decimates by `num_channels`, so each channel is sampled right at
its own Nyquist rate. Because the prototype filter's transition band isn't a brick wall, a signal
sitting near the crossover between two adjacent channels aliases back onto itself rather than
landing at a distinguishable frequency. Oversampling (decimating by less than `num_channels`) fixes
this:

```python
from pypoly import PolyphaseAnalysisChannelizer

# 2x oversampling: decimate by num_channels/2 instead of num_channels.
analysis = PolyphaseAnalysisChannelizer.from_design(
    num_channels=8, taps_per_channel=16, decimation=4
)
analysis.oversample_rate  # 2.0 (derived, read-only -- num_channels / decimation)
```

`decimation` must be an integer in `[1, num_channels]`; `from_design` also auto-widens the
prototype filter's cutoff proportionally (`cutoff_ratio = 0.9 * num_channels / decimation`) so the
extra Nyquist headroom from oversampling is actually used to flatten the channel-edge response, not
just to avoid aliasing.

This auto-scaling breaks down at very aggressive oversampling (small `decimation`): a channel's
farthest neighbor always sits at exactly half of Nyquist, so once the auto-widened cutoff reaches
that point there's no stopband left at all, regardless of filter length. `design_prototype_filter`
raises a `UserWarning` in that case rather than silently handing back a non-functional filter; pass
an explicit, narrower `cutoff_ratio` if you intentionally need that decimation factor with real
channel separation. See
[`notebooks/03_oversampled_channelizer.ipynb`](notebooks/03_oversampled_channelizer.ipynb) for a
demonstration of both the benefit and this limit.

## Notebooks

The `notebooks/` directory has runnable demonstrations/tests with plots, complementing the unit
tests in `tests/`:

- [`01_prototype_filter_and_channel_mapping.ipynb`](notebooks/01_prototype_filter_and_channel_mapping.ipynb) -- prototype filter frequency response, and the channel-to-frequency mapping check (a tone at channel `k` should land in output bin `k`).
- [`02_analysis_synthesis_reconstruction.ipynb`](notebooks/02_analysis_synthesis_reconstruction.ipynb) -- reproduces the open `PolyphaseSynthesisChannelizer` reconstruction bug described above.
- [`03_oversampled_channelizer.ipynb`](notebooks/03_oversampled_channelizer.ipynb) -- demonstrates `decimation`/oversampling: the channel mapping still holds, the in-channel aliasing it fixes, and the warning/limit at very aggressive oversampling.

Each notebook bootstraps `sys.path` to import `pypoly` from `src/` directly, so it picks up local
edits without reinstalling.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```
