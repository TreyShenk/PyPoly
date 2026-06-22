from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.signal import firwin

# A channel's farthest neighbor always sits at exactly half the Nyquist band,
# regardless of num_channels. Once the prototype's cutoff reaches that point, the
# "stopband" at the opposite channel is inside the passband -- no amount of taps can
# fix that, it's structural. Measured empirically: cutoff=0.45 still gives ~-82dB
# rejection there; cutoff=0.5 collapses to ~-6dB and beyond that it's ~0dB (no
# rejection at all). 0.45 is the warning threshold, with margin before the cliff.
_MAX_RECOMMENDED_CUTOFF = 0.45


def design_prototype_filter(
    num_channels: int,
    taps_per_channel: int,
    *,
    cutoff_ratio: float = 0.9,
    window: tuple[str, float] = ("kaiser", 8.0),
) -> np.ndarray:
    """Design a low-pass prototype for an (optionally oversampled) DFT filter bank.

    ``cutoff_ratio`` is expressed relative to one channel's critically-sampled
    half-bandwidth (``fs / (2 * num_channels)``), so a value of ``1.0`` is the
    critically-sampled Nyquist edge. When the prototype is meant to pair with an
    oversampled channelizer (decimation factor below ``num_channels``), pass a
    correspondingly larger ``cutoff_ratio`` -- see
    ``PolyphaseAnalysisChannelizer.from_design``'s ``decimation`` handling.
    """
    if num_channels < 2:
        raise ValueError("num_channels must be at least 2")
    if taps_per_channel < 2:
        raise ValueError("taps_per_channel must be at least 2")
    if not (0.0 < cutoff_ratio / num_channels <= 1.0):
        raise ValueError("cutoff_ratio / num_channels must be in (0, 1]")

    num_taps = num_channels * taps_per_channel
    cutoff = cutoff_ratio / num_channels
    if cutoff > _MAX_RECOMMENDED_CUTOFF:
        warnings.warn(
            f"cutoff_ratio={cutoff_ratio!r} gives a prototype cutoff of "
            f"{cutoff:.3f} (x Nyquist), above the recommended "
            f"{_MAX_RECOMMENDED_CUTOFF} limit. A channel's farthest neighbor sits at "
            "0.5 x Nyquist regardless of num_channels, so cutoffs this wide leave "
            "little or no stopband margin -- the resulting filter may not separate "
            "channels at all. This typically happens when oversampling very "
            "aggressively (a small decimation factor) with the default auto-scaled "
            "cutoff_ratio; pass an explicit, narrower cutoff_ratio if you need this "
            "decimation factor with real channel separation.",
            stacklevel=2,
        )
    taps = firwin(num_taps, cutoff, window=window)
    return taps.astype(np.float64, copy=False)


@njit(cache=True)
def _analysis_polyphase_kernel(
    samples: np.ndarray, phases: np.ndarray, num_blocks: int, block_stride: int
) -> np.ndarray:
    m_channels, taps_per_phase = phases.shape
    out = np.zeros((m_channels, num_blocks), dtype=np.complex128)

    for n in range(num_blocks):
        base = n * block_stride
        for phase_idx in range(m_channels):
            acc = 0.0 + 0.0j
            for tap_idx in range(taps_per_phase):
                sample_idx = base - tap_idx * m_channels - phase_idx
                if 0 <= sample_idx < samples.size:
                    acc += phases[phase_idx, tap_idx] * samples[sample_idx]
            out[phase_idx, n] = acc
    return out


@njit(cache=True)
def _synthesis_polyphase_kernel(
    phase_samples: np.ndarray, synth_phases: np.ndarray
) -> np.ndarray:
    m_channels, num_blocks = phase_samples.shape
    _, taps_per_phase = synth_phases.shape
    out = np.zeros(num_blocks * m_channels, dtype=np.complex128)

    for n in range(num_blocks):
        base = n * m_channels
        for phase_idx in range(m_channels):
            acc = 0.0 + 0.0j
            for tap_idx in range(taps_per_phase):
                block_idx = n - tap_idx
                if block_idx >= 0:
                    acc += (
                        synth_phases[phase_idx, tap_idx]
                        * phase_samples[phase_idx, block_idx]
                    )
            out[base + phase_idx] = acc
    return out


@dataclass(frozen=True)
class PolyphaseAnalysisChannelizer:
    num_channels: int
    prototype_taps: np.ndarray
    # Decimation factor D: a block of channel outputs is produced every D input
    # samples. D in [1, num_channels] is the only meaningful range -- D=num_channels
    # is critically sampled, D<num_channels oversamples (matches GNU Radio's
    # pfb_channelizer's integer "i" in oversample_rate = N/i). This is deliberately
    # an int, not a derived oversample_rate float: the valid values are a discrete
    # set, and round-tripping a float back to the integer D it came from is fragile.
    decimation: int | None = None

    def __post_init__(self) -> None:
        taps = np.asarray(self.prototype_taps, dtype=np.float64)
        if taps.ndim != 1:
            raise ValueError("prototype_taps must be a 1D array")
        if taps.size % self.num_channels != 0:
            raise ValueError("prototype_taps length must be divisible by num_channels")

        decimation = self.num_channels if self.decimation is None else self.decimation
        if not (1 <= decimation <= self.num_channels):
            raise ValueError("decimation must be an integer in [1, num_channels]")

        object.__setattr__(self, "_phases", taps.reshape(-1, self.num_channels).T.copy())
        object.__setattr__(self, "_decimation", decimation)

    @property
    def oversample_rate(self) -> float:
        """Informational only: num_channels / decimation. Not a constructor input."""
        return self.num_channels / self._decimation

    @classmethod
    def from_design(
        cls,
        num_channels: int,
        taps_per_channel: int,
        *,
        cutoff_ratio: float | None = None,
        decimation: int | None = None,
    ) -> PolyphaseAnalysisChannelizer:
        # The prototype's occupied bandwidth must grow as decimation shrinks (i.e. as
        # oversampling increases) to use the extra Nyquist headroom a smaller D
        # provides -- otherwise oversampling removes in-channel aliasing at crossover
        # frequencies but leaves them just as attenuated as the critically-sampled
        # case. 0.9 is the existing critically-sampled headroom below the Nyquist
        # edge; scaling it by num_channels/D keeps the same margin while flattening
        # crossover response.
        if cutoff_ratio is None:
            effective_decimation = num_channels if decimation is None else decimation
            if not (1 <= effective_decimation <= num_channels):
                raise ValueError("decimation must be an integer in [1, num_channels]")
            cutoff_ratio = 0.9 * num_channels / effective_decimation
        taps = design_prototype_filter(
            num_channels=num_channels,
            taps_per_channel=taps_per_channel,
            cutoff_ratio=cutoff_ratio,
        )
        return cls(num_channels=num_channels, prototype_taps=taps, decimation=decimation)

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.complex128).reshape(-1)
        if x.size == 0:
            return np.zeros((self.num_channels, 0), dtype=np.complex128)

        num_blocks = int(np.ceil(x.size / self._decimation))
        phases_out = _analysis_polyphase_kernel(x, self._phases, num_blocks, self._decimation)
        channels = np.fft.ifft(phases_out, axis=0)

        if self._decimation != self.num_channels:
            # Block m starts at sample m*D, which only lands on a num_channels-aligned
            # boundary when D divides evenly into it. The misalignment rotates each
            # channel's IFFT phase reference by exp(-j*2*pi*k*shift/M), shift = (m*D) % M;
            # undo it so consecutive blocks stay phase-coherent (verified against the
            # expected demodulated tone frequency for non-power-of-two D in
            # notebooks/03_oversampled_channelizer.ipynb).
            m_idx = np.arange(num_blocks)
            shift = (m_idx * self._decimation) % self.num_channels
            k_idx = np.arange(self.num_channels).reshape(-1, 1)
            correction = np.exp(-1j * 2 * np.pi * k_idx * shift.reshape(1, -1) / self.num_channels)
            channels *= correction

        return channels


@dataclass(frozen=True)
class PolyphaseSynthesisChannelizer:
    # TODO: analysis -> synthesis round trip does not reconstruct the input.
    # PolyphaseAnalysisChannelizer.process feeds samples to branch p in order
    # x[nM - p], but _synthesis_polyphase_kernel writes branch p's output to
    # out[nM + p] -- the commutator directions don't mirror each other, so
    # the M-fold decimation images from analysis aren't cancelled here.
    # See notebooks/02_analysis_synthesis_reconstruction.ipynb for a
    # reproduction (normalized reconstruction error ~0.97, i.e. uncorrelated
    # with the input). Needs a fix to the output indexing in
    # _synthesis_polyphase_kernel before this class can be trusted.
    num_channels: int
    prototype_taps: np.ndarray

    def __post_init__(self) -> None:
        taps = np.asarray(self.prototype_taps, dtype=np.float64)
        if taps.ndim != 1:
            raise ValueError("prototype_taps must be a 1D array")
        if taps.size % self.num_channels != 0:
            raise ValueError("prototype_taps length must be divisible by num_channels")
        object.__setattr__(self, "_phases", taps.reshape(-1, self.num_channels).T.copy())

    @classmethod
    def from_design(
        cls, num_channels: int, taps_per_channel: int, *, cutoff_ratio: float = 0.9
    ) -> PolyphaseSynthesisChannelizer:
        taps = design_prototype_filter(
            num_channels=num_channels,
            taps_per_channel=taps_per_channel,
            cutoff_ratio=cutoff_ratio,
        )
        return cls(num_channels=num_channels, prototype_taps=taps)

    def process(self, channel_samples: np.ndarray) -> np.ndarray:
        xk = np.asarray(channel_samples, dtype=np.complex128)
        if xk.ndim != 2:
            raise ValueError("channel_samples must have shape (num_channels, num_blocks)")
        if xk.shape[0] != self.num_channels:
            raise ValueError("channel_samples first dimension must equal num_channels")
        if xk.shape[1] == 0:
            return np.zeros(0, dtype=np.complex128)

        phases_in = np.fft.fft(xk, axis=0)
        return _synthesis_polyphase_kernel(phases_in, self._phases)
