from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
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


@njit(cache=True, parallel=True)
def _analysis_polyphase_kernel(
    samples_padded: np.ndarray, phases: np.ndarray, num_blocks: int, block_stride: int, pad: int
) -> np.ndarray:
    # samples_padded has (M*K - 1) leading zeros so every access is in-bounds
    # with no conditional; prange parallelises independent output blocks.
    m_channels, taps_per_phase = phases.shape
    out = np.zeros((m_channels, num_blocks), dtype=np.complex128)

    for n in prange(num_blocks):
        base = pad + n * block_stride
        for phase_idx in range(m_channels):
            acc = 0.0 + 0.0j
            for tap_idx in range(taps_per_phase):
                acc += phases[phase_idx, tap_idx] * samples_padded[base - tap_idx * m_channels - phase_idx]
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
        x = np.asarray(samples, dtype=np.complex128)
        if x.ndim != 1:
            raise ValueError(f"samples must be a 1D array, got shape {x.shape}")
        if x.size == 0:
            return np.zeros((self.num_channels, 0), dtype=np.complex128)

        num_blocks = int(np.ceil(x.size / self._decimation))
        pad = self._phases.shape[1] * self.num_channels - 1  # M*K - 1
        x_padded = np.concatenate([np.zeros(pad, dtype=np.complex128), x])
        phases_out = _analysis_polyphase_kernel(x_padded, self._phases, num_blocks, self._decimation, pad)
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


class StreamingPolyphaseAnalysisChannelizer:
    """Stateful polyphase analysis channelizer for continuous sample streams.

    Processes one chunk of input samples at a time, maintaining filter state
    across calls so that consecutive chunks produce the same channel outputs as
    a single batch call on the concatenated signal.  Each call to
    :meth:`process` returns exactly ``floor(available / decimation)`` output
    blocks, where *available* is the number of buffered samples plus the length
    of the new chunk.  Leftover samples are held in an internal buffer and
    included in the next call.

    Constructor parameters are identical to
    :class:`PolyphaseAnalysisChannelizer`.  Use :meth:`from_design` or
    :meth:`from_channelizer` for convenience construction.

    **Transport:** This class is transport-agnostic — feed it whatever numpy
    arrays your source provides.  Common patterns::

        # ZeroMQ (pyzmq) — standard in the GNU Radio / SDR ecosystem
        import zmq, numpy as np
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect("tcp://localhost:5555")
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        while True:
            data = sub.recv()
            chunk = np.frombuffer(data, dtype=np.complex64).astype(np.complex128)
            channels = streaming_ch.process(chunk)

        # UDP — lower-level, no framing guarantees
        import socket, numpy as np
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", 5000))
        while True:
            data, _ = sock.recvfrom(65536)
            chunk = np.frombuffer(data, dtype=np.complex64).astype(np.complex128)
            channels = streaming_ch.process(chunk)
    """

    def __init__(
        self,
        num_channels: int,
        prototype_taps: np.ndarray,
        *,
        decimation: int | None = None,
    ) -> None:
        # Validation mirrors PolyphaseAnalysisChannelizer.__post_init__
        taps = np.asarray(prototype_taps, dtype=np.float64)
        if taps.ndim != 1:
            raise ValueError("prototype_taps must be a 1D array")
        if taps.size % num_channels != 0:
            raise ValueError("prototype_taps length must be divisible by num_channels")
        effective_decimation = num_channels if decimation is None else decimation
        if not (1 <= effective_decimation <= num_channels):
            raise ValueError("decimation must be an integer in [1, num_channels]")

        self._num_channels = num_channels
        self._prototype_taps = taps
        self._phases = taps.reshape(-1, num_channels).T.copy()
        self._decimation = effective_decimation
        self._pad = self._phases.shape[1] * num_channels - 1  # M*K - 1

        self._history = np.zeros(self._pad, dtype=np.complex128)
        self._input_buffer = np.zeros(0, dtype=np.complex128)
        self._block_offset = 0

    @classmethod
    def from_design(
        cls,
        num_channels: int,
        taps_per_channel: int,
        *,
        cutoff_ratio: float | None = None,
        decimation: int | None = None,
    ) -> StreamingPolyphaseAnalysisChannelizer:
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

    @classmethod
    def from_channelizer(
        cls, channelizer: PolyphaseAnalysisChannelizer
    ) -> StreamingPolyphaseAnalysisChannelizer:
        """Create a streaming channelizer with the same filter as *channelizer*."""
        return cls(
            num_channels=channelizer.num_channels,
            prototype_taps=channelizer.prototype_taps,
            decimation=channelizer.decimation,
        )

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def prototype_taps(self) -> np.ndarray:
        return self._prototype_taps

    @property
    def decimation(self) -> int:
        return self._decimation

    @property
    def oversample_rate(self) -> float:
        """Informational only: num_channels / decimation."""
        return self._num_channels / self._decimation

    def reset(self) -> None:
        """Reset filter state to zero initial conditions.

        After reset the channelizer behaves as if freshly constructed: the
        filter memory is cleared and the block offset counter restarts from
        zero.  Use this to start a new stream without reallocating the filter.
        """
        self._history = np.zeros(self._pad, dtype=np.complex128)
        self._input_buffer = np.zeros(0, dtype=np.complex128)
        self._block_offset = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Process one chunk of input samples.

        Parameters
        ----------
        chunk:
            1-D array of complex input samples.  Any length is accepted,
            including lengths shorter than the decimation factor.

        Returns
        -------
        np.ndarray
            Channel outputs with shape ``(num_channels, num_blocks)`` where
            ``num_blocks = floor(available / decimation)`` and *available* is
            ``len(chunk)`` plus any samples buffered from previous calls.
            Returns shape ``(num_channels, 0)`` if fewer than ``decimation``
            samples are available.
        """
        x = np.asarray(chunk, dtype=np.complex128)
        if x.ndim != 1:
            raise ValueError(f"chunk must be a 1D array, got shape {x.shape}")

        available = np.concatenate([self._input_buffer, x])
        num_blocks = len(available) // self._decimation

        if num_blocks == 0:
            self._input_buffer = available
            return np.zeros((self._num_channels, 0), dtype=np.complex128)

        N_to_process = num_blocks * self._decimation
        samples = available[:N_to_process]
        self._input_buffer = available[N_to_process:]

        x_padded = np.concatenate([self._history, samples])
        phases_out = _analysis_polyphase_kernel(
            x_padded, self._phases, num_blocks, self._decimation, self._pad
        )
        channels = np.fft.ifft(phases_out, axis=0)

        if self._decimation != self._num_channels:
            # Same correction as the batch class, but m_idx is global so phase
            # coherence is maintained across consecutive process() calls.
            m_idx = np.arange(self._block_offset, self._block_offset + num_blocks)
            shift = (m_idx * self._decimation) % self._num_channels
            k_idx = np.arange(self._num_channels).reshape(-1, 1)
            correction = np.exp(
                -1j * 2 * np.pi * k_idx * shift.reshape(1, -1) / self._num_channels
            )
            channels *= correction

        self._history = np.concatenate([self._history, samples])[-self._pad:]
        self._block_offset += num_blocks
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
