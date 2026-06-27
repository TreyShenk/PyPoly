"""
Benchmark: PolyphaseAnalysisChannelizer vs naive per-channel demodulation.

Usage:
    uv run python benchmarks/bench_analysis.py          # full sweep
    uv run python benchmarks/bench_analysis.py --quick  # single small case
"""

import argparse
import sys
import time
import timeit
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pypoly import PolyphaseAnalysisChannelizer, design_prototype_filter


def naive_analysis(x: np.ndarray, h: np.ndarray, M: int, D: int) -> np.ndarray:
    """Per-channel frequency shift + causal FIR filter + downsample.

    This is the straightforward approach a DSP practitioner would reach for
    first: demodulate each channel individually, filter with the full prototype,
    then decimate. Cost scales as O(N * M * len(h)).
    """
    N = len(x)
    n = np.arange(N, dtype=np.float64)
    num_out = int(np.ceil(N / D))
    out = np.zeros((M, num_out), dtype=np.complex128)
    for k in range(M):
        shifted = x * np.exp(-1j * 2 * np.pi * k * n / M)
        out[k] = lfilter(h, 1.0, shifted)[::D][:num_out]
    return out


def _fmt_row(label: str, N: int, M: int, K: int, ms: float, naive_ms: float) -> str:
    throughput = N / (ms / 1000) / 1e6
    speedup = naive_ms / ms
    return (
        f"  {label:<22}  N={N:>7}  M={M:>2}  K={K:>2}"
        f"  {ms:>8.2f} ms  {throughput:>7.2f} Msps  {speedup:>5.1f}x vs naive"
    )


def benchmark(configs: list[tuple[int, int, int]], number: int = 5, repeat: int = 3) -> None:
    print("\nWarming up Numba JIT (first call compiles; not included in timings)...")
    _warmup_N, _warmup_M, _warmup_K = 1024, 8, 8
    _ch = PolyphaseAnalysisChannelizer.from_design(
        num_channels=_warmup_M, taps_per_channel=_warmup_K
    )
    _x = np.ones(_warmup_N, dtype=np.complex128)
    t0 = time.perf_counter()
    _ch.process(_x)
    print(f"  JIT compile: {(time.perf_counter() - t0) * 1000:.0f} ms\n")

    print(
        f"  {'Method':<22}  {'Signal':>13}       "
        f"{'Time':>11}  {'Throughput':>10}  {'Speedup':>13}"
    )
    print("  " + "-" * 80)

    for N, M, K in configs:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)

        analysis = PolyphaseAnalysisChannelizer.from_design(num_channels=M, taps_per_channel=K)
        h = analysis.prototype_taps
        D = M  # critically sampled

        # Naive
        naive_times = timeit.repeat(
            lambda: naive_analysis(x, h, M, D),
            number=number,
            repeat=repeat,
        )
        naive_ms = min(naive_times) / number * 1000

        # Polyphase (Numba already warm from the first call above; trigger
        # this exact shape once to ensure the parallel kernel is compiled too)
        analysis.process(x)
        poly_times = timeit.repeat(
            lambda: analysis.process(x),
            number=number,
            repeat=repeat,
        )
        poly_ms = min(poly_times) / number * 1000

        print(_fmt_row("naive (lfilter)", N, M, K, naive_ms, naive_ms))
        print(_fmt_row("polyphase (pypoly)", N, M, K, poly_ms, naive_ms))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="pypoly analysis channelizer benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Single small case (N=8192, M=8) — for CI or quick iteration",
    )
    args = parser.parse_args()

    if args.quick:
        configs = [(8_192, 8, 16)]
    else:
        configs = [
            (8_192,   8,  16),
            (65_536,  8,  16),
            (524_288, 8,  16),
            (65_536,  16, 16),
            (65_536,  32, 16),
        ]

    print("pypoly benchmark — PolyphaseAnalysisChannelizer vs naive demodulation")
    print(f"  number={5}, repeat={3}, taking min across repeats")
    benchmark(configs)


if __name__ == "__main__":
    main()
