"""
Benchmark: PolyphaseAnalysisChannelizer vs naive per-channel demodulation.

Installed as the ``pypoly-bench`` command. Useful for measuring throughput
on the current hardware after installation.

Usage:
    pypoly-bench           # full parameter sweep
    pypoly-bench --quick   # single small case (N=8192, M=8)
"""

import argparse
import time
import timeit

import numpy as np
from scipy.signal import lfilter

from pypoly import PolyphaseAnalysisChannelizer


def naive_analysis(x: np.ndarray, h: np.ndarray, M: int, D: int) -> np.ndarray:
    """Per-channel frequency shift + causal FIR filter + downsample.

    The straightforward reference implementation: demodulate each channel
    individually, filter with the full prototype, then decimate.
    Cost scales as O(N * M * len(h)).
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


def run(configs: list[tuple[int, int, int]], number: int = 5, repeat: int = 3) -> None:
    print("\nWarming up Numba JIT (first call compiles; not included in timings)...")
    _ch = PolyphaseAnalysisChannelizer.from_design(num_channels=8, taps_per_channel=8)
    _x = np.ones(1024, dtype=np.complex128)
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

        naive_times = timeit.repeat(
            lambda: naive_analysis(x, h, M, D),
            number=number,
            repeat=repeat,
        )
        naive_ms = min(naive_times) / number * 1000

        # Trigger compilation for this exact (N, M, K) shape before timing
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
    parser = argparse.ArgumentParser(
        description="Benchmark PolyphaseAnalysisChannelizer throughput on this system."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Single small case (N=8192, M=8) — fast sanity check",
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
    run(configs)


if __name__ == "__main__":
    main()
