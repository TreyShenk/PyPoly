import unittest
import warnings

import numpy as np

from pypoly import (
    PolyphaseAnalysisChannelizer,
    PolyphaseSynthesisChannelizer,
    StreamingPolyphaseAnalysisChannelizer,
    design_prototype_filter,
)


class ChannelizerTests(unittest.TestCase):
    def test_filter_size(self) -> None:
        taps = design_prototype_filter(num_channels=8, taps_per_channel=12)
        self.assertEqual(taps.shape, (96,))

    def test_analysis_output_shape(self) -> None:
        analysis = PolyphaseAnalysisChannelizer.from_design(
            num_channels=8, taps_per_channel=12
        )
        x = np.ones(257, dtype=np.complex128)
        y = analysis.process(x)
        self.assertEqual(y.shape, (8, 33))

    def test_process_rejects_non_1d_input(self) -> None:
        analysis = PolyphaseAnalysisChannelizer.from_design(
            num_channels=8, taps_per_channel=12
        )
        with self.assertRaises(ValueError):
            analysis.process(np.zeros((100, 2), dtype=np.complex128))
        with self.assertRaises(ValueError):
            analysis.process(np.complex128(1.0))

    def test_decimation_must_be_in_range(self) -> None:
        with self.assertRaises(ValueError):
            PolyphaseAnalysisChannelizer.from_design(
                num_channels=8, taps_per_channel=16, decimation=0
            )
        with self.assertRaises(ValueError):
            PolyphaseAnalysisChannelizer.from_design(
                num_channels=8, taps_per_channel=16, decimation=9
            )

    def test_decimation_doubles_block_count_at_half_num_channels(self) -> None:
        analysis = PolyphaseAnalysisChannelizer.from_design(
            num_channels=8, taps_per_channel=16, decimation=4
        )
        self.assertEqual(analysis.oversample_rate, 2.0)
        x = np.ones(257, dtype=np.complex128)
        y = analysis.process(x)
        self.assertEqual(y.shape, (8, 65))

    def test_low_decimation_warns_and_loses_channel_separation(self) -> None:
        # decimation=1 (maximal oversampling) auto-scales cutoff_ratio to 0.9 * num_channels,
        # i.e. an actual cutoff at 0.9 x Nyquist -- past the point (0.5 x Nyquist) where the
        # farthest channel sits, so it's structurally inside the passband. design_prototype_filter
        # should warn about this rather than silently handing back a non-functional filter.
        with self.assertWarns(UserWarning):
            analysis = PolyphaseAnalysisChannelizer.from_design(
                num_channels=8, taps_per_channel=16, decimation=1
            )
        n = np.arange(8192)
        x = np.ones_like(n, dtype=np.complex128)  # DC tone
        y = analysis.process(x)
        energy = np.mean(np.abs(y[:, 200:]) ** 2, axis=1)
        # DC should leak into essentially every channel with comparable energy --
        # i.e. no real channel separation -- confirming the warning's premise holds.
        non_nyquist = np.delete(energy, 4)
        self.assertLess(non_nyquist.max() / non_nyquist.min(), 1.1)

    def test_decimation_at_recommended_level_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PolyphaseAnalysisChannelizer.from_design(
                num_channels=8, taps_per_channel=16, decimation=4
            )

    def test_decimation_preserves_channel_to_frequency_mapping(self) -> None:
        num_channels = 8
        for decimation in (8, 4, 3, 2):
            analysis = PolyphaseAnalysisChannelizer.from_design(
                num_channels=num_channels, taps_per_channel=16, decimation=decimation
            )
            n = np.arange(8192)
            for k0 in range(num_channels):
                x = np.exp(1j * 2 * np.pi * k0 * n / num_channels)
                y = analysis.process(x)
                energy = np.mean(np.abs(y[:, 200:]) ** 2, axis=1)
                self.assertEqual(
                    int(np.argmax(energy)),
                    k0,
                    msg=f"decimation={decimation} k0={k0}",
                )

    def test_decimation_scales_default_cutoff_and_flattens_crossover(self) -> None:
        num_channels = 8
        critical = PolyphaseAnalysisChannelizer.from_design(
            num_channels=num_channels, taps_per_channel=16
        )
        oversampled = PolyphaseAnalysisChannelizer.from_design(
            num_channels=num_channels, taps_per_channel=16, decimation=4
        )

        n = np.arange(8192)
        f0 = 3.5 / num_channels  # exact crossover between channel 3 and channel 4
        x = np.exp(1j * 2 * np.pi * f0 * n)

        crit_energy = np.mean(np.abs(critical.process(x)[3, 200:]) ** 2)
        over_energy = np.mean(np.abs(oversampled.process(x)[3, 200:]) ** 2)
        self.assertGreater(over_energy, crit_energy * 5)

    def test_analysis_synthesis_pipeline_has_finite_output(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=1024) + 1j * rng.normal(size=1024)

        analysis = PolyphaseAnalysisChannelizer.from_design(
            num_channels=8, taps_per_channel=16
        )
        synthesis = PolyphaseSynthesisChannelizer.from_design(
            num_channels=8, taps_per_channel=16
        )

        y = analysis.process(x)
        x_hat = synthesis.process(y)

        self.assertEqual(x_hat.shape, (1024,))
        self.assertTrue(np.all(np.isfinite(x_hat)))


class StreamingChannelizerTests(unittest.TestCase):
    def _make_signal(self, N: int = 8192, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(N) + 1j * rng.standard_normal(N)

    def _batch_reference(
        self, x: np.ndarray, num_channels: int = 8, taps_per_channel: int = 16,
        decimation: int | None = None,
    ) -> np.ndarray:
        analysis = PolyphaseAnalysisChannelizer.from_design(
            num_channels=num_channels, taps_per_channel=taps_per_channel, decimation=decimation
        )
        return analysis.process(x)

    def test_chunked_equals_batch_exact_blocks(self) -> None:
        # Chunk size = exact multiple of decimation — no buffering involved
        M, D, K = 8, 8, 16
        x = self._make_signal()
        batch = self._batch_reference(x, M, K, D)

        streaming = StreamingPolyphaseAnalysisChannelizer.from_design(
            num_channels=M, taps_per_channel=K, decimation=D
        )
        chunk_size = D * 16  # 128 samples per chunk
        outputs = [streaming.process(x[i:i + chunk_size]) for i in range(0, len(x), chunk_size)]
        result = np.concatenate(outputs, axis=1)

        # Streaming uses floor(N/D) blocks; batch uses ceil(N/D). Trim to match.
        np.testing.assert_allclose(result, batch[:, :result.shape[1]], atol=1e-10)

    def test_chunked_equals_batch_partial_blocks(self) -> None:
        # Chunk size is not a multiple of decimation — exercises the input buffer
        M, D, K = 8, 8, 16
        x = self._make_signal()
        batch = self._batch_reference(x, M, K, D)

        streaming = StreamingPolyphaseAnalysisChannelizer.from_design(
            num_channels=M, taps_per_channel=K, decimation=D
        )
        chunk_size = 37  # prime, never a multiple of D=8
        outputs = [streaming.process(x[i:i + chunk_size]) for i in range(0, len(x), chunk_size)]
        result = np.concatenate(outputs, axis=1)

        np.testing.assert_allclose(result, batch[:, :result.shape[1]], atol=1e-10)

    def test_chunk_smaller_than_decimation_returns_empty(self) -> None:
        M, D, K = 8, 8, 16
        streaming = StreamingPolyphaseAnalysisChannelizer.from_design(
            num_channels=M, taps_per_channel=K, decimation=D
        )
        # Feed D-1 samples — not enough for one output block
        out = streaming.process(np.ones(D - 1, dtype=np.complex128))
        self.assertEqual(out.shape, (M, 0))

        # Feed one more sample — now we have exactly D samples buffered → one block
        out = streaming.process(np.ones(1, dtype=np.complex128))
        self.assertEqual(out.shape, (M, 1))

    def test_reset_restores_initial_conditions(self) -> None:
        M, K = 8, 16
        x = self._make_signal(N=512)

        streaming = StreamingPolyphaseAnalysisChannelizer.from_design(
            num_channels=M, taps_per_channel=K
        )
        first_run = streaming.process(x)
        streaming.reset()
        second_run = streaming.process(x)

        np.testing.assert_array_equal(first_run, second_run)

    def test_from_channelizer_matches_batch(self) -> None:
        M, K = 8, 16
        x = self._make_signal()
        batch = PolyphaseAnalysisChannelizer.from_design(num_channels=M, taps_per_channel=K)
        streaming = StreamingPolyphaseAnalysisChannelizer.from_channelizer(batch)

        batch_out = batch.process(x)
        stream_out = streaming.process(x)

        np.testing.assert_allclose(stream_out, batch_out[:, :stream_out.shape[1]], atol=1e-10)

    def test_oversampled_chunked_equals_batch(self) -> None:
        # Exercises the _block_offset phase correction accumulating across calls
        M, D, K = 8, 4, 16
        x = self._make_signal()
        batch = self._batch_reference(x, M, K, D)

        streaming = StreamingPolyphaseAnalysisChannelizer.from_design(
            num_channels=M, taps_per_channel=K, decimation=D
        )
        chunk_size = 53  # not a multiple of D=4
        outputs = [streaming.process(x[i:i + chunk_size]) for i in range(0, len(x), chunk_size)]
        result = np.concatenate(outputs, axis=1)

        np.testing.assert_allclose(result, batch[:, :result.shape[1]], atol=1e-10)

    def test_process_rejects_non_1d_input(self) -> None:
        streaming = StreamingPolyphaseAnalysisChannelizer.from_design(
            num_channels=8, taps_per_channel=16
        )
        with self.assertRaises(ValueError):
            streaming.process(np.zeros((100, 2), dtype=np.complex128))


if __name__ == "__main__":
    unittest.main()
