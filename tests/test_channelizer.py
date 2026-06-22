import unittest
import warnings

import numpy as np

from pypoly import (
    PolyphaseAnalysisChannelizer,
    PolyphaseSynthesisChannelizer,
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


if __name__ == "__main__":
    unittest.main()
