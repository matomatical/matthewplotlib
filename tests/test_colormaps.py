import numpy as np
import pytest

from matthewplotlib.colormaps import (
    reds, greens, blues, yellows, magentas, cyans,
    divreds, divgreens, divblues,
    cyber, rainbow,
    magma, inferno, plasma, viridis,
    sweetie16, pico8, tableau, nouveau,
)


ALL_CONTINUOUS = [
    reds, greens, blues, yellows, magentas, cyans,
    divreds, divgreens, divblues,
    cyber, rainbow,
    magma, inferno, plasma, viridis,
]

ALL_DISCRETE = [
    sweetie16, pico8, tableau, nouveau,
]


# # #
# Continuous colormaps — generic properties


class TestContinuousColormaps:
    @pytest.mark.parametrize("cmap", ALL_CONTINUOUS)
    def test_scalar_output_shape(self, cmap):
        result = cmap(0.5)
        assert result.shape == (3,)

    @pytest.mark.parametrize("cmap", ALL_CONTINUOUS)
    def test_1d_output_shape(self, cmap):
        result = cmap(np.array([0.0, 0.5, 1.0]))
        assert result.shape == (3, 3)

    @pytest.mark.parametrize("cmap", ALL_CONTINUOUS)
    def test_2d_broadcasting(self, cmap):
        x = np.array([[0.0, 0.5], [0.25, 0.75]])
        result = cmap(x)
        assert result.shape == (2, 2, 3)

    @pytest.mark.parametrize("cmap", ALL_CONTINUOUS)
    def test_3d_broadcasting(self, cmap):
        x = np.zeros((2, 3, 4))
        result = cmap(x)
        assert result.shape == (2, 3, 4, 3)

    @pytest.mark.parametrize("cmap", ALL_CONTINUOUS)
    def test_output_dtype(self, cmap):
        """All continuous colormaps should return uint8. Currently magma,
        inferno, plasma, and viridis return int64 from their lookup tables
        — this is a bug to fix in colormaps.py."""
        result = cmap(np.array([0.0, 0.5, 1.0]))
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("cmap", ALL_CONTINUOUS)
    def test_values_in_valid_range(self, cmap):
        x = np.linspace(0, 1, 50)
        result = cmap(x)
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    @pytest.mark.parametrize("cmap", ALL_CONTINUOUS)
    def test_boundary_values_valid(self, cmap):
        r0 = cmap(0.0)
        r1 = cmap(1.0)
        assert r0.shape == (3,)
        assert r1.shape == (3,)


# # #
# Basic single-channel colormaps — value tests


class TestBasicColormapValues:
    """Test endpoint and midpoint values for the linear single-channel maps."""

    # --- reds: R channel only ---
    def test_reds_at_zero(self):
        assert np.array_equal(reds(0.0), [0, 0, 0])

    def test_reds_at_half(self):
        result = reds(0.5)
        assert result[0] == 127
        assert result[1] == 0
        assert result[2] == 0

    def test_reds_at_one(self):
        assert np.array_equal(reds(1.0), [255, 0, 0])

    # --- greens: G channel only ---
    def test_greens_at_zero(self):
        assert np.array_equal(greens(0.0), [0, 0, 0])

    def test_greens_at_half(self):
        result = greens(0.5)
        assert result[0] == 0
        assert result[1] == 127
        assert result[2] == 0

    def test_greens_at_one(self):
        assert np.array_equal(greens(1.0), [0, 255, 0])

    # --- blues: B channel only ---
    def test_blues_at_zero(self):
        assert np.array_equal(blues(0.0), [0, 0, 0])

    def test_blues_at_half(self):
        result = blues(0.5)
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 127

    def test_blues_at_one(self):
        assert np.array_equal(blues(1.0), [0, 0, 255])

    # --- yellows: R+G channels ---
    def test_yellows_at_zero(self):
        assert np.array_equal(yellows(0.0), [0, 0, 0])

    def test_yellows_at_half(self):
        result = yellows(0.5)
        assert result[0] == 127
        assert result[1] == 127
        assert result[2] == 0

    def test_yellows_at_one(self):
        assert np.array_equal(yellows(1.0), [255, 255, 0])

    # --- magentas: R+B channels ---
    def test_magentas_at_zero(self):
        assert np.array_equal(magentas(0.0), [0, 0, 0])

    def test_magentas_at_half(self):
        result = magentas(0.5)
        assert result[0] == 127
        assert result[1] == 0
        assert result[2] == 127

    def test_magentas_at_one(self):
        assert np.array_equal(magentas(1.0), [255, 0, 255])

    # --- cyans: G+B channels ---
    def test_cyans_at_zero(self):
        assert np.array_equal(cyans(0.0), [0, 0, 0])

    def test_cyans_at_half(self):
        result = cyans(0.5)
        assert result[0] == 0
        assert result[1] == 127
        assert result[2] == 127

    def test_cyans_at_one(self):
        assert np.array_equal(cyans(1.0), [0, 255, 255])


# # #
# Diverging colormaps — changepoint tests
#
# These all have a kink at x=0.5 where channels hit 255 and are clamped.
#   divreds:  cyan (0.0) → white (0.5) → red (1.0)
#   divgreens: magenta (0.0) → white (0.5) → green (1.0)
#   divblues: yellow (0.0) → white (0.5) → blue (1.0)


class TestDivredsChangepoints:
    def test_at_0(self):
        """x=0: cyan (0, 255, 255)"""
        assert np.array_equal(divreds(0.0), [0, 255, 255])

    def test_at_025(self):
        """x=0.25: R ramps to 127, G/B still clamped at 255."""
        r = divreds(0.25)
        assert r[0] == 127   # min(255*0.5, 255) = 127
        assert r[1] == 255   # min(255*1.5, 255) = 255
        assert r[2] == 255

    def test_at_05(self):
        """x=0.5: white (255, 255, 255) — the changepoint."""
        assert np.array_equal(divreds(0.5), [255, 255, 255])

    def test_at_075(self):
        """x=0.75: R clamped at 255, G/B ramp down to 127."""
        r = divreds(0.75)
        assert r[0] == 255   # min(255*1.5, 255) = 255
        assert r[1] == 127   # min(255*0.5, 255) = 127
        assert r[2] == 127

    def test_at_1(self):
        """x=1: red (255, 0, 0)"""
        assert np.array_equal(divreds(1.0), [255, 0, 0])


class TestDivgreensChangepoints:
    def test_at_0(self):
        """x=0: magenta (255, 0, 255)"""
        assert np.array_equal(divgreens(0.0), [255, 0, 255])

    def test_at_025(self):
        r = divgreens(0.25)
        assert r[0] == 255
        assert r[1] == 127
        assert r[2] == 255

    def test_at_05(self):
        """x=0.5: white (255, 255, 255) — the changepoint."""
        assert np.array_equal(divgreens(0.5), [255, 255, 255])

    def test_at_075(self):
        r = divgreens(0.75)
        assert r[0] == 127
        assert r[1] == 255
        assert r[2] == 127

    def test_at_1(self):
        """x=1: green (0, 255, 0)"""
        assert np.array_equal(divgreens(1.0), [0, 255, 0])


class TestDivbluesChangepoints:
    def test_at_0(self):
        """x=0: yellow (255, 255, 0)"""
        assert np.array_equal(divblues(0.0), [255, 255, 0])

    def test_at_025(self):
        r = divblues(0.25)
        assert r[0] == 255
        assert r[1] == 255
        assert r[2] == 127

    def test_at_05(self):
        """x=0.5: white (255, 255, 255) — the changepoint."""
        assert np.array_equal(divblues(0.5), [255, 255, 255])

    def test_at_075(self):
        r = divblues(0.75)
        assert r[0] == 127
        assert r[1] == 127
        assert r[2] == 255

    def test_at_1(self):
        """x=1: blue (0, 0, 255)"""
        assert np.array_equal(divblues(1.0), [0, 0, 255])


# # #
# Cyber colormap — value tests
#
# cyber: magenta (0) → cyan (1), B channel always 255


class TestCyberValues:
    def test_at_0(self):
        """x=0: magenta (255, 0, 255)"""
        assert np.array_equal(cyber(0.0), [255, 0, 255])

    def test_at_05(self):
        """x=0.5: (127, 127, 255) — midpoint"""
        r = cyber(0.5)
        assert r[0] == 127
        assert r[1] == 127
        assert r[2] == 255

    def test_at_1(self):
        """x=1: cyan (0, 255, 255)"""
        assert np.array_equal(cyber(1.0), [0, 255, 255])

    def test_blue_always_255(self):
        """B channel should be 255 for all inputs."""
        x = np.linspace(0, 1, 20)
        result = cyber(x)
        assert np.all(result[:, 2] == 255)


# # #
# Rainbow colormap — segment boundary tests
#
# The rainbow has 6 segments, each 1/6 of [0, 1]:
#   x=0/6: red       (255, 0,   0)
#   x=1/6: yellow    (255, 255, 0)
#   x=2/6: green     (0,   255, 0)
#   x=3/6: cyan      (0,   255, 255)
#   x=4/6: blue      (0,   0,   255)
#   x=5/6: magenta   (255, 0,   255)


class TestRainbowChangepoints:
    def test_at_0_red(self):
        assert np.array_equal(rainbow(0.0), [255, 0, 0])

    def test_at_1_6_yellow(self):
        assert np.array_equal(rainbow(1/6), [255, 255, 0])

    def test_at_2_6_green(self):
        assert np.array_equal(rainbow(2/6), [0, 255, 0])

    def test_at_3_6_cyan(self):
        assert np.array_equal(rainbow(3/6), [0, 255, 255])

    def test_at_4_6_blue(self):
        assert np.array_equal(rainbow(4/6), [0, 0, 255])

    def test_at_5_6_magenta(self):
        assert np.array_equal(rainbow(5/6), [255, 0, 255])

    def test_mid_segment_0(self):
        """Midpoint of segment 0 (red→yellow): R=255, G≈127, B=0."""
        r = rainbow(1/12)
        assert r[0] == 255
        assert 120 <= r[1] <= 135  # approximately half
        assert r[2] == 0

    def test_mid_segment_3(self):
        """Midpoint of segment 3 (cyan→blue): R=0, G≈127, B=255."""
        r = rainbow(3/6 + 1/12)
        assert r[0] == 0
        assert 120 <= r[1] <= 135
        assert r[2] == 255

    def test_full_cycle_has_six_distinct_primary_colors(self):
        """At the 6 segment boundaries, all primary/secondary colors appear."""
        boundary_colors = set()
        for k in range(6):
            c = tuple(rainbow(k / 6).tolist())
            boundary_colors.add(c)
        assert len(boundary_colors) == 6


# # #
# Discrete colormaps


class TestDiscreteColormaps:
    @pytest.mark.parametrize("cmap", ALL_DISCRETE)
    def test_scalar_output_shape(self, cmap):
        result = cmap(0)
        assert result.shape == (3,)

    @pytest.mark.parametrize("cmap", ALL_DISCRETE)
    def test_1d_output_shape(self, cmap):
        result = cmap(np.array([0, 1, 2]))
        assert result.shape == (3, 3)

    @pytest.mark.parametrize("cmap", ALL_DISCRETE)
    def test_2d_broadcasting(self, cmap):
        x = np.array([[0, 1], [2, 3]])
        result = cmap(x)
        assert result.shape == (2, 2, 3)

    @pytest.mark.parametrize("cmap", ALL_DISCRETE)
    def test_output_dtype(self, cmap):
        result = cmap(np.array([0, 1, 2]))
        assert result.dtype == np.int64 or np.issubdtype(result.dtype, np.integer)

    @pytest.mark.parametrize("cmap", ALL_DISCRETE)
    def test_consistent_colors(self, cmap):
        """Same index should always produce the same color."""
        r1 = cmap(np.array([3]))
        r2 = cmap(np.array([3]))
        assert np.array_equal(r1, r2)

    @pytest.mark.parametrize("cmap", ALL_DISCRETE)
    def test_wrapping(self, cmap):
        """Out-of-range indices should wrap via modulo."""
        palette_size = {sweetie16: 16, pico8: 16, tableau: 10, nouveau: 10}
        n = palette_size[cmap]
        r_base = cmap(np.array([0]))
        r_wrap = cmap(np.array([n]))
        assert np.array_equal(r_base, r_wrap)
