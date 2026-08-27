"""Unit tests for the pixel-grid plots: images, heatmaps, and the plots
that sample a function over a rectangle."""

import numpy as np
import pytest

import matthewplotlib as mp

from matthewplotlib.plots import (
    cfunction2,
    function2,
    heatmap,
    histogram2,
    image,
    vfunction2,
)

from tests.plots_common import RecordingColormap, RecordingVectorColormap


# # #
# heatmaps


class TestImage:
    @pytest.mark.parametrize("channels", [1, 2, 4])
    def test_an_rgb_image_must_have_three_channels(self, channels):
        with pytest.raises(ValueError, match="RGB colors of shape"):
            mp.image(np.zeros((4, 5, channels), dtype=np.uint8))

    def test_a_colormap_may_reduce_feature_vectors_to_rgb(self):
        features = np.zeros((4, 5, 7))

        plot = mp.image(
            features,
            colormap=lambda values: np.zeros((*values.shape[:2], 3)),
        )

        assert (plot.height, plot.width) == (2, 5)


class TestHeatmap:
    def test_the_scale_spans_the_data_by_default(self):
        colormap = RecordingColormap()

        plot = heatmap([[0.0, 5.0], [10.0, 20.0]], colormap=colormap)

        assert plot.vrange == (0.0, 20.0)
        assert np.array_equal(colormap.input, [[0.0, 0.25], [0.5, 1.0]])

    def test_values_outside_a_given_interval_saturate(self):
        colormap = RecordingColormap()

        plot = heatmap(
            [[-1.0, 0.5], [1.0, 2.0]],
            colormap=colormap,
            vrange=(0.0, 1.0),
        )

        assert plot.vrange == (0.0, 1.0)
        assert np.array_equal(colormap.input, [[0.0, 0.5], [1.0, 1.0]])

    def test_a_descending_interval_turns_the_scale_around(self):
        colormap = RecordingColormap()

        heatmap([[0.0, 1.0]], colormap=colormap, vrange=(1.0, 0.0))

        assert np.array_equal(colormap.input, [[1.0, 0.0]])

    def test_a_constant_grid_comes_out_at_the_bottom(self):
        """There is no interval for the colours to span, so rather than
        dividing by nothing every value sits at the bottom of the colormap."""
        colormap = RecordingColormap()

        plot = heatmap([[7.0, 7.0]], colormap=colormap)

        assert plot.vrange == (7.0, 7.0)
        assert np.array_equal(colormap.input, [[0.0, 0.0]])

    def test_a_value_that_is_not_a_number_comes_out_at_the_bottom(self):
        colormap = RecordingColormap()

        plot = heatmap([[1.0, float("nan"), 3.0]], colormap=colormap)

        assert plot.vrange == (1.0, 3.0)
        assert np.array_equal(colormap.input, [[0.0, 0.0, 1.0]])

    def test_an_interval_covering_nothing_is_refused(self):
        with pytest.raises(ValueError, match="covers no interval"):
            heatmap([[0.0, 1.0]], vrange=(5.0, 5.0))

    def test_it_needs_a_grid_of_values(self):
        with pytest.raises(ValueError, match="2d grid"):
            heatmap(np.zeros((2, 3, 3)))

    def test_it_keeps_the_interval_it_was_given(self):
        plot = heatmap([[0.0, 1.0]], colormap=mp.viridis, vrange=(0.0, 4.0))

        assert plot.vrange == (0.0, 4.0)

    def test_an_image_keeps_no_interval(self):
        """Its data is already colours, or already scaled, so there is no
        interval to report and nothing for a colorbar to label."""
        assert not hasattr(image([[0.0, 1.0]]), "vrange")


class TestFunction2:
    def test_values_outside_vrange_saturate_before_colormapping(self):
        colormap = RecordingColormap()

        function2(
            lambda xy: xy[:, 0],
            xrange=(-1.0, 2.0),
            yrange=(0.0, 1.0),
            width=2,
            height=1,
            vrange=(0.0, 1.0),
            colormap=colormap,
            endpoints=True,
        )

        assert np.array_equal(colormap.input, [[0.0, 1.0], [0.0, 1.0]])

    def test_squares_are_sampled_at_their_centres(self):
        """The squares tile the ranges, and each shows the value of the
        function at the centre of the square it stands for."""
        sampled = []

        def record(xy):
            sampled.append(xy)
            return xy[:, 0]

        function2(record, xrange=(0.0, 1.0), yrange=(0.0, 2.0), width=4, height=1)

        assert np.allclose(np.unique(sampled[0][:, 0]), [0.125, 0.375, 0.625, 0.875])
        assert np.allclose(np.unique(sampled[0][:, 1]), [0.5, 1.5])

    def test_endpoints_reaches_the_ends_of_both_ranges(self):
        sampled = []

        def record(xy):
            sampled.append(xy)
            return xy[:, 0]

        function2(
            record,
            xrange=(0.0, 1.0),
            yrange=(0.0, 2.0),
            width=4,
            height=1,
            endpoints=True,
        )

        assert np.allclose(np.unique(sampled[0][:, 0]), [0.0, 1 / 3, 2 / 3, 1.0])
        assert np.allclose(np.unique(sampled[0][:, 1]), [0.0, 2.0])


class TestVFunction2:
    def test_the_field_is_scaled_into_the_unit_disc(self):
        colormap = RecordingVectorColormap()

        vfunction2(
            lambda xy: np.stack([xy[:, 0], np.zeros(len(xy))], axis=-1),
            xrange=(0.0, 4.0),
            yrange=(0.0, 1.0),
            width=2,
            height=1,
            colormap=colormap,
        )

        # x is sampled at 1 and 3, and the largest magnitude becomes one
        assert np.allclose(colormap.input[..., 0], [[1 / 3, 1.0], [1 / 3, 1.0]])

    def test_magnitudes_outside_vrange_saturate(self):
        colormap = RecordingVectorColormap()

        vfunction2(
            lambda xy: np.stack([xy[:, 0], np.zeros(len(xy))], axis=-1),
            xrange=(0.0, 4.0),
            yrange=(0.0, 1.0),
            width=2,
            height=1,
            vrange=(0.0, 2.0),
            colormap=colormap,
        )

        assert np.allclose(colormap.input[..., 0], [[0.5, 1.0], [0.5, 1.0]])

    def test_scaling_keeps_the_direction(self):
        colormap = RecordingVectorColormap()

        vfunction2(
            lambda xy: np.full((len(xy), 2), [3.0, 4.0]),
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            width=1,
            height=1,
            vrange=(0.0, 10.0),
            colormap=colormap,
        )

        # magnitude five out of ten, along the same 3:4 diagonal
        assert np.allclose(colormap.input, [[[0.3, 0.4]], [[0.3, 0.4]]])

    def test_a_field_of_zeroes_does_not_divide_by_zero(self):
        with np.errstate(all="raise"):
            plot = vfunction2(
                lambda xy: np.zeros_like(xy),
                xrange=(0.0, 1.0),
                yrange=(0.0, 1.0),
                width=2,
                height=1,
            )
        assert plot.width == 2

    def test_a_field_that_returns_the_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match="one .u, v. vector per point"):
            vfunction2(
                lambda xy: xy[:, 0],
                xrange=(0.0, 1.0),
                yrange=(0.0, 1.0),
                width=2,
                height=1,
            )


class TestCFunction2:
    def test_the_values_reach_the_colormap_unnormalised(self):
        """A domain colouring reads the modulus on an absolute scale."""
        seen = []

        def colormap(values):
            seen.append(np.array(values, copy=True))
            return np.zeros((*np.shape(values), 3), dtype=np.uint8)

        cfunction2(
            lambda z: 10 * z,
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            width=2,
            height=1,
            colormap=colormap,
        )

        assert np.allclose(seen[0].real, [[5.0, 15.0], [5.0, 15.0]])

    def test_the_function_is_given_complex_numbers(self):
        sampled = []

        def record(z):
            sampled.append(np.array(z, copy=True))
            return z

        cfunction2(record, (0.0, 2.0), (0.0, 2.0), width=2, height=1)

        assert np.iscomplexobj(sampled[0])
        assert np.allclose(np.unique(sampled[0].real), [0.5, 1.5])
        assert np.allclose(np.unique(sampled[0].imag), [0.5, 1.5])

    def test_a_function_that_returns_the_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match="one value per point"):
            cfunction2(
                lambda z: np.stack([z.real, z.imag], axis=-1),
                xrange=(0.0, 1.0),
                yrange=(0.0, 1.0),
                width=2,
                height=1,
            )

    def test_a_pole_at_the_origin_is_not_sampled_on(self):
        """Centre sampling never lands on a round number."""
        with np.errstate(all="raise"):
            cfunction2(lambda z: 1 / z, (-1.0, 1.0), (-1.0, 1.0), 4, 2)


class TestHistogram2:
    def test_counts_above_max_count_saturate_before_colormapping(self):
        colormap = RecordingColormap()

        histogram2(
            x=[0.0, 0.0, 0.0],
            y=[0.0, 0.0, 0.0],
            width=1,
            height=1,
            xrange=(-1.0, 1.0),
            yrange=(-1.0, 1.0),
            max_count=2,
            colormap=colormap,
        )

        assert colormap.input.max() == 1.0
        assert np.count_nonzero(colormap.input == 1.0) == 1

    def test_an_inferred_zero_maximum_produces_a_blank_heatmap(self):
        colormap = RecordingColormap()

        histogram2(
            x=[2.0],
            y=[2.0],
            width=1,
            height=1,
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            colormap=colormap,
        )

        assert np.array_equal(colormap.input, np.zeros((2, 1)))

    @pytest.mark.parametrize("max_count", [0, -1])
    def test_an_explicit_maximum_must_be_positive(self, max_count):
        with pytest.raises(ValueError, match="max_count must be positive"):
            histogram2(
                x=[0.0],
                y=[0.0],
                width=1,
                height=1,
                xrange=(-1.0, 1.0),
                yrange=(-1.0, 1.0),
                max_count=max_count,
            )
