"""Unit tests for the scales: the interval as a value, and the nonlinear
spacings within it."""

import numpy as np
import pytest

from matthewplotlib.scales import (
    scale,
    logscale,
    symlogscale,
    powscale,
    _resolve_vrange,
    _resolve_coordinate_range,
)


# # #
# construction


class TestConstruction:
    def test_endpoints_are_kept_as_floats(self):
        s = scale(0, 5)
        assert s.lo == 0.0 and isinstance(s.lo, float)
        assert s.hi == 5.0 and isinstance(s.hi, float)

    def test_endpoints_may_each_be_left_out(self):
        assert scale().lo is None
        assert scale(0).hi is None
        assert scale(hi=5).lo is None

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_a_non_finite_endpoint_is_refused(self, bad):
        with pytest.raises(ValueError, match="finite"):
            scale(0, bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_logscale_is_refused_an_endpoint_at_or_below_zero(self, bad):
        with pytest.raises(ValueError, match="above zero"):
            logscale(bad, 10)
        # a bad endpoint is refused even on a partial scale
        with pytest.raises(ValueError, match="above zero"):
            logscale(bad)

    def test_powscale_is_refused_a_negative_endpoint(self):
        with pytest.raises(ValueError, match="from zero up"):
            powscale(-1, 10, exponent=0.5)

    def test_powscale_accepts_an_endpoint_at_zero(self):
        assert powscale(0, 10, exponent=0.5).lo == 0.0

    @pytest.mark.parametrize("bad", [0, -0.5, float("inf")])
    def test_powscale_needs_a_positive_finite_exponent(self, bad):
        with pytest.raises(ValueError, match="positive exponent"):
            powscale(0, 1, exponent=bad)

    @pytest.mark.parametrize("bad", [0, -0.5, float("inf")])
    def test_symlogscale_needs_a_positive_finite_linear_width(self, bad):
        with pytest.raises(ValueError, match="positive linear_width"):
            symlogscale(-1, 1, linear_width=bad)

    def test_symlogscale_spans_zero(self):
        assert symlogscale(-10, 10).interval == (-10.0, 10.0)


# # #
# the value as a value: equality, unpacking, repr


class TestValueSemantics:
    def test_scales_compare_by_class_and_interval(self):
        assert scale(0, 5) == scale(0.0, 5.0)
        assert logscale(1, 255) == logscale(1, 255)
        assert logscale(1, 255) != scale(1, 255)
        assert scale(0, 5) != scale(0, 6)

    def test_extra_fields_take_part_in_equality(self):
        assert powscale(0, 1, exponent=0.5) != powscale(0, 1, exponent=2.0)
        assert symlogscale(linear_width=1) != symlogscale(linear_width=2)

    def test_a_scale_unpacks_as_its_interval(self):
        lo, hi = scale(0, 5)
        assert (lo, hi) == (0.0, 5.0)
        assert tuple(logscale(1, 100)) == (1.0, 100.0)

    def test_the_interval_is_reachable_as_a_pair(self):
        assert logscale(1, 100).interval == (1.0, 100.0)

    def test_a_partial_scale_has_no_interval(self):
        with pytest.raises(ValueError, match="missing endpoint"):
            scale(0).interval

    def test_reprs_name_the_spacing_and_the_interval(self):
        assert repr(scale(0, 5)) == "scale(0, 5)"
        assert repr(logscale(1, 255)) == "logscale(1, 255)"
        assert repr(scale(None, 5)) == "scale(None, 5)"
        assert repr(powscale(0, 1, exponent=0.5)) \
            == "powscale(0, 1, exponent=0.5)"
        assert repr(symlogscale(-3, 3, linear_width=0.1)) \
            == "symlogscale(-3, 3, linear_width=0.1)"


# # #
# measuring values


class TestCall:
    def test_linear_values_are_measured_linearly(self):
        assert np.array_equal(
            scale(0, 4)([0.0, 1.0, 2.0, 4.0]),
            [0.0, 0.25, 0.5, 1.0],
        )

    def test_values_beyond_the_interval_saturate(self):
        assert np.array_equal(
            scale(0, 1)([-1.0, 0.5, 2.0, float("inf")]),
            [0.0, 0.5, 1.0, 1.0],
        )

    def test_a_descending_interval_turns_the_scale_around(self):
        assert np.array_equal(scale(1, 0)([0.0, 0.25, 1.0]), [1.0, 0.75, 0.0])

    def test_a_value_that_is_not_a_number_comes_out_at_the_bottom(self):
        assert np.array_equal(scale(0, 1)([float("nan"), 1.0]), [0.0, 1.0])

    def test_a_flat_interval_puts_everything_at_the_bottom(self):
        assert np.array_equal(scale(5, 5)([4.0, 5.0, 6.0]), [0.0, 0.0, 0.0])

    def test_a_partial_scale_cannot_measure(self):
        with pytest.raises(ValueError, match="missing endpoint"):
            scale(0)([1.0])

    def test_logscale_measures_equal_factors_as_equal_steps(self):
        assert np.allclose(
            logscale(1, 100)([1.0, 10.0, 100.0]),
            [0.0, 0.5, 1.0],
        )

    def test_logscale_saturates_before_it_takes_the_log(self):
        # a value at or below zero clips to lo before np.log can object
        assert np.array_equal(logscale(1, 100)([-5.0, 0.0]), [0.0, 0.0])

    def test_a_descending_logscale_turns_around(self):
        assert np.allclose(logscale(100, 1)([1.0, 10.0, 100.0]), [1., .5, 0.])

    def test_symlogscale_is_symmetric_about_zero(self):
        s = symlogscale(-10, 10)
        assert s([0.0]) == [0.5]
        assert np.allclose(s([-7.0]) + s([7.0]), [1.0])

    def test_powscale_of_a_half_is_a_square_root(self):
        assert np.allclose(
            powscale(0, 4, exponent=0.5)([0.0, 1.0, 4.0]),
            [0.0, 0.5, 1.0],
        )

    def test_a_transform_flattening_the_interval_is_an_error(self):
        class brokenscale(scale):
            def transform(self, values):
                return np.zeros_like(np.asarray(values, dtype=float))

        with pytest.raises(ValueError, match="strictly monotonic"):
            brokenscale(0, 1)([0.5])


class TestPosition:
    def test_position_agrees_with_calling_inside_the_interval(self):
        assert np.array_equal(
            scale(0, 4).position([0.0, 1.0, 2.0, 4.0]),
            [0.0, 0.25, 0.5, 1.0],
        )

    def test_a_value_beyond_the_interval_keeps_going(self):
        assert np.array_equal(
            scale(0, 1).position([-1.0, 0.5, 2.0]),
            [-1.0, 0.5, 2.0],
        )

    def test_a_descending_interval_turns_the_positions_around(self):
        assert np.array_equal(
            scale(1, 0).position([0.0, 2.0]),
            [1.0, -1.0],
        )

    def test_a_value_that_is_not_a_number_is_unplaceable(self):
        assert np.isnan(scale(0, 1).position([float("nan")])).all()

    def test_a_value_outside_the_spacings_domain_is_unplaceable(self):
        positions = logscale(1, 100).position([-5.0, 10.0])
        assert np.isnan(positions[0])
        assert positions[1] == 0.5

    def test_a_partial_scale_has_no_positions(self):
        with pytest.raises(ValueError, match="missing endpoint"):
            scale(0).position([1.0])


# # #
# completion against data


class TestResolveVrange:
    VALUES = np.array([3.0, 9.0, float("nan")])

    def test_none_becomes_a_linear_scale_spanning_the_data(self):
        assert _resolve_vrange(None, self.VALUES, "test") == scale(3.0, 9.0)

    def test_a_pair_becomes_a_linear_scale(self):
        resolved = _resolve_vrange((0, 20), self.VALUES, "test")
        assert resolved == scale(0.0, 20.0)

    def test_a_complete_scale_is_returned_as_it_stands(self):
        given = logscale(1, 255)
        assert _resolve_vrange(given, self.VALUES, "test") is given

    def test_a_partial_scale_keeps_its_spacing_when_completed(self):
        resolved = _resolve_vrange(logscale(), self.VALUES, "test")
        assert resolved == logscale(3.0, 9.0)

    def test_completion_preserves_extra_fields(self):
        resolved = _resolve_vrange(
            symlogscale(linear_width=0.5), self.VALUES, "test",
        )
        assert resolved == symlogscale(3.0, 9.0, linear_width=0.5)

    def test_a_given_endpoint_survives_completion(self):
        resolved = _resolve_vrange(scale(0), self.VALUES, "test")
        assert resolved == scale(0.0, 9.0)

    def test_from_zero_starts_an_inferred_interval_at_zero(self):
        resolved = _resolve_vrange(None, self.VALUES, "test", from_zero=True)
        assert resolved == scale(0.0, 9.0)

    def test_an_explicit_interval_covering_nothing_is_an_error(self):
        with pytest.raises(ValueError, match="covers no interval"):
            _resolve_vrange((5, 5), self.VALUES, "test")
        with pytest.raises(ValueError, match="covers no interval"):
            _resolve_vrange(scale(5, 5), self.VALUES, "test")

    def test_an_inferred_flat_interval_is_returned_as_it_stands(self):
        flat = np.array([5.0, 5.0])
        assert _resolve_vrange(None, flat, "test") == scale(5.0, 5.0)

    def test_no_finite_values_fall_back_to_an_interval_of_the_scales_own(self):
        empty = np.array([float("nan")])
        assert _resolve_vrange(None, empty, "test") == scale(0.0, 1.0)
        assert _resolve_vrange(logscale(), empty, "test") == logscale(1., 10.)

    def test_inference_ignores_values_outside_the_spacings_domain(self):
        values = np.array([-2.0, 0.0, 3.0, 9.0])
        assert _resolve_vrange(logscale(), values, "test") == logscale(3., 9.)
        # with none of the values in the domain, the fallback stands in
        assert _resolve_vrange(logscale(), np.array([-2.0, 0.0]), "test") \
            == logscale(1.0, 10.0)

    def test_an_inferred_endpoint_outside_the_domain_says_to_give_one(self):
        # from_zero is a policy rather than an inference, so a logscale's
        # domain cannot save it: the baseline itself is out of reach
        with pytest.raises(ValueError, match="give one explicitly"):
            _resolve_vrange(logscale(), self.VALUES, "test", from_zero=True)


class TestResolveCoordinateRange:
    def test_completion_works_as_for_a_value_range(self):
        values = np.array([3.0, 9.0])
        resolve = _resolve_coordinate_range
        assert resolve((0, 20), values, "t", "xrange") == scale(0.0, 20.0)
        assert resolve(None, values, "t", "xrange") == scale(3.0, 9.0)
        assert resolve(logscale(), values, "t", "xrange") == logscale(3., 9.)

    def test_an_explicit_interval_covering_nothing_is_an_error(self):
        with pytest.raises(ValueError, match="xrange covers no interval"):
            _resolve_coordinate_range(
                (5, 5), np.array([3.0, 9.0]), "test", "xrange",
            )

    def test_a_constant_series_is_given_room(self):
        """Values that reach no distance still need an interval to sit in, so
        a flat inferred interval widens by half a unit each way."""
        constant = np.full(10, 4.0)
        assert _resolve_coordinate_range(None, constant, "t", "xrange") \
            == scale(3.5, 4.5)

    def test_a_single_element_is_given_room(self):
        assert _resolve_coordinate_range(
            None, np.array([3.0]), "t", "xrange",
        ) == scale(2.5, 3.5)

    def test_a_flat_nonlinear_interval_widens_along_its_own_spacing(self):
        """Half a unit each way in transformed space, so a logscale widens by
        a factor of sqrt(e) each way and stays inside its domain."""
        resolved = _resolve_coordinate_range(
            logscale(), np.array([4.0, 4.0]), "t", "xrange",
        )
        assert type(resolved) is logscale
        assert np.allclose(
            resolved.interval,
            (4.0 / np.sqrt(np.e), 4.0 * np.sqrt(np.e)),
        )

    def test_widening_leaves_a_given_endpoint_where_it_was_written(self):
        resolved = _resolve_coordinate_range(
            (4.0, None), np.array([4.0, 4.0]), "t", "xrange",
        )
        assert resolved == scale(4.0, 4.5)
        resolved = _resolve_coordinate_range(
            (None, 4.0), np.array([4.0, 4.0]), "t", "xrange",
        )
        assert resolved == scale(3.5, 4.0)

    def test_no_data_at_all_falls_back_to_an_interval_of_the_scales_own(self):
        empty = np.array([])
        assert _resolve_coordinate_range(None, empty, "t", "xrange") \
            == scale(0.0, 1.0)
        assert _resolve_coordinate_range(logscale(), empty, "t", "xrange") \
            == logscale(1.0, 10.0)

    def test_a_scale_with_no_room_to_widen_is_an_error(self):
        # a powscale flat at zero has no scale below its value to widen into
        with pytest.raises(ValueError, match="no room to widen"):
            _resolve_coordinate_range(
                powscale(exponent=2.0), np.array([0.0, 0.0]), "t", "xrange",
            )
