import datetime

import numpy as np
import pytest

from matthewplotlib.data import (
    parse_date,
    parse_date_series,
    parse_range,
    parse_series,
    parse_multiple_series,
    parse_series3,
    parse_multiple_series3,
    xaxis,
    yaxis,
    zaxis,
)


# # #
# parse_range


class TestParseRange:
    def test_none_range_infers_from_data(self):
        data = np.array([1.0, 5.0, 3.0])
        lo, hi = parse_range(data, None)
        assert lo == 1.0
        assert hi == 5.0

    def test_explicit_range(self):
        data = np.array([1.0, 5.0, 3.0])
        lo, hi = parse_range(data, (0.0, 10.0))
        assert lo == 0.0
        assert hi == 10.0

    def test_partial_range_lo_only(self):
        data = np.array([1.0, 5.0, 3.0])
        lo, hi = parse_range(data, (0.0, None))
        assert lo == 0.0
        assert hi == 5.0

    def test_partial_range_hi_only(self):
        data = np.array([1.0, 5.0, 3.0])
        lo, hi = parse_range(data, (None, 10.0))
        assert lo == 1.0
        assert hi == 10.0

    def test_single_element(self):
        """One point reaches no distance, so it is given room around itself
        rather than a range of zero width to be divided by. `np.histogram2d`
        already expanded such a range by the same half either side, so what a
        scatter draws is unchanged; what it reports as its range is now what it
        drew."""
        data = np.array([3.0])
        lo, hi = parse_range(data, None)
        assert lo == 2.5
        assert hi == 3.5

    def test_a_constant_series_is_given_room(self):
        lo, hi = parse_range(np.full(10, 4.0), None)
        assert (lo, hi) == (3.5, 4.5)

    def test_gaps_do_not_describe_how_far_the_data_reaches(self):
        data = np.array([1.0, np.nan, 3.0, np.inf])
        assert parse_range(data, None) == (1.0, 3.0)

    def test_no_data_at_all(self):
        lo, hi = parse_range(np.array([]), None)
        assert lo < hi


# # #
# parse_series


class TestParseSeries:
    def test_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        xs, ys, cs = parse_series(arr)
        assert np.array_equal(xs, [1.0, 3.0, 5.0])
        assert np.array_equal(ys, [2.0, 4.0, 6.0])
        assert cs.shape == (3, 3)

    def test_tuple_of_arrays(self):
        xs_in = np.array([1.0, 2.0, 3.0])
        ys_in = np.array([4.0, 5.0, 6.0])
        xs, ys, cs = parse_series((xs_in, ys_in))
        assert np.array_equal(xs, xs_in)
        assert np.array_equal(ys, ys_in)
        assert cs.shape == (3, 3)

    def test_tuple_with_colors(self):
        xs_in = np.array([1.0, 2.0])
        ys_in = np.array([3.0, 4.0])
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        xs, ys, cs = parse_series((xs_in, ys_in, colors))
        assert np.array_equal(xs, xs_in)
        assert np.array_equal(ys, ys_in)
        assert np.array_equal(cs, colors)

    def test_2d_array_with_colors(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        colors = "red"
        xs, ys, cs = parse_series((arr, colors))
        assert np.array_equal(xs, [1.0, 3.0])
        assert np.array_equal(ys, [2.0, 4.0])
        assert np.all(cs[:, 0] == 255)

    def test_axis_object(self):
        a = xaxis(a=0, b=1, n=5)
        xs, ys, cs = parse_series(a)
        assert len(xs) == 5
        assert np.allclose(xs, np.linspace(0, 1, 5))
        assert np.all(ys == 0)

    def test_axis_with_colors(self):
        a = yaxis(a=0, b=1, n=3)
        xs, ys, cs = parse_series((a, "blue"))
        assert np.all(xs == 0)
        assert np.allclose(ys, np.linspace(0, 1, 3))
        assert np.all(cs[:, 2] == 255)

    def test_invalid_series_raises(self):
        with pytest.raises(TypeError):
            parse_series("not a series")

    def test_default_colors_are_white(self):
        arr = np.array([[0.0, 0.0]])
        _, _, cs = parse_series(arr)
        assert np.all(cs == 255)


# # #
# parse_multiple_series


class TestParseMultipleSeries:
    def test_concatenates_multiple(self):
        s1 = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        s2 = (np.array([5.0]), np.array([6.0]))
        xs, ys, cs = parse_multiple_series(s1, s2)
        assert len(xs) == 3
        assert np.array_equal(xs, [1.0, 2.0, 5.0])
        assert np.array_equal(ys, [3.0, 4.0, 6.0])
        assert cs.shape == (3, 3)


# # #
# parse_series3


class TestParseSeries3:
    def test_3d_array(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        xs, ys, zs, cs = parse_series3(arr)
        assert np.array_equal(xs, [1, 4])
        assert np.array_equal(ys, [2, 5])
        assert np.array_equal(zs, [3, 6])
        assert cs.shape == (2, 3)

    def test_tuple_of_three_arrays(self):
        xs_in = np.array([1.0, 2.0])
        ys_in = np.array([3.0, 4.0])
        zs_in = np.array([5.0, 6.0])
        xs, ys, zs, cs = parse_series3((xs_in, ys_in, zs_in))
        assert np.array_equal(xs, xs_in)
        assert np.array_equal(ys, ys_in)
        assert np.array_equal(zs, zs_in)

    def test_tuple_with_colors(self):
        xs_in = np.array([1.0])
        ys_in = np.array([2.0])
        zs_in = np.array([3.0])
        colors = np.array([[255, 0, 0]], dtype=np.uint8)
        xs, ys, zs, cs = parse_series3((xs_in, ys_in, zs_in, colors))
        assert np.array_equal(cs, colors)

    def test_axis_object(self):
        a = zaxis(a=0, b=1, n=5)
        xs, ys, zs, cs = parse_series3(a)
        assert np.all(xs == 0)
        assert np.all(ys == 0)
        assert np.allclose(zs, np.linspace(0, 1, 5))

    def test_invalid_series3_raises(self):
        with pytest.raises(TypeError):
            parse_series3("not a series")


class TestParseMultipleSeries3:
    def test_concatenates_multiple(self):
        s1 = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
        s2 = (np.array([4.0]), np.array([5.0]), np.array([6.0]))
        xs, ys, zs, cs = parse_multiple_series3(s1, s2)
        assert len(xs) == 2
        assert np.array_equal(xs, [1.0, 4.0])


# # #
# parse_date and parse_date_series


class TestParseDate:
    def test_a_date_is_itself(self):
        assert parse_date(datetime.date(2025, 3, 4)) == datetime.date(2025, 3, 4)

    def test_a_datetime_loses_its_time_of_day(self):
        stamp = datetime.datetime(2025, 3, 4, 11, 30)
        assert parse_date(stamp) == datetime.date(2025, 3, 4)

    def test_a_numpy_datetime_loses_its_time_of_day(self):
        stamp = np.datetime64("2025-03-04T11:30")
        assert parse_date(stamp) == datetime.date(2025, 3, 4)

    def test_an_iso_string(self):
        assert parse_date("2025-03-04") == datetime.date(2025, 3, 4)

    def test_something_that_is_not_a_date(self):
        with pytest.raises(TypeError, match="Invalid date"):
            parse_date(20250304)


class TestParseDateSeries:
    def test_a_mapping_is_sorted_by_date(self):
        dates, values = parse_date_series({
            datetime.date(2025, 1, 3): 3.0,
            datetime.date(2025, 1, 1): 1.0,
        })
        assert dates == [datetime.date(2025, 1, 1), datetime.date(2025, 1, 3)]
        assert np.array_equal(values, [1.0, 3.0])

    def test_separate_sequences_of_dates_and_values(self):
        dates, values = parse_date_series((
            ["2025-01-01", "2025-01-03"],
            [1.0, 3.0],
        ))
        assert dates == [datetime.date(2025, 1, 1), datetime.date(2025, 1, 3)]
        assert np.array_equal(values, [1.0, 3.0])

    def test_one_date_stands_in_for_consecutive_days(self):
        dates, values = parse_date_series(("2025-01-30", [1.0, 2.0, 3.0]))
        assert dates == [
            datetime.date(2025, 1, 30),
            datetime.date(2025, 1, 31),
            datetime.date(2025, 2, 1),
        ]
        assert np.array_equal(values, [1.0, 2.0, 3.0])

    def test_values_follow_their_own_dates_when_sorted(self):
        """The pairing has to survive the reordering, not just the dates."""
        dates, values = parse_date_series((
            ["2025-01-03", "2025-01-01", "2025-01-02"],
            [30.0, 10.0, 20.0],
        ))
        assert np.array_equal(values, [10.0, 20.0, 30.0])

    def test_a_numpy_array_of_dates(self):
        dates, values = parse_date_series((
            np.array(["2025-01-02", "2025-01-01"], dtype="datetime64[D]"),
            [2.0, 1.0],
        ))
        assert dates == [datetime.date(2025, 1, 1), datetime.date(2025, 1, 2)]
        assert np.array_equal(values, [1.0, 2.0])

    def test_more_values_than_dates(self):
        with pytest.raises(ValueError, match="as many values as dates"):
            parse_date_series((["2025-01-01"], [1.0, 2.0]))

    def test_a_repeated_date(self):
        with pytest.raises(ValueError, match="more than once"):
            parse_date_series((["2025-01-01", "2025-01-01"], [1.0, 2.0]))

    def test_values_with_more_than_one_axis(self):
        with pytest.raises(ValueError, match="one axis of values"):
            parse_date_series(("2025-01-01", [[1.0], [2.0]]))

    def test_something_that_is_not_a_series(self):
        with pytest.raises(TypeError, match="Invalid DateSeries"):
            parse_date_series(42)
