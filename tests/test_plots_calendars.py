"""Unit tests for the calendar plots: month blocks and strips of weeks."""

import datetime

import pytest

from matthewplotlib.plots import calendar, weeks

from matthewplotlib.scales import scale

from tests.plots_common import drawn_cells


# # #
# calendar


def a_month(month=1, year=2025, value=1.0):
    """Every day of one month, all with the same value."""
    day = datetime.date(year, month, 1)
    days = {}
    while day.month == month:
        days[day] = value
        day += datetime.timedelta(days=1)
    return days


class TestCalendar:
    def test_a_month_is_a_caption_a_header_and_its_weeks(self):
        plot = calendar(a_month(), cols=1, month_spacing=0)
        # January 2025 runs Wednesday to Friday, over five Monday-start weeks.
        assert plot.height == 1 + 1 + 5
        assert plot.width == 7 * 2

    def test_day_width_scales_the_block(self):
        plot = calendar(a_month(), cols=1, month_spacing=0, day_width=3)
        assert plot.width == 7 * 3

    def test_months_are_wrapped_into_a_grid(self):
        days = a_month(1) | a_month(2) | a_month(3) | a_month(4)
        plot = calendar(days, cols=2, month_spacing=0)
        assert plot.width == 2 * 7 * 2

    def test_the_spacing_is_not_left_on_the_far_edges(self):
        """A gap between the months should not become a margin around them."""
        days = a_month(1) | a_month(2)
        snug = calendar(days, cols=2, month_spacing=0)
        spaced = calendar(days, cols=2, month_spacing=1)
        # One gap between the two months, and none past the second.
        assert snug.width == 2 * 7 * 2
        assert spaced.width == 2 * 7 * 2 + 1 * 2
        # February 2025 ends on a Friday, so its last column has days in it.
        assert drawn_cells(spaced)[:, -1].any()

    def test_a_grid_is_no_wider_than_the_months_in_it(self):
        """Asking for four columns and giving it one month should not leave
        three columns of blank beside it."""
        for cols in (4, None):
            plot = calendar(a_month(), cols=cols, month_spacing=0)
            assert plot.width == 7 * 2

    def test_a_day_with_no_value_is_blank(self):
        days = a_month()
        del days[datetime.date(2025, 1, 15)]
        plot = calendar(days, cols=1, month_spacing=0)
        # The 15th is a Wednesday, in the third full week of the month.
        row, column = 2 + 2, 2 * 2
        assert not drawn_cells(plot)[row, column]

    def test_a_day_whose_value_is_not_finite_is_blank(self):
        days = a_month()
        days[datetime.date(2025, 1, 15)] = float("nan")
        plot = calendar(days, cols=1, month_spacing=0)
        assert not drawn_cells(plot)[2 + 2, 2 * 2]
        assert plot.num_days == len(days) - 1

    def test_a_day_whose_value_is_zero_is_drawn(self):
        """Distinctly from a day with no value at all."""
        days = a_month(value=0.0)
        plot = calendar(days, vrange=(0.0, 1.0), cols=1, month_spacing=0)
        assert drawn_cells(plot)[2 + 2, 2 * 2]
        assert plot.num_days == len(days)

    def test_days_outside_the_daterange_are_left_out(self):
        days = a_month(1) | a_month(2)
        plot = calendar(
            days,
            daterange=("2025-01-01", "2025-01-31"),
            cols=1,
            month_spacing=0,
        )
        assert plot.num_days == 31

    def test_the_daterange_can_reach_past_the_data(self):
        plot = calendar(
            a_month(1),
            daterange=("2025-01-01", "2025-02-28"),
            cols=2,
            month_spacing=0,
        )
        assert plot.width == 2 * 7 * 2

    def test_the_value_scale_spans_the_data_by_default(self):
        plot = calendar({"2025-01-01": 3.0, "2025-01-02": 9.0}, cols=1)
        assert plot.vrange == scale(3.0, 9.0)

    def test_a_given_value_scale_is_kept_as_it_stands(self):
        plot = calendar({"2025-01-01": 3.0}, vrange=(0.0, 10.0), cols=1)
        assert plot.vrange == scale(0.0, 10.0)

    def test_the_value_scale_ignores_the_days_with_no_value(self):
        days = {"2025-01-01": 3.0, "2025-01-02": float("nan")}
        plot = calendar(days, cols=1)
        assert plot.vrange == scale(3.0, 3.0)

    def test_the_first_weekday_shifts_the_columns(self):
        """The 1st of January 2025 is a Wednesday, so it lands in the third
        column of a Monday-start week and the fourth of a Sunday-start one."""
        days = a_month()
        monday = calendar(days, cols=1, month_spacing=0)
        sunday = calendar(days, cols=1, month_spacing=0, first_weekday=6)
        week = 2
        assert drawn_cells(monday)[week, 2 * 2]
        assert not drawn_cells(monday)[week, 1 * 2]
        assert drawn_cells(sunday)[week, 3 * 2]
        assert not drawn_cells(sunday)[week, 2 * 2]

    def test_the_labels_can_be_left_off(self):
        plot = calendar(
            a_month(),
            cols=1,
            month_spacing=0,
            month_labels=False,
            weekday_labels=False,
        )
        assert plot.height == 5

    def test_a_narrow_month_is_captioned_in_a_shorter_spelling(self):
        plot = calendar(a_month(), cols=1, month_spacing=0, day_width=1)
        caption = "".join(chr(c) for c in plot.chars.codes[0])
        assert caption == "Jan  25"

    def test_a_wide_month_is_captioned_in_full(self):
        plot = calendar(a_month(), cols=1, month_spacing=0)
        caption = "".join(chr(c) for c in plot.chars.codes[0])
        assert caption == "January   2025"

    def test_the_years_line_up_across_the_months(self):
        """Whichever months are drawn, and however long their names are."""
        days = a_month(5) | a_month(9)
        plot = calendar(days, cols=1, month_spacing=0)
        rows = ["".join(chr(c) for c in row) for row in plot.chars.codes]
        captions = [row for row in rows if "2025" in row]
        # May through September, since the months between them are filled in.
        assert len(captions) == 5
        assert captions[0] == "May       2025"
        assert captions[-1] == "September 2025"
        assert len({len(caption) for caption in captions}) == 1

    def test_it_needs_something_to_draw(self):
        with pytest.raises(ValueError, match="at least one"):
            calendar({})

    def test_the_daterange_has_to_run_forwards(self):
        with pytest.raises(ValueError, match="ends"):
            calendar(a_month(), daterange=("2025-02-01", "2025-01-01"))

    def test_a_day_has_to_have_a_width(self):
        with pytest.raises(ValueError, match="day_width"):
            calendar(a_month(), day_width=0)

    def test_the_spacing_cannot_be_negative(self):
        with pytest.raises(ValueError, match="month_spacing"):
            calendar(a_month(), month_spacing=-1)

    def test_the_week_has_to_start_on_a_weekday(self):
        with pytest.raises(ValueError, match="first_weekday"):
            calendar(a_month(), first_weekday=7)


# # #
# weeks


def a_year(year=2025, value=1.0):
    """Every day of one year, all with the same value."""
    day = datetime.date(year, 1, 1)
    days = {}
    while day.year == year:
        days[day] = value
        day += datetime.timedelta(days=1)
    return days


def rows_of(plot):
    """Each row of a plot as a string, for reading its captions back."""
    return ["".join(chr(code) for code in row) for row in plot.chars.codes]


class TestWeeks:
    def test_a_strip_is_two_captions_and_the_seven_weekdays(self):
        plot = weeks(a_year())
        # 2025 starts on a Wednesday, so its 365 days touch 53 Monday-weeks.
        assert plot.height == 2 + 7
        assert plot.width == 2 + 53 * 2
        assert plot.num_weeks == 53

    def test_day_width_scales_the_strip(self):
        plot = weeks(a_year(), day_width=3)
        assert plot.width == 2 + 53 * 3

    def test_the_strip_starts_at_the_top_of_its_first_week(self):
        """So that a weekday keeps to one row. The 1st of January 2025 is a
        Wednesday, so the Monday and Tuesday above it are blank."""
        plot = weeks(a_year())
        assert not drawn_cells(plot)[2 + 0, 2]
        assert not drawn_cells(plot)[2 + 1, 2]
        assert drawn_cells(plot)[2 + 2, 2]

    def test_the_weekdays_are_named_down_the_gutter(self):
        plot = weeks(a_year())
        assert [chr(plot.chars.codes[2 + i, 0]) for i in range(7)] == list(
            "MTWtFSs"
        )

    def test_the_first_weekday_rotates_the_rows(self):
        plot = weeks(a_year(), first_weekday=6)
        assert [chr(plot.chars.codes[2 + i, 0]) for i in range(7)] == list(
            "sMTWtFS"
        )

    def test_the_months_are_captioned(self):
        captions = " ".join(rows_of(weeks(a_year()))[:2])
        for month in ("Jan", "Feb", "Jun", "Dec"):
            assert month in captions

    def test_the_year_is_captioned_once(self):
        assert rows_of(weeks(a_year()))[0].count("2025") == 1

    def test_a_wide_strip_wraps_into_bands(self):
        plot = weeks(a_year(), width=80)
        assert plot.width == 80
        # Two bands of nine rows, with a blank row between them.
        assert plot.height == 9 + 1 + 9
        assert not drawn_cells(plot)[9, :].any()

    def test_every_band_names_its_year(self):
        """A band should be readable without looking back at the one above."""
        rows = rows_of(weeks(a_year(), width=80))
        assert "2025" in rows[0]
        assert "2025" in rows[10]

    def test_a_span_of_two_years_names_both(self):
        days = a_year(2024) | a_year(2025)
        captions = rows_of(weeks(days))[0]
        assert "2024" in captions
        assert "2025" in captions

    def test_a_caption_that_will_not_fit_is_dropped_not_truncated(self):
        months = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        plot = weeks(a_year(), width=20)
        # Only the caption rows, so that the weekday initials in the gutter of
        # each day row are not mistaken for clipped month names.
        drawn = []
        for row, codes in zip(rows_of(plot), plot.chars.codes):
            if (codes == ord("\u2588")).any():
                continue
            drawn.extend(run for run in row.split() if run.isalpha())
        # Narrow bands leave some months no room, but never half a name.
        assert drawn
        assert all(run in months for run in drawn)
        assert len(drawn) < 12

    def test_the_captions_can_be_left_off(self):
        assert weeks(a_year(), year_labels=False).height == 1 + 7
        assert weeks(a_year(), month_labels=False).height == 1 + 7
        assert weeks(
            a_year(), year_labels=False, month_labels=False
        ).height == 7

    def test_the_gutter_can_be_left_off(self):
        assert weeks(a_year(), weekday_labels=False).width == 53 * 2

    def test_a_day_with_no_value_is_blank(self):
        days = a_year()
        del days[datetime.date(2025, 1, 8)]
        plot = weeks(days)
        # The 8th is the Wednesday of the second week.
        assert not drawn_cells(plot)[2 + 2, 2 + 2]

    def test_a_day_whose_value_is_not_finite_is_blank(self):
        days = a_year()
        days[datetime.date(2025, 1, 8)] = float("nan")
        plot = weeks(days)
        assert not drawn_cells(plot)[2 + 2, 2 + 2]
        assert plot.num_days == 364

    def test_a_day_whose_value_is_zero_is_drawn(self):
        plot = weeks(a_year(value=0.0), vrange=(0.0, 1.0))
        assert drawn_cells(plot)[2 + 2, 2]
        assert plot.num_days == 365

    def test_days_outside_the_daterange_are_left_out(self):
        plot = weeks(a_year(), daterange=("2025-01-01", "2025-01-31"))
        assert plot.num_days == 31

    def test_the_value_scale_spans_the_data_by_default(self):
        plot = weeks({"2025-01-01": 3.0, "2025-01-02": 9.0})
        assert plot.vrange == scale(3.0, 9.0)

    def test_it_draws_the_same_days_as_a_calendar_would(self):
        """The two share the front end that decides which days get a colour."""
        days = a_year()
        days[datetime.date(2025, 5, 5)] = float("nan")
        del days[datetime.date(2025, 6, 6)]
        strip = weeks(days, daterange=("2025-02-01", "2025-11-30"))
        grid = calendar(days, daterange=("2025-02-01", "2025-11-30"))
        assert strip.num_days == grid.num_days
        assert strip.vrange == grid.vrange

    def test_it_needs_something_to_draw(self):
        with pytest.raises(ValueError, match="at least one"):
            weeks({})

    def test_the_daterange_has_to_run_forwards(self):
        with pytest.raises(ValueError, match="ends"):
            weeks(a_year(), daterange=("2025-02-01", "2025-01-01"))

    def test_a_day_has_to_have_a_width(self):
        with pytest.raises(ValueError, match="day_width"):
            weeks(a_year(), day_width=0)

    def test_the_week_has_to_start_on_a_weekday(self):
        with pytest.raises(ValueError, match="first_weekday"):
            weeks(a_year(), first_weekday=7)

    def test_the_width_has_to_fit_the_gutter_and_a_week(self):
        with pytest.raises(ValueError, match="width must leave room"):
            weeks(a_year(), width=3)

    def test_the_narrowest_width_is_one_week_per_band(self):
        plot = weeks(a_year(), width=4)
        assert plot.width == 4
        assert plot.height == 53 * 9 + 52
