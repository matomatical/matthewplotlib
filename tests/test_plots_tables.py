"""Unit tests for tables."""

import numpy as np
import pytest

from matthewplotlib.plots import table

from tests.plots_common import RecordingColormap


# # #
# table


RUNS = [
    {"run": "baseline", "lr": 1e-3, "acc": 0.873},
    {"run": "wider", "lr": 3e-4, "acc": 0.902},
]


def _rows(plot):
    """The lines of a table, as plain text."""
    return plot.chars.to_plain_str().splitlines()


class TestTableData:
    def test_a_list_of_dicts_takes_its_columns_from_the_keys(self):
        plot = table(RUNS)

        assert plot.headers == ["run", "lr", "acc"]
        assert plot.num_rows == 2
        assert _rows(plot)[1].split() == ["run", "lr", "acc"]

    def test_a_dict_of_lists_takes_its_columns_from_the_keys(self):
        plot = table({"run": ["baseline", "wider"], "acc": [0.873, 0.902]})

        assert plot.headers == ["run", "acc"]
        assert plot.num_rows == 2

    def test_a_2d_array_has_no_headers_of_its_own(self):
        plot = table(np.arange(6).reshape(2, 3))

        assert plot.headers is None
        assert plot.num_rows == 2
        assert plot.num_columns == 3

    def test_a_2d_array_can_be_given_headers(self):
        plot = table(np.arange(6).reshape(2, 3), headers=["a", "b", "c"])

        assert plot.headers == ["a", "b", "c"]

    def test_headers_pick_out_and_order_the_columns(self):
        plot = table(RUNS, headers=["acc", "run"])

        assert plot.headers == ["acc", "run"]
        assert _rows(plot)[3].split() == ["0.873", "baseline"]

    def test_headers_given_as_a_mapping_rename_as_they_pick(self):
        plot = table(RUNS, headers={"acc": "accuracy"})

        assert plot.headers == ["accuracy"]

    def test_a_row_missing_a_key_leaves_that_cell_blank(self):
        plot = table([{"a": 1, "b": 2}, {"a": 3}])

        assert _rows(plot)[4].split() == ["3"]

    def test_a_short_column_runs_out_into_blanks(self):
        plot = table({"a": [1, 2, 3], "b": [4]})

        assert plot.num_rows == 3
        assert _rows(plot)[5].split() == ["3"]

    def test_a_value_of_none_is_blank_however_it_is_formatted(self):
        plot = table([[None]], formats=".3f")

        assert _rows(plot)[1].strip() == ""


class TestTableFormatting:
    def test_a_float_is_shown_to_four_significant_figures(self):
        plot = table([[1 / 3]])

        assert _rows(plot)[1].strip() == "0.3333"

    def test_a_format_spec_applies_to_every_column(self):
        plot = table([[1.5, 2.5]], formats=".2f")

        assert _rows(plot)[1].split() == ["1.50", "2.50"]

    def test_a_template_with_a_field_in_it_is_filled_in(self):
        plot = table([[0.873]], formats="{:.1%}")

        assert _rows(plot)[1].strip() == "87.3%"

    def test_a_callable_formats_a_value_however_it_likes(self):
        plot = table([["ab"]], formats=lambda value: value.upper())

        assert _rows(plot)[1].strip() == "AB"

    def test_a_format_can_be_given_per_column_by_name(self):
        plot = table(RUNS, formats={"acc": "{:.1%}"})

        assert _rows(plot)[3].split() == ["baseline", "0.001", "87.3%"]

    def test_a_column_is_cut_to_the_widest_it_may_be(self):
        plot = table([["abcdefgh"]], max_col_width=4)

        assert _rows(plot)[1].strip() == "abc…"

    def test_a_cell_with_newlines_in_it_grows_its_row(self):
        plot = table([["one\ntwo", "x"]], toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["one  x", "two   "]


class TestTableAlignment:
    def test_a_column_of_numbers_is_aligned_right_by_default(self):
        plot = table([[1], [100]], toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["  1", "100"]

    def test_a_column_of_anything_else_is_aligned_left(self):
        plot = table([["a"], ["bcd"]], toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["a  ", "bcd"]

    def test_a_column_of_numbers_with_a_gap_is_still_aligned_right(self):
        plot = table([[1], [None], [100]], toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["  1", "   ", "100"]

    def test_an_alignment_applies_to_every_column(self):
        plot = table(
            [["a"], ["bcd"]],
            aligns="center",
            toprule="skip",
            bottomrule="skip",
        )

        assert _rows(plot) == [" a ", "bcd"]

    def test_a_header_follows_the_alignment_of_its_column(self):
        plot = table([[1000]], headers=["n"], toprule="skip", midrule="skip")

        assert _rows(plot)[0] == "   n"


class TestTableRules:
    def test_the_default_is_a_double_rule_under_the_header(self):
        plot = table(RUNS)
        top, _header, mid, _first, _second, bottom = _rows(plot)

        assert set(top) == {"─"}
        assert set(mid) == {"═"}
        assert set(bottom) == {"─"}

    def test_a_table_with_no_header_row_has_no_midrule(self):
        plot = table([[1], [2]])

        assert len(_rows(plot)) == 2 + 2

    def test_every_rule_can_be_drawn_at_once(self):
        plot = table(
            [[1, 2]],
            headers=["a", "b"],
            leftrule="single",
            colrule="single",
            rightrule="single",
        )

        assert _rows(plot) == ["┌───┬───┐", "│ a │ b │", "╞═══╪═══╡",
                               "│ 1 │ 2 │", "└───┴───┘"]

    def test_a_blank_rule_holds_its_row_open(self):
        ruled = table([[1], [2]], rowrule="single")
        blank = table([[1], [2]], rowrule="blank")

        assert _rows(ruled)[2] == "─"
        assert _rows(blank)[2] == " "

    def test_a_skipped_rule_takes_no_space(self):
        plot = table([[1]], toprule="skip", bottomrule="skip")

        assert plot.height == 1

    def test_an_index_gets_a_column_of_its_own(self):
        plot = table([[1], [2]], index=["a", "b"], index_name="i",
                     headers=["n"], indexrule="single")

        assert _rows(plot)[1] == "i │ n"

    def test_a_rule_has_to_be_one_of_the_four_weights(self):
        with pytest.raises(ValueError, match="not 'dotted'"):
            table([[1]], toprule="dotted")

    def test_a_midrule_needs_a_header_to_go_under(self):
        with pytest.raises(ValueError, match="no header row"):
            table([[1]], midrule="single")

    def test_an_indexrule_needs_an_index_to_go_beside(self):
        with pytest.raises(ValueError, match="no index"):
            table([[1]], indexrule="single")


class TestTablePadding:
    def test_a_cell_is_held_away_from_the_rules_beside_it(self):
        plot = table([[1, 2]], leftrule="single", rightrule="single",
                     colrule="single", toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["│ 1 │ 2 │"]

    def test_an_edge_with_no_rule_on_it_is_not_padded(self):
        plot = table([[1, 2]], toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["1  2"]

    def test_the_padding_can_be_widened(self):
        plot = table([[1, 2]], cell_padding=2, toprule="skip",
                     bottomrule="skip")

        assert _rows(plot) == ["1    2"]

    def test_the_padding_cannot_be_negative(self):
        with pytest.raises(ValueError, match="cell_padding|-1 columns"):
            table([[1]], cell_padding=-1)


class TestTableColors:
    def test_the_body_can_be_colored_cell_by_cell(self):
        plot = table(
            [[1, 2]],
            bgcolors=np.array([[[255, 0, 0], [0, 0, 255]]]),
            toprule="skip",
            bottomrule="skip",
        )

        assert tuple(plot.chars.bg_rgb[0, 0]) == (255, 0, 0)
        assert tuple(plot.chars.bg_rgb[0, -1]) == (0, 0, 255)

    def test_a_colormap_shades_the_body_by_its_values(self):
        colormap = RecordingColormap()
        table([[0.25, 0.75]], bgcolors=[[0.25, 0.75]], colormap=colormap)

        assert colormap.input.tolist() == [[0.25, 0.75]]

    def test_the_header_and_the_index_take_the_header_color(self):
        plot = table(
            [[1]],
            headers=["n"],
            index=["r"],
            header_color="red",
            color="green",
        )

        # the header row, then the index label on the row below the midrule
        assert tuple(plot.chars.fg_rgb[1, 0]) == (255, 0, 0)
        assert tuple(plot.chars.fg_rgb[3, 0]) == (255, 0, 0)
        assert tuple(plot.chars.fg_rgb[3, -1]) == (0, 255, 0)

    def test_the_rules_follow_the_table_color_unless_told_otherwise(self):
        plot = table([[1]], color="green")
        recolored = table([[1]], color="green", rule_color="red")

        assert tuple(plot.chars.fg_rgb[0, 0]) == (0, 255, 0)
        assert tuple(recolored.chars.fg_rgb[0, 0]) == (255, 0, 0)


class TestTableRejections:
    def test_it_needs_something_to_tabulate(self):
        with pytest.raises(ValueError, match="at least one column"):
            table([])

    def test_the_rows_all_have_to_be_the_same_length(self):
        with pytest.raises(ValueError, match="row 1 has 1 values"):
            table([[1, 2], [3]])

    def test_a_row_has_to_be_a_row_of_values(self):
        with pytest.raises(ValueError, match="not a row of values"):
            table([1, 2, 3])

    def test_a_header_has_to_name_a_column_the_data_has(self):
        with pytest.raises(ValueError, match="no column 'nope'"):
            table(RUNS, headers=["nope"])

    def test_a_format_has_to_name_a_column_the_table_has(self):
        with pytest.raises(ValueError, match="not a column"):
            table(RUNS, formats={"nope": ".2f"})

    def test_a_format_per_column_needs_one_for_each(self):
        with pytest.raises(ValueError, match="2 entries for 3 columns"):
            table(RUNS, formats=[".2f", ".2f"])

    def test_a_format_per_column_name_needs_headers(self):
        with pytest.raises(ValueError, match="no headers"):
            table([[1]], formats={"a": ".2f"})

    def test_a_column_can_only_be_aligned_three_ways(self):
        with pytest.raises(ValueError, match="cannot align"):
            table([[1]], aligns="middle")

    def test_the_index_has_to_have_a_label_per_row(self):
        with pytest.raises(ValueError, match="2 index labels for 1 rows"):
            table([[1]], index=["a", "b"])

    def test_a_2d_table_has_no_column_names_to_rename(self):
        with pytest.raises(ValueError, match="names none"):
            table([[1]], headers={"a": "b"})

    def test_a_cell_cannot_smuggle_in_control_characters(self):
        with pytest.raises(ValueError, match="control characters"):
            table([["a\x1b[31mb"]])
