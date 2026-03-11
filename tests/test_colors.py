import numpy as np
import pytest

from matthewplotlib.colors import parse_color


class TestParseColorNamed:
    @pytest.mark.parametrize("name, expected", [
        ("black",   (0, 0, 0)),
        ("red",     (255, 0, 0)),
        ("green",   (0, 255, 0)),
        ("blue",    (0, 0, 255)),
        ("cyan",    (0, 255, 255)),
        ("magenta", (255, 0, 255)),
        ("yellow",  (255, 255, 0)),
        ("white",   (255, 255, 255)),
    ])
    def test_named_colors(self, name, expected):
        result = parse_color(name)
        assert np.array_equal(result, np.array(expected, dtype=np.uint8))

    @pytest.mark.parametrize("name", [
        "Black", "RED", "Green", "BLUE",
    ])
    def test_named_colors_case_insensitive(self, name):
        result = parse_color(name)
        assert result is not None
        assert result.dtype == np.uint8

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            parse_color("orange")


class TestParseColorHex:
    @pytest.mark.parametrize("hex_str, expected", [
        ("#f00", (255, 0, 0)),
        ("#0f0", (0, 255, 0)),
        ("#00f", (0, 0, 255)),
        ("#fff", (255, 255, 255)),
        ("#000", (0, 0, 0)),
    ])
    def test_short_hex(self, hex_str, expected):
        result = parse_color(hex_str)
        assert np.array_equal(result, np.array(expected, dtype=np.uint8))

    @pytest.mark.parametrize("hex_str, expected", [
        ("#ff0000", (255, 0, 0)),
        ("#00ff00", (0, 255, 0)),
        ("#0000ff", (0, 0, 255)),
        ("#808080", (128, 128, 128)),
    ])
    def test_full_hex(self, hex_str, expected):
        result = parse_color(hex_str)
        assert np.array_equal(result, np.array(expected, dtype=np.uint8))

    def test_short_hex_expansion(self):
        # #abc should become #aabbcc = (170, 187, 204)
        result = parse_color("#abc")
        assert np.array_equal(result, np.array((170, 187, 204), dtype=np.uint8))


class TestParseColorTuples:
    def test_int_tuple(self):
        result = parse_color((255, 128, 0))
        assert np.array_equal(result, np.array((255, 128, 0), dtype=np.uint8))

    def test_float_tuple(self):
        result = parse_color((1.0, 0.5, 0.0))
        assert result is not None
        assert result[0] == 255
        assert result[1] == 127
        assert result[2] == 0

    def test_int_list(self):
        result = parse_color([0, 128, 255])
        assert np.array_equal(result, np.array((0, 128, 255), dtype=np.uint8))

    def test_ndarray_int(self):
        result = parse_color(np.array([100, 200, 50], dtype=np.uint8))
        assert np.array_equal(result, np.array((100, 200, 50), dtype=np.uint8))

    def test_ndarray_float(self):
        result = parse_color(np.array([0.0, 0.5, 1.0]))
        assert result is not None
        assert result[0] == 0
        assert result[2] == 255

    def test_float_clipping(self):
        result = parse_color((1.5, -0.5, 0.5))
        assert result is not None
        assert result[0] == 255
        assert result[1] == 0
        assert result[2] == 127

    def test_int_clipping(self):
        result = parse_color(np.array([300, -10, 128], dtype=np.int32))
        assert result is not None
        assert result[0] == 255
        assert result[1] == 0
        assert result[2] == 128


class TestParseColorNone:
    def test_none_returns_none(self):
        assert parse_color(None) is None


class TestParseColorOutput:
    def test_output_dtype_is_uint8(self):
        result = parse_color("red")
        assert result is not None
        assert result.dtype == np.uint8

    def test_output_shape_is_3(self):
        result = parse_color("red")
        assert result is not None
        assert result.shape == (3,)
