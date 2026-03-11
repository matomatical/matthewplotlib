import numpy as np
import pytest

from matthewplotlib.data import (
    parse_range,
    parse_color_spec,
    parse_series,
    parse_multiple_series,
    parse_series3,
    parse_multiple_series3,
    xaxis,
    yaxis,
    zaxis,
    project3,
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
        data = np.array([3.0])
        lo, hi = parse_range(data, None)
        assert lo == 3.0
        assert hi == 3.0


# # #
# parse_color_spec


class TestParseColorSpec:
    def test_none_defaults_to_white(self):
        result = parse_color_spec(None, 5)
        assert result.shape == (5, 3)
        assert result.dtype == np.uint8
        assert np.all(result == 255)

    def test_single_color_broadcasts(self):
        result = parse_color_spec("red", 4)
        assert result.shape == (4, 3)
        assert np.all(result[:, 0] == 255)
        assert np.all(result[:, 1] == 0)
        assert np.all(result[:, 2] == 0)

    def test_per_point_colors(self):
        colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ], dtype=np.uint8)
        result = parse_color_spec(colors, 3)
        assert np.array_equal(result, colors)


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
# project3


class TestProject3:
    def test_origin_projects_to_origin(self):
        """A point at the camera target should project near (0, 0)."""
        xyz = np.array([[0.0, 0.0, 0.0]])
        xy, valid = project3(xyz)
        assert valid[0]
        assert np.allclose(xy[0], [0.0, 0.0], atol=1e-10)

    def test_default_camera_x_maps_to_screen_x(self):
        """With default camera (on +Z looking at origin), +X should map to
        positive screen X (left in camera coords)."""
        xyz = np.array([[0.5, 0.0, 0.0]])
        xy, valid = project3(xyz)
        assert valid[0]
        # x offset should produce nonzero screen x
        assert xy[0, 0] != 0.0

    def test_default_camera_y_maps_to_screen_y(self):
        """With default camera, +Y should map to positive screen Y."""
        xyz = np.array([[0.0, 0.5, 0.0]])
        xy, valid = project3(xyz)
        assert valid[0]
        assert xy[0, 1] > 0.0

    def test_point_behind_camera_is_invalid(self):
        """A point behind the camera should be marked invalid."""
        # Default camera at (0,0,2) looking towards origin.
        # A point at (0,0,3) is behind the camera.
        xyz = np.array([[0.0, 0.0, 3.0]])
        _, valid = project3(xyz)
        assert not valid[0]

    def test_point_in_front_is_valid(self):
        """A point between camera and target should be valid."""
        xyz = np.array([[0.0, 0.0, 1.0]])
        _, valid = project3(xyz)
        assert valid[0]

    def test_closer_points_project_larger(self):
        """Perspective: a point closer to camera should have larger projected
        coordinates than the same offset point further away."""
        near = np.array([[1.0, 0.0, 0.5]])
        far = np.array([[1.0, 0.0, -1.0]])
        xy_near, valid_near = project3(near)
        xy_far, valid_far = project3(far)
        assert valid_near[0] and valid_far[0]
        assert abs(xy_near[0, 0]) > abs(xy_far[0, 0])

    def test_output_shapes(self):
        """Output shapes should be (n, 2) and (n,)."""
        xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        xy, valid = project3(xyz)
        assert xy.shape == (3, 2)
        assert valid.shape == (3,)

    def test_symmetric_projection(self):
        """Points symmetric about the view axis should have symmetric
        projections."""
        left = np.array([[1.0, 0.0, 0.0]])
        right = np.array([[-1.0, 0.0, 0.0]])
        xy_left, _ = project3(left)
        xy_right, _ = project3(right)
        assert np.allclose(xy_left[0, 0], -xy_right[0, 0], atol=1e-10)
        assert np.allclose(xy_left[0, 1], xy_right[0, 1], atol=1e-10)

    def test_custom_camera_position(self):
        """Camera on +X axis looking at origin: Y scene axis should map to
        some screen axis."""
        xyz = np.array([[0.0, 1.0, 0.0]])
        xy, valid = project3(
            xyz,
            camera_position=np.array([5.0, 0.0, 0.0]),
            camera_target=np.zeros(3),
        )
        assert valid[0]
        # The point is offset in Y, so it should project to nonzero
        assert np.any(xy[0] != 0.0)

    def test_fov_scaling(self):
        """Narrower FOV should produce larger projected coordinates."""
        xyz = np.array([[0.5, 0.0, 0.0]])
        xy_wide, _ = project3(xyz, fov_degrees=120.0)
        xy_narrow, _ = project3(xyz, fov_degrees=30.0)
        assert abs(xy_narrow[0, 0]) > abs(xy_wide[0, 0])
