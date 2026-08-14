"""Unit tests for the camera: view matrix, perspective, and projection."""

import numpy as np
import pytest

from matthewplotlib.camera import (
    view_matrix,
    perspective,
    project3,
    project3_segments,
)


# # #
# perspective


class TestPerspective:
    def test_divides_out_the_depth(self):
        # at a 90 degree field of view the focal length is one, so a point one
        # unit deep and one unit across lands at the edge of the view
        xy = perspective(np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 2.0]]))
        assert np.allclose(xy, [[1.0, 0.0], [0.5, 0.0]])

    def test_a_narrower_field_of_view_spreads_points_out(self):
        xyz = np.array([[0.5, 0.0, 1.0]])
        wide = perspective(xyz, fov_degrees=120.0)
        narrow = perspective(xyz, fov_degrees=30.0)
        assert abs(narrow[0, 0]) > abs(wide[0, 0])

    def test_points_left_out_come_back_as_zeros(self):
        """The caller holding points at or behind the camera says so, rather
        than dividing by a depth of zero or less."""
        xyz = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, -1.0]])
        valid = np.array([True, False, False])
        xy = perspective(xyz, valid=valid)
        assert np.allclose(xy[0], [1.0, 1.0])
        assert np.all(xy[1:] == 0.0)

    def test_no_points_at_all(self):
        assert perspective(np.zeros((0, 3))).shape == (0, 2)


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


# # #
# view_matrix


class TestViewMatrix:
    def test_columns_are_orthonormal(self):
        V = view_matrix(
            camera_position=np.array([3.0, 2.0, 1.0]),
            camera_target=np.array([0.0, 0.0, 0.0]),
        )
        assert np.allclose(V.T @ V, np.eye(3), atol=1e-12)

    def test_z_points_at_the_target(self):
        camera_position = np.array([0.0, 0.0, 5.0])
        camera_target = np.zeros(3)
        V = view_matrix(
            camera_position=camera_position,
            camera_target=camera_target,
        )
        towards = (camera_target - camera_position) @ V
        assert np.allclose(towards[:2], 0.0, atol=1e-12)
        assert towards[2] > 0

    def test_up_stays_up(self):
        V = view_matrix(
            camera_position=np.array([0.0, 0.0, 2.0]),
            camera_target=np.zeros(3),
            scene_up=np.array([0.0, 1.0, 0.0]),
        )
        assert (np.array([0.0, 1.0, 0.0]) @ V)[1] > 0


# # #
# project3_segments


class TestProject3Segments:
    def test_agrees_with_project3_when_both_ends_are_ahead(self):
        starts = np.array([[0.5, 0.5, 0.0], [-1.0, 0.0, 0.0]])
        ends = np.array([[-0.5, -0.5, 0.0], [1.0, 0.25, 0.0]])
        xy_starts, xy_ends, drawn = project3_segments(starts, ends)
        assert drawn.all()
        assert np.allclose(xy_starts, project3(starts)[0])
        assert np.allclose(xy_ends, project3(ends)[0])

    def test_a_segment_behind_the_camera_is_not_drawn(self):
        # the camera sits at z=2 looking towards the origin, so these are
        # both behind it
        starts = np.array([[0.0, 0.0, 3.0]])
        ends = np.array([[1.0, 0.0, 5.0]])
        _xy_starts, _xy_ends, drawn = project3_segments(starts, ends)
        assert not drawn.any()

    def test_a_segment_reaching_behind_the_camera_is_cut(self):
        # from behind the camera to in front of it, passing it on the +x side
        starts = np.array([[1.0, 0.0, 4.0]])
        ends = np.array([[1.0, 0.0, -4.0]])
        xy_starts, xy_ends, drawn = project3_segments(starts, ends, near=1e-3)
        assert drawn.all()
        assert np.all(np.isfinite(xy_starts))
        # the cut end is off the side of the view, not reflected through it: it
        # keeps the sign of the visible end
        assert xy_starts[0, 0] > 0
        assert xy_ends[0, 0] > 0
        assert abs(xy_starts[0, 0]) > abs(xy_ends[0, 0])

    def test_cutting_leaves_the_visible_end_alone(self):
        starts = np.array([[0.5, 0.25, 4.0]])
        ends = np.array([[0.5, 0.25, -2.0]])
        _xy_starts, xy_ends, _drawn = project3_segments(starts, ends)
        assert np.allclose(xy_ends, project3(ends)[0])

    def test_a_non_finite_end_is_not_drawn(self):
        starts = np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ends = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        _xy_starts, _xy_ends, drawn = project3_segments(starts, ends)
        assert drawn.tolist() == [False, True]

    def test_projected_arrays_have_one_entry_per_drawn_segment(self):
        starts = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0],
            [0.5, 0.0, 0.0],
        ])
        ends = np.array([
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 8.0],
            [-0.5, 0.0, 0.0],
        ])
        xy_starts, xy_ends, drawn = project3_segments(starts, ends)
        assert drawn.tolist() == [True, False, True]
        assert len(xy_starts) == len(xy_ends) == 2

    def test_no_segments_at_all(self):
        xy_starts, xy_ends, drawn = project3_segments(
            np.zeros((0, 3)),
            np.zeros((0, 3)),
        )
        assert xy_starts.shape == (0, 2)
        assert xy_ends.shape == (0, 2)
        assert drawn.shape == (0,)
