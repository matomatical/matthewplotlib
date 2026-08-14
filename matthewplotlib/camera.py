"""
Seeing three dimensions from a camera.

A camera is a position, a thing it is pointed at, and how much of the world it
takes in. This module turns points and line segments in space into positions on
the film, for the 3d plot types in `matthewplotlib.plots`.

The two stages:

* `view_matrix`: The coordinate system the camera sees the scene in.
* `perspective`: Divide out the depth of points already in that system.

End to end:

* `project3`: Project 3d points onto the viewing plane of a camera.
* `project3_segments`: The same for line segments, cutting those that reach
  behind the camera. A segment needs its own function because its image is not
  the line between the images of its ends.
"""

from __future__ import annotations

import numpy as np


def view_matrix(
    camera_position: np.ndarray = np.array([0., 0., 2.]),   # float[3]
    camera_target: np.ndarray = np.zeros(3),                # float[3]
    scene_up: np.ndarray = np.array([0.,1.,0.]),            # float[3]
) -> np.ndarray: # float[3, 3]
    """
    The basis of a camera's own coordinate system, as columns: X to the right,
    Y up, and Z towards whatever the camera is pointed at.

    Inputs:

    * camera_position: float[3] (default: [0. 0. 2.]).
        The position at which the camera is placed.
    * camera_target: float[3] (default: [0. 0. 0.]).
        The position towards which the camera is facing. Should be distinct
        from camera position.
    * scene_up: float[3] (default: [0. 1. 0.]).
        The unit vector designating the 'up' direction for the scene. Should
        not have the same direction as camera_target - camera_position.

    Returns:

    * V: float[3, 3].
        Right-multiply a displacement from the camera by this to express it in
        the camera's coordinates.
    """
    V_z = camera_target - camera_position
    V_z = V_z / np.linalg.norm(V_z)
    V_x = np.cross(V_z, scene_up)
    V_x = V_x / np.linalg.norm(V_x)
    V_y = np.cross(V_x, V_z)
    return np.array([V_x, V_y, V_z]).T


def perspective(
    xyz: np.ndarray,                    # float[n, 3]
    fov_degrees: float = 90.0,
    valid: np.ndarray | None = None,    # bool[n]
) -> np.ndarray: # float[n, 2]
    """
    Where points already in camera coordinates land on the film.

    Inputs:

    * xyz: float[n, 3].
        Points in the camera's own coordinate system, so that the third column
        is depth into the scene.
    * fov_degrees: float (default 90).
        Field of view. Points within a cone (or frustum) of this angle leaving
        the camera are projected into the unit disk (or the square [-1,1]^2).
    * valid: optional bool[n].
        Which points to divide, for callers holding points at or behind the
        camera. Those come back as zeros rather than as a division by a depth
        of zero or less.

    Returns:

    * xy: float[n, 2].
        Projected points.
    """
    focal_length = 1 / np.tan(np.radians(fov_degrees) / 2)
    if valid is None:
        return focal_length * xyz[:, :2] / xyz[:, 2, np.newaxis]
    xy = np.zeros((len(xyz), 2))
    np.divide(
        xyz[:, :2],
        xyz[:, 2, np.newaxis],
        out=xy,
        where=valid[:, np.newaxis],
    )
    return focal_length * xy


def project3(
    xyz: np.ndarray, # float[n, 3]
    camera_position: np.ndarray = np.array([0., 0., 2.]), # float[3]
    camera_target: np.ndarray = np.zeros(3), # float[3]
    scene_up: np.ndarray = np.array([0.,1.,0.]), # float[3]
    fov_degrees: float = 90.0,
) -> tuple[
    np.ndarray, # float[n, 2]
    np.ndarray, # bool[n]
]:
    """
    Project a 3d point cloud into two dimensions based on a given camera
    configuration.

    Inputs:

    * xyz: float[n, 3].
        The points to project, with columns corresponding to X, Y, and Z.
    * camera_position: float[3] (default: [0. 0. 2.]).
        The position at which the camera is placed. The default is positioned
        along the positive Z axis.
    * camera_target: float[3] (default: [0. 0. 0.]).
        The position towards which the camera is facing. Should be distinct
        from camera position. The default is that the camera is facing towards
        the origin.
    * scene_up: float[3] (default: [0. 1. 0.]).
        The unit vector designating the 'up' direction for the scene. The
        default is the positive Y direction. Should not have the same direction
        as camera_target - camera_position.
    * fov_degrees: float (default 90).
        Field of view. Points within a cone (or frustum) of this angle leaving
        the camera are projected into the unit disk (or the square [-1,1]^2).

    Returns:

    * xy: float[n, 2].
        Projected points.
    * valid: bool[n].
        Mask indicating which of the points are in front of the camera.

    Notes:

    * The combined effect of the defaults is that the camera is looking down
      the Z axis towards the origin from the positive direction, with the X
      axis extending towards the right and the Y axis extending upwards, with
      the field of view ensuring that points within the cube [-1,1]^3 are
      projected into the square [-1,1]^2.
    * The valid mask only considers whether points are in front of the camera.
      A more comprehensive frustum clipping approach is not supported.
    
    Internal notes:

    * This implementation uses a coordinate system for the camera where X and Y
      point left and up respectively and Z points towards the object ahead of
      the camera (an alternative convention is for Z to point behind the
      camera).
    """
    # transform points to camera coordinate system
    V = view_matrix(
        camera_position=camera_position,
        camera_target=camera_target,
        scene_up=scene_up,
    )
    xyz_ = (xyz - camera_position) @ V

    # mask for valid points, and project the rest
    valid = xyz_[:, 2] > 0.
    xy = perspective(xyz_, fov_degrees=fov_degrees, valid=valid)

    return xy, valid


def project3_segments(
    starts: np.ndarray, # float[n, 3]
    ends: np.ndarray,   # float[n, 3]
    camera_position: np.ndarray = np.array([0., 0., 2.]),   # float[3]
    camera_target: np.ndarray = np.zeros(3),                # float[3]
    scene_up: np.ndarray = np.array([0.,1.,0.]),            # float[3]
    fov_degrees: float = 90.0,
    near: float = 1e-6,
) -> tuple[
    np.ndarray, # float[m, 2]
    np.ndarray, # float[m, 2]
    np.ndarray, # bool[n]
]:
    """
    Project 3d line segments onto the viewing plane of a camera.

    Inputs:

    * starts: float[n, 3].
        The point at which each segment begins.
    * ends: float[n, 3].
        The point at which each segment ends.
    * camera_position: float[3] (default: [0. 0. 2.]).
        The position at which the camera is placed.
    * camera_target: float[3] (default: [0. 0. 0.]).
        The position towards which the camera is facing. Should be distinct
        from camera position.
    * scene_up: float[3] (default: [0. 1. 0.]).
        The unit vector designating the 'up' direction for the scene. Should
        not have the same direction as camera_target - camera_position.
    * fov_degrees: float (default 90).
        Field of view. Points within a cone (or frustum) of this angle leaving
        the camera are projected into the unit disk (or the square [-1,1]^2).
    * near: float (default 1e-6).
        Distance in front of the camera at which segments are cut off.

    Returns:

    * xy_starts: float[m, 2].
        Where each drawn segment begins, projected.
    * xy_ends: float[m, 2].
        Where each drawn segment ends, projected.
    * drawn: bool[n].
        Which of the input segments are drawn at all. The two arrays of
        projected points have one entry per set bit, in order, so anything
        else the caller holds per segment should be masked with this.

    A segment with one end behind the camera is cut at the near plane, keeping
    the part in front. A segment with both ends behind it is not drawn, nor is
    one with a non-finite end. Projecting the endpoints without cutting would
    place that first kind of segment on the wrong side of the view, since
    perspective division by a negative depth reflects a point through the
    centre of the image.
    """
    V = view_matrix(
        camera_position=camera_position,
        camera_target=camera_target,
        scene_up=scene_up,
    )
    xyz_starts = (starts - camera_position) @ V
    xyz_ends = (ends - camera_position) @ V

    # segments that are not fully specified are stood down to the camera's own
    # position, which is behind the near plane, so they fall out below
    specified = (
        np.isfinite(xyz_starts).all(axis=1) & np.isfinite(xyz_ends).all(axis=1)
    )
    xyz_starts = np.where(specified[:, np.newaxis], xyz_starts, 0.)
    xyz_ends = np.where(specified[:, np.newaxis], xyz_ends, 0.)

    # which ends are far enough in front of the camera to project
    ahead_starts = xyz_starts[:, 2] > near
    ahead_ends = xyz_ends[:, 2] > near
    drawn = ahead_starts | ahead_ends

    # cut whichever end is behind back to the near plane. one crossing point
    # serves both cases, since a segment kept here crosses the plane at most
    # once
    depths = xyz_ends[:, 2] - xyz_starts[:, 2]
    t = np.divide(
        near - xyz_starts[:, 2],
        depths,
        out=np.zeros_like(depths),
        where=(depths != 0),
    )
    crossings = xyz_starts + t[:, np.newaxis] * (xyz_ends - xyz_starts)
    xyz_starts = np.where(ahead_starts[:, np.newaxis], xyz_starts, crossings)
    xyz_ends = np.where(ahead_ends[:, np.newaxis], xyz_ends, crossings)

    # project what is left, which is now all in front of the near plane
    xy_starts = perspective(xyz_starts[drawn], fov_degrees=fov_degrees)
    xy_ends = perspective(xyz_ends[drawn], fov_degrees=fov_degrees)

    return xy_starts, xy_ends, drawn
