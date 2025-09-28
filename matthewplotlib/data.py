from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import einops
from numpy.typing import NDArray, ArrayLike

from matthewplotlib.colors import ColorLike, parse_color


# # # 
# Types


type number = int | float | np.integer | np.floating


type ColorSpec = (
    None
    | ColorLike
    | ArrayLike # uint8[n, 3]
)


type Series = (
    NDArray                                     # number[n,2]
    | tuple[NDArray, ColorSpec]                 # number[n,2], colors
    | tuple[ArrayLike, ArrayLike]               # number[n]^2
    | tuple[ArrayLike, ArrayLike, ColorSpec]    # number[n]^2, colors
    | axis                                      # axis
    | tuple[axis, ColorSpec]                    # axis, colors
)


type Series3 = (
    NDArray                                             # number[n,3]
    | tuple[NDArray, ColorSpec]                         # number[n,3], colors
    | tuple[ArrayLike, ArrayLike, ArrayLike]            # number[n]^3
    | tuple[ArrayLike, ArrayLike, ArrayLike, ColorSpec] # number[n]^3, colors
    | axis                                              # axis
    | tuple[axis, ColorSpec]                            # axis, uint8[n,rgb]
)


# # # 
# Parsers


def parse_range(
    data: NDArray,
    range: tuple[number | None, number | None] | None,
) -> tuple[number, number]:
    if range is None:
        range = (None, None)
    lo, hi = range
    if lo is None:
        lo = data.min()
    if hi is None:
        hi = data.max()
    return lo, hi


def parse_color_spec(
    cs: ColorSpec,
    n: int,
) -> NDArray: # uint8[n, 3]
    try:
        color = parse_color(cs) # type: ignore
        if color is None:
            return np.full((n, 3), 255, dtype=np.uint8)
        else:
            return np.full((n, 3), color, dtype=np.uint8)
    except ValueError:
        return np.asarray(cs, dtype=np.uint8)


def parse_series(
    series: Series, # Series<n>
) -> tuple[
    NDArray,        # number[n]
    NDArray,        # number[n]
    NDArray,        # uint8[n,3]
]:
    match series:
        case axis() as a:
            xs = a.xs
            ys = a.ys
            cs = parse_color_spec(None, a.n)
        case (axis() as a, cs_):
            xs = a.xs
            ys = a.ys
            cs = parse_color_spec(cast(ColorSpec, cs_), a.n)
        case np.ndarray(shape=(n, 2)) as a:
            xs = a[:, 0]
            ys = a[:, 1]
            cs = parse_color_spec(None, n)
        case (np.ndarray(shape=(n, 2)) as a, cs_):
            xs = a[:, 0]
            ys = a[:, 1]
            cs = parse_color_spec(cast(ColorSpec, cs_), n)
        case (xs_, ys_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            n, = xs.shape
            cs = parse_color_spec(None, n)
        case (xs_, ys_, cs_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            n, = xs.shape
            cs = parse_color_spec(cast(ColorSpec, cs_), n)
        case _:
            raise TypeError(f"Invalid Series {series!r}")
    return xs, ys, cs

            
def parse_multiple_series(
    *seriess: Series,
) -> tuple[
    NDArray,    # number[N]
    NDArray,    # number[N]
    NDArray,    # uint8[N,3]
]:
    xss, yss, css = zip(*map(parse_series, seriess))
    return (
        np.concatenate(xss),
        np.concatenate(yss),
        np.concatenate(css),
    )

    
def parse_series3(
    series: Series3, # Series3<n>
) -> tuple[
    NDArray,        # number[n]
    NDArray,        # number[n]
    NDArray,        # number[n]
    NDArray,        # uint8[n,3]
]:
    match series:
        case axis() as a:
            xs = a.xs
            ys = a.ys
            zs = a.zs
            cs = parse_color_spec(None, a.n)
        case (axis() as a, cs_):
            xs = a.xs
            ys = a.ys
            zs = a.zs
            cs = parse_color_spec(cast(ColorSpec, cs_), a.n)
        case np.ndarray(shape=(n, 3)) as a:
            xs = a[:, 0]
            ys = a[:, 1]
            zs = a[:, 2]
            cs = parse_color_spec(None, n)
        case (np.ndarray(shape=(n, 3)) as a, cs_):
            xs = a[:, 0]
            ys = a[:, 1]
            zs = a[:, 2]
            cs = parse_color_spec(cast(ColorSpec, cs_), n)
        case (xs_, ys_, zs_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            zs = np.asarray(zs_)
            n, = xs.shape
            cs = parse_color_spec(None, n)
        case (xs_, ys_, zs_, cs_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            zs = np.asarray(zs_)
            n, = xs.shape
            cs = parse_color_spec(cast(ColorSpec, cs_), n)
        case _:
            raise TypeError(f"Invalid Series3 {series!r}")
    return xs, ys, zs, cs


def parse_multiple_series3(
    *seriess: Series3,
) -> tuple[
    NDArray,    # number[N]
    NDArray,    # number[N]
    NDArray,    # number[N]
    NDArray,    # uint8[N,3]
]:
    xss, yss, zss, css = zip(*map(parse_series3, seriess))
    return (
        np.concatenate(xss),
        np.concatenate(yss),
        np.concatenate(zss),
        np.concatenate(css),
    )

    
# # # 
# Special series


@dataclasses.dataclass(frozen=True)
class axis:
    a: number = 0.
    b: number = 1.
    n: int = 10
    
    @property
    def xs(self) -> NDArray:
        return np.zeros(self.n)

    @property
    def ys(self) -> NDArray:
        return np.zeros(self.n)
    
    @property
    def zs(self) -> NDArray:
        return np.zeros(self.n)


class xaxis(axis):
    @property
    def xs(self) -> NDArray:
        return np.linspace(self.a, self.b, self.n)


class yaxis(axis):
    @property
    def ys(self) -> NDArray:
        return np.linspace(self.a, self.b, self.n)


class zaxis(axis):
    @property
    def zs(self) -> NDArray:
        return np.linspace(self.a, self.b, self.n)


# # # 
# 3D projection


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
    n, _3 = xyz.shape

    # compute view matrix
    V_z = camera_target - camera_position
    V_z /= np.linalg.norm(V_z)
    V_x = np.cross(V_z, scene_up)
    V_x /= np.linalg.norm(V_x)
    V_y = np.cross(V_x, V_z)
    V = np.array([V_x, V_y, V_z]).T

    # transform points to camera coordinate system
    xyz_ = (xyz - camera_position) @ V
    
    # mask for valid points
    valid = xyz_[:, 2] > 0.
    
    # perspective projection
    xy = np.zeros((n, 2))
    np.divide(
        xyz_[:, :2],
        xyz_[:, 2, np.newaxis],
        out=xy,
        where=valid[:, np.newaxis],
    )

    # scale fov to within [-1,1]^2
    focal_length = 1 / np.tan(np.radians(fov_degrees) / 2)
    xy *= focal_length

    return xy, valid


