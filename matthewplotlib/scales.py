"""
How values are measured against an interval. A scale carries the interval of
values a plot's visual channel covers---the colours of a heatmap, the lengths
of a bar chart---and answers how far along it each value lies. The named
scales space values nonlinearly within the interval; a plain `(lo, hi)` pair
means a linear `scale` wherever one is accepted.

* `scale`: values spaced linearly, and the base class the others extend.
* `logscale`: values spaced logarithmically, for data that ranges over
  orders of magnitude.
* `symlogscale`: logarithmic away from zero and linear near it, for data
  that spans zero.
* `powscale`: values spaced by a power, such as a square root.

A custom spacing is a subclass of `scale` overriding `transform` and
`inverse`.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from collections.abc import Iterator
from numpy.typing import ArrayLike, NDArray
from matthewplotlib.data import number


# # #
# THE SCALES


@dataclasses.dataclass(frozen=True)
class scale:
    """
    The interval of values a plot's visual channel covers, spaced linearly.

    A scale is a callable from values to the range 0.0 to 1.0, the way a
    colormap is a callable from that range to colours: calling it answers how
    far along the interval each value lies, saturating at the ends. The named
    subclasses space values nonlinearly within the interval; this base class
    is the linear case, and the one a plain `(lo, hi)` pair stands for
    wherever a scale is accepted.

    Inputs:

    * lo : optional number.
        The value at the bottom of the scale. Left out, a plot completes it
        from its data.
    * hi : optional number.
        The value at the top of the scale. Left out, a plot completes it from
        its data. Given below `lo`, the scale turns around: values at `lo`
        come out at 1.0 and values at `hi` at 0.0.

    A custom spacing is a subclass overriding two methods:

    * `transform`, a vectorised function applied to the values, strictly
      monotonic on any interval the scale will be given.
    * `inverse`, the function undoing it. Nothing that measures values
      against the scale consults it, so a subclass that is never asked for
      it back may leave it unwritten.

    The affine step from transformed values onto the range 0.0 to 1.0 is the
    scale's own, so a transform is free of bookkeeping---and any two
    transforms that differ by an affine map draw the identical picture, which
    is why `logscale` has no base to choose.
    """
    lo: number | None = None
    hi: number | None = None

    def __post_init__(self):
        for name in ("lo", "hi"):
            given = getattr(self, name)
            if given is None:
                continue
            value = float(given)
            if not math.isfinite(value):
                raise ValueError(
                    f"{type(self).__name__} endpoints must be finite, not "
                    f"{name}={value:g}"
                )
            object.__setattr__(self, name, value)
            self._check_endpoint(value)
        self._check_parameters()

    def transform(
        self,
        values: ArrayLike,  # number[...]
    ) -> NDArray:           # -> float[...]
        """
        The spacing of the scale, as the function applied to values before
        they are measured linearly. The identity: this scale is the linear
        case.
        """
        return np.asarray(values, dtype=float)

    def inverse(
        self,
        values: ArrayLike,  # float[...]
    ) -> NDArray:           # -> float[...]
        """
        The function undoing `transform`, taking transformed values back to
        the values they came from.
        """
        return np.asarray(values, dtype=float)

    def __call__(
        self,
        values: ArrayLike,  # number[...]
    ) -> NDArray:           # -> float[...]
        """
        How far along the interval each value lies, saturating at its ends.

        Runs 0.0 to 1.0 from `lo` to `hi`, so that an interval given
        descending turns the scale around. Values beyond the interval are
        clipped to it before the transform sees them, so they saturate at
        the nearest end. Where the interval covers nothing---which only an
        inferred one does, over values that are all the same---everything
        comes out at the bottom, and so does any value that is not a number.
        """
        lo, hi = self.lo, self.hi
        if lo is None or hi is None:
            raise ValueError(
                f"{self!r} has a missing endpoint, so there is nothing to "
                "measure values against; give both ends, or give the scale "
                "to a plot, which completes it from the data"
            )
        array = np.asarray(values, dtype=float)
        if lo == hi:
            return np.zeros(array.shape, dtype=float)
        tlo = float(self.transform(lo))
        thi = float(self.transform(hi))
        if not (math.isfinite(tlo) and math.isfinite(thi)) or tlo == thi:
            raise ValueError(
                f"{self!r} sends its own endpoints to {tlo:g} and {thi:g}, "
                "leaving values no scale to lie along; the transform must be "
                "finite and strictly monotonic over the interval"
            )
        clipped = np.clip(array, min(lo, hi), max(lo, hi))
        scaled = (np.asarray(self.transform(clipped)) - tlo) / (thi - tlo)
        scaled = np.clip(scaled, 0.0, 1.0)
        return np.where(np.isnan(scaled), 0.0, scaled)

    @property
    def interval(self) -> tuple[float, float]:
        """
        The two endpoints as a plain pair, for a scale that has both.
        """
        if self.lo is None or self.hi is None:
            raise ValueError(f"{self!r} has a missing endpoint")
        return (float(self.lo), float(self.hi))

    def __iter__(self) -> Iterator[float | None]:
        """
        A scale unpacks as its interval: `lo, hi = scale(0, 5)`.
        """
        yield None if self.lo is None else float(self.lo)
        yield None if self.hi is None else float(self.hi)

    def __repr__(self) -> str:
        ends = [
            "None" if end is None else format(float(end), "g")
            for end in (self.lo, self.hi)
        ]
        extras = [
            f"{field.name}={float(getattr(self, field.name)):g}"
            for field in dataclasses.fields(self)[2:]
        ]
        return f"{type(self).__name__}({', '.join(ends + extras)})"

    def _check_endpoint(self, value: float) -> None:
        """
        Refuse an endpoint outside the values the spacing is defined over.
        Nothing to check for the linear case; subclasses raise a `ValueError`
        naming their limit.
        """

    def _check_parameters(self) -> None:
        """
        Validate any fields beyond the endpoints. Nothing to check for the
        linear case; subclasses raise a `ValueError` naming the field.
        """

    def _fallback_interval(self) -> tuple[float, float]:
        """
        The interval standing in when there are no finite values to infer one
        from. The unit interval, except where the spacing is not defined over
        it.
        """
        return (0.0, 1.0)


@dataclasses.dataclass(frozen=True, repr=False)
class logscale(scale):
    """
    An interval of values spaced logarithmically.

    An equal step along the scale multiplies the value by an equal factor, so
    data ranging over orders of magnitude keeps its low decades visible where
    a linear scale would crush them into the bottom. The interval must sit
    above zero; data reaching zero or below wants a `symlogscale`, or an
    explicit `lo` cutting the interval off above zero.

    The logarithm's base makes no difference---two bases differ by a constant
    factor, which the mapping onto the range 0.0 to 1.0 cancels---so there is
    no base to choose.
    """

    def transform(self, values: ArrayLike) -> NDArray:
        return np.log(np.asarray(values, dtype=float))

    def inverse(self, values: ArrayLike) -> NDArray:
        return np.exp(np.asarray(values, dtype=float))

    def _check_endpoint(self, value: float) -> None:
        if value <= 0:
            raise ValueError(
                "logscale covers values above zero, and this interval "
                f"reaches {value:g}"
            )

    def _fallback_interval(self) -> tuple[float, float]:
        return (1.0, 10.0)


@dataclasses.dataclass(frozen=True, repr=False)
class symlogscale(scale):
    """
    An interval of values spaced logarithmically away from zero and linearly
    near it, so that it can span zero, which a `logscale` cannot.

    The spacing is `arcsinh(value / linear_width)`: smooth everywhere, with
    the changeover from linear to logarithmic happening gradually around
    `linear_width` rather than at a corner.

    Inputs:

    * lo : optional number.
        The value at the bottom of the scale, as for `scale`.
    * hi : optional number.
        The value at the top of the scale, as for `scale`.
    * linear_width : keyword number (default: 1.0).
        The size of the values that are spaced roughly linearly. Values well
        below it sit on the linear stretch around zero; values well above it
        are spaced logarithmically. Unlike a logarithm's base, this genuinely
        changes the picture: it decides how much of the scale the small
        values get.
    """
    linear_width: number = dataclasses.field(default=1.0, kw_only=True)

    def transform(self, values: ArrayLike) -> NDArray:
        return np.arcsinh(np.asarray(values, dtype=float) / self.linear_width)

    def inverse(self, values: ArrayLike) -> NDArray:
        return np.sinh(np.asarray(values, dtype=float)) * self.linear_width

    def _check_parameters(self) -> None:
        value = float(self.linear_width)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"symlogscale needs a positive linear_width, not {value:g}"
            )
        object.__setattr__(self, "linear_width", value)


@dataclasses.dataclass(frozen=True, repr=False)
class powscale(scale):
    """
    An interval of values spaced by a power.

    An exponent below one stretches the low end of the interval and squeezes
    the high end---`powscale(exponent=0.5)` is a square-root scale---and an
    exponent above one does the opposite. The interval must not reach below
    zero, where a fractional power is undefined.

    Inputs:

    * lo : optional number.
        The value at the bottom of the scale, as for `scale`.
    * hi : optional number.
        The value at the top of the scale, as for `scale`.
    * exponent : keyword number.
        The power the values are raised to. Must be positive; an exponent of
        one is a linear `scale` wearing a longer name.
    """
    exponent: number = dataclasses.field(kw_only=True)

    def transform(self, values: ArrayLike) -> NDArray:
        return np.power(np.asarray(values, dtype=float), self.exponent)

    def inverse(self, values: ArrayLike) -> NDArray:
        return np.power(np.asarray(values, dtype=float), 1.0 / self.exponent)

    def _check_endpoint(self, value: float) -> None:
        if value < 0:
            raise ValueError(
                "powscale covers values from zero up, and this interval "
                f"reaches {value:g}"
            )

    def _check_parameters(self) -> None:
        value = float(self.exponent)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"powscale needs a positive exponent, not {value:g}"
            )
        object.__setattr__(self, "exponent", value)


# # #
# COMPLETING A SCALE AGAINST DATA


def _resolve_vrange(
    vrange: tuple[number, number] | scale | None,
    values: NDArray,
    what: str,
    from_zero: bool = False,
    allow_flat: bool = True,
) -> scale:
    """
    The scale a plot measures its values against, completed and validated.

    A pair becomes a linear `scale` and `None` an empty one. A missing
    endpoint is inferred from the finite values there are: the lowest of them
    for `lo`---or zero if `from_zero`, for a scale whose bottom end is a
    baseline rather than the smallest value measured---and the highest for
    `hi`, so that the scale spans the data. With no finite values at all, a
    given endpoint stands in for the missing one, and a scale missing both
    falls back to an interval of its own choosing.

    An interval covering nothing is an error where the caller wrote it, since
    there is no reading of it to act on. Inferred, over values that are all
    the same, it is returned as it stands unless `allow_flat` is false, for a
    plot that draws positions along the interval and so has nowhere to put
    them. `what` names the caller in the errors.
    """
    if vrange is None:
        given = scale()
    elif isinstance(vrange, scale):
        given = vrange
    else:
        given = scale(vrange[0], vrange[1])

    if given.lo is not None and given.hi is not None:
        if given.lo == given.hi:
            raise ValueError(f"{what} vrange covers no interval: {vrange!r}")
        return given

    lo, hi = given.lo, given.hi
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        if lo is None and hi is None:
            lo, hi = given._fallback_interval()
        else:
            lo = hi = lo if lo is not None else hi
    else:
        if lo is None:
            lo = 0.0 if from_zero else float(finite.min())
        if hi is None:
            hi = float(finite.max())
    if lo == hi and not allow_flat:
        raise ValueError(
            f"every value in the {what} sits at {lo}; give a vrange "
            "spanning an interval to plot them in"
        )
    try:
        return dataclasses.replace(given, lo=lo, hi=hi)
    except ValueError as e:
        raise ValueError(
            f"{what}: {e}; the missing endpoint was inferred from the data, "
            "so give one explicitly"
        ) from None


def _resolve_linear_vrange(
    vrange: tuple[number, number] | scale | None,
    values: NDArray,
    what: str,
    allow_flat: bool = True,
) -> scale:
    """
    The scale for a plot that lays values out along a coordinate axis.

    Such a plot hands its interval to the window that its axes are labelled
    from, and a window places coordinates linearly, so a scale that moves the
    marks would part them from their labels. A plain interval is accepted and
    completed as usual; a scale that spaces values nonlinearly is refused.
    """
    if isinstance(vrange, scale) and type(vrange) is not scale:
        raise TypeError(
            f"{what} lays values out along a coordinate axis, which "
            f"{vrange!r} does not reach; give a plain (lo, hi) interval"
        )
    return _resolve_vrange(vrange, values, what, allow_flat=allow_flat)
