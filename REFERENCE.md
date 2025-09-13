---
title: Matthew's plotting library (`matthewplotlib`)
date: Version 0.1.1
---

## module matthewplotlib

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/__init__.py)]

Matthew's plotting library (matthewplotlib).

A python plotting library that isn't painful.

See https://github.com/matomatical/matthewplotlib

## module matthewplotlib.colormaps

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py)]

### function matthewplotlib.colormaps.reds

#### reds(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L30)]

Red colormap. Simply embeds greyscale value into red channel.

### function matthewplotlib.colormaps.greens

#### greens(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L42)]

Green colormap. Simply embeds greyscale value into green channel.

### function matthewplotlib.colormaps.blues

#### blues(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L54)]

Blue colormap. Simply embeds greyscale value into blue channel.

### function matthewplotlib.colormaps.yellows

#### yellows(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L66)]

Yellow colormap. Simply embeds greyscale value into red and green channels.

### function matthewplotlib.colormaps.magentas

#### magentas(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L79)]

Magenta colormap. Simply embeds greyscale value into red and blue
channels.

### function matthewplotlib.colormaps.cyans

#### cyans(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L93)]

Cyan colormap. Simply embeds greyscale value into green and blue
channels.

### function matthewplotlib.colormaps.cyber

#### cyber(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L111)]

Cyberpunk colormap. Uses greyscale value to interpolate between cyan and
magenta.

### function matthewplotlib.colormaps.rainbow

#### rainbow(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L126)]

Rainbow colormap. Effectively embeds greyscale values as hue in HSV color
space.

### function matthewplotlib.colormaps.magma

#### magma(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L174)]

Magma colormap by Nathaniel J. Smith and Stefan van der Walt (see
https://bids.github.io/colormap/).

Discretised to 256 8-bit colours.

### function matthewplotlib.colormaps.inferno

#### inferno(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L252)]

Inferno colormap by Nathaniel J. Smith and Stefan van der Walt (see
https://bids.github.io/colormap/).

Discretised to 256 8-bit colours.

### function matthewplotlib.colormaps.plasma

#### plasma(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L330)]

Plasma colormap by Nathaniel J. Smith and Stefan van der Walt (see
https://bids.github.io/colormap/).

Discretised to 256 8-bit colours.

### function matthewplotlib.colormaps.viridis

#### viridis(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L408)]

Viridis colormap by Nathaniel J. Smith, Stefan van der Walt, and Eric
Firing (see https://bids.github.io/colormap/).

Discretised to 256 8-bit colours.

### function matthewplotlib.colormaps.sweetie16

#### sweetie16(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L490)]

Sweetie-16 colour palette by GrafxKid (see
https://lospec.com/palette-list/sweetie-16).

Input should be an array of indices in the range [0,15].

### function matthewplotlib.colormaps.pico8

#### pico8(x: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colormaps.py#L508)]

PICO-8 colour palette (see https://pico-8.fandom.com/wiki/Palette).

Input should be an array of indices in the range [0,15].

## module matthewplotlib.colors

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colors.py)]

### class matthewplotlib.colors.Color

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colors.py#L17)]

An RGB color triple.

### method matthewplotlib.colors.Color.\_\_iter\_\_

#### \_\_iter\_\_() -> Iterator[int]

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colors.py#L26)]

### method matthewplotlib.colors.Color.parse

#### parse(color: ColorLike) -> Color | None

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/colors.py#L31)]

Accept and standardise RGB triples in various formats.

## module matthewplotlib.core

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/core.py)]

### class matthewplotlib.core.Char

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/core.py#L16)]

A single possibly-coloured character. Plots are assembled from characters
like these.

### method matthewplotlib.core.Char.\_\_bool\_\_

#### \_\_bool\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/core.py#L26)]

True if the character has visible content, false if it is blank.

### method matthewplotlib.core.Char.to\_ansi\_str

#### to\_ansi\_str(self: Self) -> str

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/core.py#L33)]

If necessary, wrap a Char in ANSI control codes that switch the color into
the given fg and bg colors; plus a control code to switch back to default
mode.

### method matthewplotlib.core.Char.to\_rgba\_array

#### to\_rgba\_array(self: Self) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/core.py#L50)]

Convert a Char to a small RGBA image patch, with the specified foreground
color (or white) and background color (or a transparent background).

### function matthewplotlib.core.braille\_encode

#### braille\_encode(a: ArrayLike) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/core.py#L92)]

Turns a HxW array of booleans into a (H//4)x(W//2) array of braille
binary codes.

Inputs:

* 

Here is a visual explanation of this function:

```
Start with an array with height divisible by 4, width divisible by 2:
    ____
   [1  0] 0  1  0  1  1  1  1  0  1  0  0  0  0  1  0  0  0  0  0  1  1  0
   [1  0] 0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
 .-[1  0] 0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
 | [1__0] 0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
 |  1  1  1  1  0  1  1  1  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
 |  1  0  0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
 |  1  0  0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
 |  1  0  0  1  0  1  1  1  1  0  1  1  1  1  0  1  1  1  1  0  0  1  1  0
 |
 | take each 4x2 subarray and ...
 |                                                               braille
 | identify the 4x2 bits with the                                unicode
 | eight numbered braille dots:                                  start pt
 |                                                               |
 |  (dot 1) 1 0 (dot 4)     convert to                           v
 `> (dot 2) 1 0 (dot 5) -----------------> 0 b 0 1 0 0 0 1 1 1 + 0x2800 -.
    (dot 3) 1 0 (dot 6)    braille code        | | | | | | | |           |
    (dot 7) 1 0 (dot 8)                    dot 8 7 6 5 4 3 2 1           |
                                                                         |
  convert the braille code to a unicode character and collate into array |
 .-----------------------------------------------------------------------'
 |  '''
 `->⡇⢸⢸⠉⠁⡇⠀⢸⠀⠀⡎⢱  (Note: this function returns codepoints, use `chr()`
    ⡏⢹⢸⣉⡁⣇⣀⢸⣀⡀⢇⡸  to convert these into braille characters for printing.)
    '''
```

## module matthewplotlib.plots

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py)]

### class matthewplotlib.plots.plot

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L21)]

Abstract base class for all plot objects.

A plot is essentially a 2D grid of `Char` objects. This class provides the
core functionality for rendering and composing plots. Not typically
instantiated directly, but it's useful to know its properties and methods.

Properties:

* height : int
    The height of the plot in character lines.
* width : int
    The width of the plot in character columns.

Methods:

* renderstr() -> str
    Returns a string representation of the plot with ANSI color codes,
    ready to be printed to a compatible terminal.
* clearstr() -> str
    Returns control characters that will clear the plot from the
    terminal after it has been printed.
* saveimg(filename: str)
    Renders the plot to an image file (e.g., "plot.png") using a
    pixel font.

Operators:

* `str(plot)`: Shortcut for `plot.renderstr()`. This means you can render
   the plot just by calling `print(plot)`.
* `~plot`: Shortcut for `plot.clearstr()`. Useful for animations.
* `plot1 | plot2`: Horizontally stacks plots (see `hstack`).
* `plot1 ^ plot2`: Vertically stacks plots (see `vstack`).
* `plot1 & plot2`: Overlays plots (see `dstack`).

### method matthewplotlib.plots.plot.height

#### height() -> int

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L62)]

Number of character rows in the plot.

### method matthewplotlib.plots.plot.width

#### width() -> int

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L70)]

Number of character columns in the plot.

### method matthewplotlib.plots.plot.renderstr

#### renderstr() -> str

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L77)]

Convert the plot into a string for printing to the terminal.

Note: plot.renderstr() is equivalent to str(plot).

### method matthewplotlib.plots.plot.clearstr

#### clearstr(self: Self) -> str

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L86)]

Convert the plot into a string that, if printed immediately after
plot.renderstr(), will clear that plot from the terminal.

### method matthewplotlib.plots.plot.renderimg

#### renderimg(, scale\_factor: int) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L94)]

Convert the plot into an RGBA array for rendering with Pillow.

### method matthewplotlib.plots.plot.saveimg

#### saveimg(, filename: str, scale\_factor: int)

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L120)]

Render the plot as an RGBA image and save it as a PNG file at the path
`filename`.

### method matthewplotlib.plots.plot.\_\_str\_\_

#### \_\_str\_\_() -> str

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L134)]

Shortcut for the string for printing the plot.

### method matthewplotlib.plots.plot.\_\_invert\_\_

#### \_\_invert\_\_(self: Self) -> str

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L141)]

Shortcut for the string for clearing the plot.

### method matthewplotlib.plots.plot.\_\_or\_\_

#### \_\_or\_\_(self: Self, other: Self) -> 'hstack'

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L148)]

Shortcut for horizontally stacking plots:

plot1 | plot2 = hstack(plot1, plot2).

### method matthewplotlib.plots.plot.\_\_xor\_\_

#### \_\_xor\_\_(self: Self, other: Self) -> 'vstack'

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L157)]

Shortcut for vertically stacking plots:

plot1 ^ plot2 = vstack(plot1, plot2).

### method matthewplotlib.plots.plot.\_\_and\_\_

#### \_\_and\_\_(self: Self, other: Self) -> 'dstack'

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L166)]

Shortcut for depth-stacking plots:

plot1 & plot2 = dstack(plot1, plot2).

### class matthewplotlib.plots.image

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L179)]

[Inherits from plot]

Render a small image using a grid of unicode half-block characters.

Represents an image by mapping pairs of vertically adjacent pixels to the
foreground and background colors of a single character cell (this
effectively doubles the vertical resolution in the terminal).

Inputs:

* im : float[h,w,3] | int[h,w,3] | float[h,w] | int[h,w] | ArrayLike
    The image data. It can be in any of the following formats:
    * `float[h,w,3]`: A 2D array of RGB triples of floats in range [0,1].
    * `int[h,w,3]`: A 2D array of RGB triples of ints in range [0,255].
    * `float[h,w]`: A 2D array of scalars in the range [0,1]. If no
      colormap is provided, values are treated as greyscale (uniform
      colorisation). If a continuous colormap is provided, values are
      mapped to RGB values.
    * `int[h,w]`: A 2D array of scalars. If no colormap is provided,
      values should be in the range [0,255], they are treated as greyscale
      (uniform colorisation). If a discrete colormap is provided, values
      should be in range as indices for the colormap, they will be mapped
      to RGB triples as such.
      
* colormap : optional ColorMap
    Function mapping (batches of) scalars to (batches of) RGB triples.
    Examples are provided by this library, such as:
    * continuous colormaps like `viridis : float[...] -> uint8[...,3]`, and
    * discrete colormaps like `pico8 : int[...] -> uint8[...,3]`.
    If `im` has no RGB dimension, it is transformed to a grid of RGB
    triples using one of these colormaps.

### method matthewplotlib.plots.image.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L251)]

### class matthewplotlib.plots.fimage

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L255)]

[Inherits from image]

Heatmap representing the image of a 2d function over a square. Inputs:

* F : float[batch, 2] -> float[batch]
    The (vectorised) function to plot. The input should be a batch of
    (x, y) vectors. The output should be a batch of scalars f(x, y).
* xrange : (float, float)
    Lower and upper bounds on the x values to pass into the function.
* yrange : (float, float)
    Lower and upper bounds on the y values to pass into the function.
* width : int
    The number of grid squares along the x axis. This will also become the
    width of the plot.
* height : int
    The number of grid squares along the y axis. This will become double
    the height of the plot in lines (since the result is an image plot with
    two pixels per line).
* zrange : optional (float, float)
    Expected lower and upper bounds on the f(x, y) values. Used for
    determining the bounds of the colour scale. By default, the minimum and
    maximum output over the grid are used.
* colormap : optional colormap (e.g. mp.viridis)
    By default, the output will be in greyscale, with black corresponding
    to zrange[0] and white corresponding to zrange[1]. You can choose a
    different colormap (e.g. mp.reds, mp.viridis, etc.) here.
* endpoints : bool (default: False)
    If true, endpoints are included from the linspaced inputs, and so the
    grid elements in each corner will represent the different combinations
    of xrange/yrange.
    If false (default), the endpoints are excluded, so the lower bounds are
    met but the upper bounds are not, meaning each grid square color shows
    the value of the function precisely at its lower left corner.

### method matthewplotlib.plots.fimage.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L330)]

### class matthewplotlib.plots.scatter

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L338)]

[Inherits from plot]

Render a scatterplot using a grid of braille unicode characters.

Each character cell in the plot corresponds to a 2x4 grid of sub-pixels,
represented by braille dots.

Inputs:

* data : float[n, 2]
    An array of n 2D points to plot. Each row is an (x, y) coordinate.
* height : int (default: 10)
    The height of the plot in rows. The effective pixel height will be 4 *
    height.
* width : int (default: 30)
    The width of the plot in characters. The effective pixel width will be
    2 * width.
* yrange : optional (float, float)
    The y-axis limits `(ymin, ymax)`. If not provided, the limits are
    inferred from the min and max y-values in the data.
* xrange : optional (float, float)
    The x-axis limits `(xmin, xmax)`. If not provided, the limits are
    inferred from the min and max x-values in the data.
* color : optional ColorLike
    The color of the plotted points (see `Color.parse`). Defaults to the
    terminal's default foreground color.
* check_bounds : bool (default: False)
    If True, raises a `ValueError` if any data points fall outside the
    specified `xrange` or `yrange`.

### method matthewplotlib.plots.scatter.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L434)]

### class matthewplotlib.plots.hilbert

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L443)]

[Inherits from plot]

Visualize a 1D boolean array along a 2D Hilbert curve.

Maps a 1D sequence of data points to a 2D grid using a space-filling
Hilbert curve, which helps preserve locality. The curve is rendered using
braille unicode characters for increased resolution.

Inputs:

* data : bool[N]
    A 1D array of booleans. The length `N` determines the order of the
    Hilbert curve required to fit all points. True values are rendered as
    dots, and False values are rendered as blank spaces.
* dotcolor : optional ColorLike
    The foreground color used for dots (points along the curve where `data`
    is `True`). Defaults to the terminal's default foreground color.
* bgcolor : optional ColorLike
    The background color for the entire path of the Hilbert curve (points
    along the curve where `data` is `False`, plus possibly some extra
    points if the curve does not exactly fit the last character cell).
    Defaults to a transparent background.
* nullcolor : optional ColorLike
    The background color for the grid area not occupied by the curve. This
    is relevant for non-square-power-of-2 data lengths. Defaults to a
    transparent background.

### method matthewplotlib.plots.hilbert.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L520)]

### class matthewplotlib.plots.text

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L528)]

[Inherits from plot]

    A plot object containing one or more lines of text.

    This class wraps a string in the plot interface, allowing it to be
    composed with other plot objects. It handles multi-line strings by
    splitting them at newline characters.

    Inputs:

    * text : str
        The text to be displayed. Newline characters (`
`) will create
        separate lines in the plot.
    * color : optional ColorLike
        The foreground color of the text. Defaults to the terminal's default
        foreground color.
    * bgcolor : optional ColorLike
        The background color for the text. Defaults to a transparent
        background.
    
    TODO:

    * Allow alignment and resizing.
    * Account for non-printable and wide characters.
    

### method matthewplotlib.plots.text.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L576)]

### class matthewplotlib.plots.progress

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L583)]

[Inherits from plot]

A single-line progress bar.

Displays a progress bar with a percentage label. The bar is rendered using
block element characters to show fractional progress with finer granularity
than a single character.

Inputs:

* progress : float
    The progress to display, as a float between 0.0 and 1.0. Values outside
    this range will be clipped.
* width : int (default: 40)
    The total width of the progress bar plot in character columns,
    including the label and brackets.
* color : optional ColorLike
    The color of the filled portion of the progress bar. Defaults to the
    terminal's default foreground color.

### method matthewplotlib.plots.progress.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L637)]

### class matthewplotlib.plots.blank

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L645)]

[Inherits from plot]

Creates a rectangular plot composed entirely of blank space.

Useful for adding padding or aligning items in a complex layout.

Inputs:

* height : optional int

  The height of the blank area in character rows. Default 1.

* width : optional int

  The width of the blank area in character columns. Default 1.

### method matthewplotlib.plots.blank.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L670)]

### class matthewplotlib.plots.hstack

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L674)]

[Inherits from plot]

Horizontally arrange one or more plots side-by-side.

If the plots have different heights, the shorter plots will be padded with
blank space at the bottom to match the height of the tallest plot.

Inputs:

* *plots : plot
    A sequence of plot objects to be horizontally stacked.

### method matthewplotlib.plots.hstack.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L702)]

### class matthewplotlib.plots.vstack

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L709)]

[Inherits from plot]

Vertically arrange one or more plots, one above the other.

If the plots have different widths, the narrower plots will be padded with
blank space on the right to match the width of the widest plot.

Inputs:

* *plots : plot
    A sequence of plot objects to be vertically stacked.

### method matthewplotlib.plots.vstack.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L735)]

### class matthewplotlib.plots.dstack

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L742)]

[Inherits from plot]

Overlay one or more plots on top of each other.

The plots are layered in the order they are given, with later plots in the
sequence drawn on top of earlier ones. The final size of the plot is
determined by the maximum width and height among all input plots. Non-blank
characters from upper layers will obscure characters from lower layers.

Inputs:

* *plots : plot
    A sequence of plot objects to be overlaid.

### method matthewplotlib.plots.dstack.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L772)]

### class matthewplotlib.plots.wrap

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L779)]

[Inherits from plot]

Arrange a sequence of plots into a grid.

The plots are arranged from left to right, wrapping to a new line when
the specified number of columns is reached. All cells in the grid are
padded to the size of the largest plot in the sequence.

Inputs:

* *plots : plot
    A sequence of plot objects to be arranged in a grid.
* cols : optional int
    The number of columns in the grid. If not provided, it is automatically
    determined based on the terminal width and the width of the largest
    plot.

### method matthewplotlib.plots.wrap.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L831)]

### class matthewplotlib.plots.border

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L838)]

[Inherits from plot]

Add a border around a plot using box-drawing characters.

Inputs:

* plot : plot
    The plot object to be enclosed by the border.
* style : optional Style (default: Style.ROUND)
    The style of the border. Predefined styles are available in
    `border.Style`.
* color : optional ColorLike
    The color of the border characters. Defaults to the terminal's
    default foreground color.

### class matthewplotlib.plots.border.Style

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L853)]

[Inherits from str, enum.Enum]

A string enum defining preset styles for the `border` plot.

Each style is a string of six characters representing the border
elements in the following order: horizontal, vertical, top-left,
top-right, bottom-left, and bottom-right.

Available Styles:

* `LIGHT`: A standard, single-line border.
* `HEAVY`: A thicker, bold border.
* `DOUBLE`: A double-line border.
* `BLANK`: An invisible border, useful for adding padding around a
  plot while maintaining layout alignment.
* `ROUND`: A single-line border with rounded corners.
* `BUMPER`: A single-line border with corners made of blocks.

Demo:

```
┌──────┐ ┏━━━━━━┓ ╔══════╗         ╭──────╮ ▛──────▜
│LIGHT │ ┃HEAVY ┃ ║DOUBLE║  BLANK  │ROUND │ │BUMPER│
└──────┘ ┗━━━━━━┛ ╚══════╝         ╰──────╯ ▙──────▟
```

### method matthewplotlib.plots.border.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L922)]

### class matthewplotlib.plots.center

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L926)]

[Inherits from plot]

Pad a plot with blank space to center it within a larger area.

If the specified `height` or `width` is smaller than the plot's dimensions,
the larger dimension is used, effectively preventing the plot from being
cropped.

Inputs:

* plot : plot
    The plot object to be centered.
* height : optional int
    The target height of the new padded plot. If not provided, it defaults
    to the original plot's height (no vertical padding).
* width : optional int
    The target width of the new padded plot. If not provided, it defaults
    to the original plot's width (no horizontal padding).

### method matthewplotlib.plots.center.\_\_repr\_\_

#### \_\_repr\_\_()

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/plots.py#L970)]

## module matthewplotlib.unscii16

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/unscii16.py)]

Unscii 2.0, bitmap unicode font created by Viznut (http://viznut.fi/unscii/).

This module is a port of all non-wide characters from unscii-16 (16px by 8px).

### function matthewplotlib.unscii16.bitmap

#### bitmap(char: str) -> np.ndarray

[[source](https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/unscii16.py#L11)]

Look up the bitmap for a single character.

Inputs:

* char: str (len 1)
    A single-character string.

Returns:

* bits: bool[16, 8]
    A boolean array representing the character's bitmap.

