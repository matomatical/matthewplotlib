Examples
========

See the [examples/](https://github.com/matomatical/matthewplotlib/tree/main/examples)
folder for source code.

<table>
<thead>
  <th width="50%">Image</th>
  <th width="50%">Example</th>
</thead>
<tbody>
  <tr>
    <td align="center">
      <img src="images/boing.gif">
    </td>
    <td>
      <p><strong>Boing</strong></p>
      <p>The Amiga Boing Ball, 1984, animated the way the Amiga animated it: the ball is drawn once as colour <em>indices</em> and spins because the palette is rewritten between frames, while the bounce moves a rigid sprite to whole-pixel positions. Two lookups, so the whole animation is one array with a time axis &mdash; built with <code>mp.animation</code> and played as a value rather than a loop. Run it and the palette driving the spin is shown cycling underneath; the gif here is the ball on its own.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/boing.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/calendar_heatmap.png">
    </td>
    <td>
      <p><strong>Calendar heatmap</strong></p>
      <p>Calendar heatmap of daily maximum temperatures in Oxford, 2025.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/calendar_heatmap.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/colormaps.png">
    </td>
    <td>
      <p><strong>Colormaps</strong></p>
      <p>Gallery of all available continuous and discrete colormaps.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/colormaps.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/dashboard.gif">
    </td>
    <td>
      <p><strong>Dashboard</strong></p>
      <p>Live system monitoring dashboard showing CPU and memory usage.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/dashboard.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/demo.png">
    </td>
    <td>
      <p><strong>Demo</strong></p>
      <p>General demonstration combining images, borders, and scatter plots.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/demo.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/functions.png">
    </td>
    <td>
      <p><strong>Functions</strong></p>
      <p>Mathematical function visualisation with scatter and function2.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/functions.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/life.gif">
    </td>
    <td>
      <p><strong>Game of Life</strong></p>
      <p>Conway's Game of Life, coloured by what just happened to each cell:
      newborn, stable, or recently dead. The panels underneath track the cell
      counts and, on the right, what each frame cost to write against what a
      full redraw would have cost &mdash; the two curves separate as the board
      settles, because a differential redraw costs in proportion to the cells
      that changed rather than the size of the board.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/life.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/hilbert_curve.png">
    </td>
    <td>
      <p><strong>Hilbert curve</strong></p>
      <p>Hilbert curve visualisation of binomial data.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/hilbert_curve.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/image.png">
    </td>
    <td>
      <p><strong>Image rendering</strong></p>
      <p>Image rendering with various colormaps.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/image.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/jointplot.png">
    </td>
    <td>
      <p><strong>Joint distribution</strong></p>
      <p>Joint distribution with marginal histograms, demonstrating plot composition with hstack and vstack.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/jointplot.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/lines.png">
    </td>
    <td>
      <p><strong>Line charts</strong></p>
      <p>Two loss curves, one measured every step and one every eighth of one with a stretch missing, drawn as a gap rather than as a straight line across the hole. Underneath, one spiral drawn four times with the pen set wider each time: the stroke is the curve widened by a disc, so it keeps its width around the tightest part of the turn and the joins fill in rather than notching.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/lines.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/lissajous.png">
    </td>
    <td>
      <p><strong>Lissajous curves</strong></p>
      <p>Brownian motion PCA visualisation with scatterplots and plot arrangement.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/lissajous.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/mandelbrot.gif">
    </td>
    <td>
      <p><strong>Mandelbrot</strong></p>
      <p>Animated Mandelbrot fractal zoom using function heatmaps and colormaps.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/mandelbrot.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/quickstart.png">
    </td>
    <td>
      <p><strong>Quickstart 1</strong></p>
      <p>Coloured cosine waves with phase offsets.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/quickstart1.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/quickstart2.gif">
    </td>
    <td>
      <p><strong>Quickstart 2</strong></p>
      <p>Animated cosine wave with shifting phase and amplitude, as a loop of <code>print(plot - prev)</code>.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/quickstart2.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/quickstart3.gif">
    </td>
    <td>
      <p><strong>Quickstart 3</strong></p>
      <p>The same animation with the loop handed to <code>mp.animate</code>, which keeps the previous frame and the frame clock.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/quickstart3.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/scatter.png">
    </td>
    <td>
      <p><strong>Scatter</strong></p>
      <p>Spiral scatter plot with viridis colormap.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/scatter.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/starburst.gif">
    </td>
    <td>
      <p><strong>Starburst</strong></p>
      <p>A rose of rays, turning, with the pen swelling from one dot to six and back. Twenty-four directions and three lengths, so the same pen has to draw a stub and a full radius; each ray runs from a dim version of its hue at the hub to the full hue at the tip, interpolated along the segment as it is drawn. The rays are separate strokes of one series, separated by gaps.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/starburst.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/teacher_student.gif">
    </td>
    <td>
      <p><strong>Teacher-student regression</strong></p>
      <p>Gradient descent on a simple teacher-student linear regression model.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/teacher_student.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/teapot.gif">
    </td>
    <td>
      <p><strong>Teapot</strong></p>
      <p>3D scatter plot with animated camera orbit.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/teapot.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Nothing to show here: a picture of this one running on someone
      else's terminal tells you nothing about yours.</em>
    </td>
    <td>
      <p><strong>Terminal test</strong></p>
      <p>Does your terminal render matthewplotlib correctly? Four stages &mdash; colour, redrawing in place, resizing, and a plot pushed against the right margin &mdash; exercising every escape sequence the library can emit, each saying what it should look like so you can judge it. Measures your terminal's width and draws to it. See the <a href="compatibility.html">compatibility page</a>.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/terminal_test.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/time_series_histogram.png">
    </td>
    <td>
      <p><strong>Time series histogram</strong></p>
      <p>Time series visualisation with stacked scatter, pooled scatter, and 2D histogram.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/time_series_histogram.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/voronoi.png">
    </td>
    <td>
      <p><strong>Voronoi diagram</strong></p>
      <p>Voronoi diagram using function heatmaps and scipy.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/voronoi.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/landscape.gif">
    </td>
    <td>
      <p><strong>Wireframe landscape</strong></p>
      <p>A landscape scrolling under a banded sun, with the terrain mesh and the sun both drawn by a single call to <code>mp.line3</code>: every wire is a separate stroke of one series, separated by gaps, and there is no surface or shading anywhere. Colour by depth is the only cue that a wire is far away. The scroll loops seamlessly, the terrain being a sum of sines whose periods divide the distance travelled over one loop.</p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/landscape.py">Source</a></p>
    </td>
  </tr>
</tbody>
</table>
