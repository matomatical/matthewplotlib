Examples
========

> [!NOTE] Do you have a cool, standalone matthewplotlib example?
> 
> I'd love to include it on this page! Please send me a link.

Contents:

* [Series Data Visualisation](#series-data-visualisation)
* [Surface Data Visualisation](#surface-data-visualisation)
* [Nonlinear Visualisation](#nonlinear-visualisation)
* [Bars and Columns](#bars-and-columns)
* [Media](#media)
* [Retro Animations](#retro-animations)
* [Simulations](#simulations)
* [Dashboards and UI](#dashboards-and-ui)
* [Utilities](#utilities)

See the [examples/](https://github.com/matomatical/matthewplotlib/tree/main/examples)
folder for source code.

Series Data Visualisation
-------------------------


<table>
<thead>
  <th width="50%">Image</th>
  <th width="50%">Example</th>
</thead>
<tbody>
  <tr>
    <td align="center">
      <img src="../images/lines.png" width="100%">
    </td>
    <td>
      <p><strong>Line charts</strong></p>
      <p>Line plot test, including series with missing values (NaN).</p>
      <p><em>By Claude Opus 5.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/lines.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/lissajous.png" width="100%">
    </td>
    <td>
      <p><strong>Lissajous curves</strong></p>
      <p>Brownian motion PCA visualisation with scatterplots and plot
      arrangement.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/lissajous.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/quickstart.png" width="100%">
    </td>
    <td>
      <p><strong>Quickstart 1</strong></p>
      <p>Coloured cosine waves with phase offsets.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/quickstart1.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/quickstart2.gif" width="100%">
    </td>
    <td>
      <p><strong>Quickstart 2</strong></p>
      <p>Animated cosine wave with shifting phase and amplitude, as a loop of
      <code>print(plot - prev)</code>.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/quickstart2.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/quickstart3.gif" width="100%">
    </td>
    <td>
      <p><strong>Quickstart 3</strong></p>
      <p>The same animation with the loop handed to <code>mp.animate</code>,
      which keeps the previous frame and the frame clock.</p>
      <p><em>By Claude Opus 5.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/quickstart3.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/scatter.png" width="100%">
    </td>
    <td>
      <p><strong>Scatter</strong></p>
      <p>Spiral scatter plot with viridis colormap.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/scatter.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/starburst.gif" width="100%">
    </td>
    <td>
      <p><strong>Starburst</strong></p>
      <p>Line plot colour and thickness test.</p>
      <p><em>By Claude Opus 5.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/starburst.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/teacher_student.gif" width="100%">
    </td>
    <td>
      <p><strong>Teacher-student regression</strong></p>
      <p>Gradient descent on a simple teacher-student linear regression
      model.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/teacher_student.py">Source</a></p>
    </td>
  </tr>
</tbody>
</table>

Surface Data Visualisation
--------------------------


<table>
<thead>
  <th width="50%">Image</th>
  <th width="50%">Example</th>
</thead>
<tbody>
  <tr>
    <td align="center">
      <img src="../images/chromatic_flow.gif" width="100%">
    </td>
    <td>
      <p><strong>Chromatic flow</strong></p>
      <p>A moving incompressible velocity field, with hue showing direction
      and brightness showing speed. Its custom colormap consumes two-component
      vectors directly.</p>
      <p><em>By GPT 5.6 Sol.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/chromatic_flow.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/functions.png" width="100%">
    </td>
    <td>
      <p><strong>Functions</strong></p>
      <p>Mathematical function visualisation with scatter and function2.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/functions.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/time_series_histogram.png" width="100%">
    </td>
    <td>
      <p><strong>Time series histogram</strong></p>
      <p>Time series visualisation with stacked scatter, pooled scatter, and 2D
      histogram.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/time_series_histogram.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/voronoi.png" width="100%">
    </td>
    <td>
      <p><strong>Voronoi diagram</strong></p>
      <p>Voronoi diagram using function heatmaps and scipy.</p>
      <p><em>By Gemini 2.5 Pro.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/voronoi.py">Source</a></p>
    </td>
  </tr>
</tbody>
</table>

Nonlinear Visualisation
-----------------------


<table>
<thead>
  <th width="50%">Image</th>
  <th width="50%">Example</th>
</thead>
<tbody>
  <tr>
    <td align="center">
      <img src="../images/calendar_heatmap.png" width="100%">
    </td>
    <td>
      <p><strong>Calendar heatmap</strong></p>
      <p>Calendar heatmap of daily maximum temperatures in Oxford, 2025.</p>
      <p><em>By Matthew Farrugia-Roberts and Claude Opus 4.6.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/calendar_heatmap.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/hilbert_curve.png" width="100%">
    </td>
    <td>
      <p><strong>Hilbert curve</strong></p>
      <p>Hilbert curve visualisation of binomial data.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/hilbert_curve.py">Source</a></p>
    </td>
  </tr>
</tbody>
</table>

Bars and Columns
----------------


<table>
<thead>
  <th width="50%">Image</th>
  <th width="50%">Example</th>
</thead>
<tbody>
  <tr>
    <td align="center">
      <img src="../images/jointplot.png" width="100%">
    </td>
    <td>
      <p><strong>Joint distribution</strong></p>
      <p>Joint distribution with marginal histograms, demonstrating plot
      composition with hstack and vstack.</p>
      <p><em>By Claude Opus 4.6.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/jointplot.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/sorting.gif" width="100%">
    </td>
    <td>
      <p><strong>Sorting algorithms</strong></p>
      <p>Various sorting algorithms racing in parallel, visualised with <code>mp.columns</code> and <code>mp.wrap</code>.</p>
      <p><em>By Gemini 3.1 Pro.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/sorting.py">Source</a></p>
    </td>
  </tr>
</tbody>
</table>

Media
-----


<table>
<thead>
  <th width="50%">Image</th>
  <th width="50%">Example</th>
</thead>
<tbody>
  <tr>
    <td align="center">
      <img src="../images/image.png" width="100%">
    </td>
    <td>
      <p><strong>Image rendering</strong></p>
      <p>Image rendering with various colormaps.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/image.py">Source</a></p>
    </td>
  </tr>
</tbody>
</table>

Retro Animations
----------------


<table>
<thead>
  <th width="50%">Image</th>
  <th width="50%">Example</th>
</thead>
<tbody>
  <tr>
    <td align="center">
      <img src="../images/boing.gif" width="100%">
    </td>
    <td>
      <p><strong>Boing</strong></p>
      <p>The Amiga Boing Ball, 1984, animated using a rotating colour
      palette.</p>
      <p><em>By Claude Opus 5.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/boing.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/doomfire.gif" width="100%">
    </td>
    <td>
      <p><strong>Doom fire</strong></p>
      <p>The classic 1997 PSX Doom fire effect mapped through a custom palette using <code>mp.image</code>.</p>
      <p><em>By Gemini 3.1 Pro.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/doomfire.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/teapot.gif" width="100%">
    </td>
    <td>
      <p><strong>Teapot</strong></p>
      <p>3D scatter plot with animated camera orbit.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/teapot.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/vaporwave.gif" width="100%">
    </td>
    <td>
      <p><strong>Vaporwave</strong></p>
      <p>A wireframe landscape scrolling under a banded sun. The backdrop of
      sky, ground and sun is an image of half-blocks; the terrain is one
      <code>mp.line3</code> over the top of it.</p>
      <p><em>By Matthew Farrugia-Roberts and Claude Opus 5.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/vaporwave.py">Source</a></p>
    </td>
  </tr>
</tbody>
</table>

Simulations
-----------


<table>
<thead>
  <th width="50%">Image</th>
  <th width="50%">Example</th>
</thead>
<tbody>
  <tr>
    <td align="center">
      <img src="../images/boids.gif" width="100%">
    </td>
    <td>
      <p><strong>Boids</strong></p>
      <p>
      Boids Flocking Simulation. Simulates autonomous agents (boids) moving in
      2D space based on simple rules.
      </p>
      <p><em>By Gemini 3.1 Pro.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/boids.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/life.gif" width="100%">
    </td>
    <td>
      <p><strong>Life</strong></p>
      <p>Conway's Game of Life with extra colours for newly alive/dead cells.
      The panels underneath track the cell counts and, on the right, number of
      terminal bytes written using different rendering methods.</p>
      <p><em>By Claude Opus 5.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/life.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/lorenz.gif" width="100%">
    </td>
    <td>
      <p><strong>Lorenz attractor</strong></p>
      <p>Animated 3D Lorenz attractors showing sensitive dependence on initial conditions.</p>
      <p><em>By Gemini 3.1 Pro.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/lorenz.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/mandelbrot.gif" width="100%">
    </td>
    <td>
      <p><strong>Mandelbrot</strong></p>
      <p>Animated Mandelbrot fractal zoom using function heatmaps and
      colormaps.</p>
      <p><em>By Gemini 2.5 Pro.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/mandelbrot.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/three_body.gif" width="100%">
    </td>
    <td>
      <p><strong>Three bodies</strong></p>
      <p>Three equal masses chasing one another around a shared figure-eight
      orbit, integrated under Newtonian gravity.</p>
      <p><em>By GPT 5.6 Sol.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/three_body.py">Source</a></p>
    </td>
  </tr>
</tbody>
</table>

Dashboards and UI
-----------------


<table>
<thead>
  <th width="50%">Image</th>
  <th width="50%">Example</th>
</thead>
<tbody>
  <tr>
    <td align="center">
      <img src="../images/dashboard.gif" width="100%">
    </td>
    <td>
      <p><strong>Dashboard</strong></p>
      <p>Live system monitoring dashboard showing CPU and memory usage.</p>
      <p><em>By Gemini 2.5 Pro.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/dashboard.py">Source</a></p>
    </td>
  </tr>
</tbody>
</table>

Utilities
---------


<table>
<thead>
  <th width="50%">Image</th>
  <th width="50%">Example</th>
</thead>
<tbody>
  <tr>
    <td align="center">
      <img src="../images/axes_gallery.png" width="100%">
    </td>
    <td>
      <p><strong>Axes gallery</strong></p>
      <p>Every way of drawing an axis, around one two-slit interference
      pattern: what each side can be, which sides carry the scale, the weight
      of the rules, and how a quantity with a single coordinate is labelled.</p>
      <p><em>By Claude Opus 5.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/axes_gallery.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/colormaps.png" width="100%">
    </td>
    <td>
      <p><strong>Colormaps</strong></p>
      <p>Gallery of all available continuous and discrete colormaps.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/colormaps.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../images/demo.png" width="100%">
    </td>
    <td>
      <p><strong>Demo</strong></p>
      <p>Original library example combining images, borders, and scatter plots.</p>
      <p><em>By Matthew Farrugia-Roberts.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/demo.py">Source</a></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      Run <code>python examples/terminal_test.py</code>
    </td>
    <td>
      <p><strong>Terminal test</strong></p>
      <p>Does your terminal render matthewplotlib correctly? Test of escape
      sequences for colour, redrawing in place, resizing, and a plot pushed
      against the right margin.</p>
      <p>See also <a href="compatibility.html">compatibility</a>.</p>
      <p><em>By Claude Opus 5.</em></p>
      <p><a href="https://github.com/matomatical/matthewplotlib/blob/main/examples/terminal_test.py">Source</a></p>
    </td>
  </tr>
</tbody>
</table>
