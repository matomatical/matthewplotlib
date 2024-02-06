"""
A quick and ditry implementation of a unicode-based plotting library for
displaying images in the terminal. Useful for visualising gridworlds, such as
mazes.
```

See documentation for more advanced usage.
"""

import numpy as np
import einops


# # # PLOT BASE CLASS


class plot:
    """
    Abstract base class for an ANSI plot that renders to a specified height
    and width.
    """
    def __init__(self, height, width, lines):
        self.height = height
        self.width = width
        self.lines = lines

    def __str__(self):
        return "\n".join(self.lines)

    def __and__(self, other):
        return hstack(self, other)

    def __xor__(self, other):
        return vstack(self, other)


# # # PLOT CLASSES


class image(plot):
    """
    Render a small image using a grid of unicode half-characters with
    different foreground and background colours to represent pairs of
    pixels.
    """
    def __init__(self, im, colormap=None):
        im = np.asarray(im)
        # convert to RGB
        if len(im.shape) == 2 and colormap is None:
            im = einops.repeat(im, 'h w -> h w 3') # uniform colorization
        elif colormap is not None:
            im = colormap(im) # colormap: h w bw -> h w [r g b]
        # pad to even height (and width, latter not strictly necessary)
        im = np.pad(
            array=im,
            pad_width=(
                (0, im.shape[0] % 2),
                (0, im.shape[1] % 2),
                (0, 0),
            ),
            mode='constant',
            constant_values=0.,
        )
        # stack to fg/bg
        im = einops.rearrange(im, '(h h2) w c -> h w h2 c', h2=2)
        # render the image as a plot object
        super().__init__(
            height=im.shape[0],
            width=im.shape[1],
            lines=[
                "".join([_color("▀", fg=fg, bg=bg) for fg, bg in row])
                for row in im
            ],
        )


class scatter(plot):
    """
    Render a scatterplot using a grid of braille unicode characters.
    """
    def __init__(
        self,
        data,
        xrange=None,
        yrange=None,
        width=30,
        height=10,
        color=None,
    ):
        data = np.asarray(data) # todo: enforce shape for empty
        # determine sensible bounds
        xmin, ymin = data.min(axis=0)
        xmax, ymax = data.max(axis=0)
        if xrange is None:
            xrange = (xmin, xmax)
        else:
            if xmin < xrange[0] or xmax > xrange[1]:
                print("warning: points out of x range will be clipped")
            xmin, xmax = xrange
        if yrange is None:
            yrange = (ymin, ymax)
        else:
            if ymin < yrange[0] or ymax > yrange[1]:
                print("warning: points out of y range will be clipped")
            ymin, ymax = yrange
        
        # converting float coordinates into data and character matrices
        
        # form the data grid
        dots, *_bins = np.histogram2d(
            x=data[:,0],
            y=data[:,1],
            bins=(2*width, 4*height),
            range=(xrange, yrange),
        )
        dots = dots.T     # we want y first
        dots = dots[::-1] # correct y for top-down drawing
        
        # draw onto a grid of braille dots
        grid = [[" " for _ in range(width)] for _ in range(height)]
        bgrid = _braille_encode(dots > 0)
        for i in range(height):
            for j in range(width):
                if bgrid[i, j]:
                    grid[i][j] = _color(chr(0x2800+bgrid[i, j]), fg=color)

        # render as lines
        super().__init__(
            height=height,
            width=width,
            lines=["".join(row) for row in grid],
        )
        
        # # are axes present?
        # if xmin <= 0 <= xmax:
        #     x0 = _discretize((0-xmin)/(xmax-xmin), n=width)
        # else:
        #     x0 = None
        # if ymin <= 0 <= ymax:
        #     y0 = _discretize((ymax-0)/(ymax-ymin), n=height) # sign-corrected
        # else:
        #     y0 = None

        # # draw axes onto grid (if applicable)
        # if x0 is not None:
        #     for i in range(height): grid[i][x0] = '│'
        # if y0 is not None:
        #     grid[y0] = ['─' for _ in range(width)]
        # if x0 is not None and y0 is not None:
        #     grid[y0][x0] = '┼'


# # # ARRANGEMENT


class hstack(plot):
    """
    Horizontally arrange a group of plots.
    """
    def __init__(self, *plots):
        height = max(p.height for p in plots)
        width = sum(p.width for p in plots)
        lines = [
            "".join([
                p.lines[i] if i < p.height else p.width * " "
                for p in plots
            ])
            for i in range(height)
        ]
        super().__init__(
            height=height,
            width=width,
            lines=lines,
        )
        self.plots = plots


class vstack(plot):
    """
    Vertically arrange a group of plots.
    """
    def __init__(self, *plots):
        height = sum(p.height for p in plots)
        width = max(p.width for p in plots)
        lines = [l + " " * (width - p.width) for p in plots for l in p.lines]
        super().__init__(
            height=height,
            width=width,
            lines=lines,
        )
        self.plots = plots


class wrap(plot):
    """
    Horizontally and vertically arrange a group of plots.
    """
    def __init__(self, *plots, cols=None):
        cell_height = max(p.height for p in plots)
        cell_width = max(p.width for p in plots)
        if cols is None:
            cols = 80 // cell_width
        # wrap list of plots into groups, of length `cols` (except last)
        wrapped_plots = []
        for i, plot in enumerate(plots):
            if i % cols == 0:
                wrapped_plots.append([])
            wrapped_plots[-1].append(plot)
        # combine functionality of hstack/vstack
        lines = [
            "".join([
                p.lines[i] + " " * (cell_width - p.width)
                if i < p.height else " " * cell_width
                for p in group
            ])
            for group in wrapped_plots
            for i in range(cell_height)
        ]
        # done!
        super().__init__(
            height=len(lines),
            width=min(len(plots), cols) * cell_width,
            lines=lines,
        )
        self.wrapped_plots = wrapped_plots


# # # COLOR SCHEMES


def viridis(x):
    """
    https://youtu.be/xAoljeRJ3lU
    """
    return np.array([
        [.267,.004,.329],[.268,.009,.335],[.269,.014,.341],[.271,.019,.347],
        [.272,.025,.353],[.273,.031,.358],[.274,.037,.364],[.276,.044,.370],
        [.277,.050,.375],[.277,.056,.381],[.278,.062,.386],[.279,.067,.391],
        [.280,.073,.397],[.280,.078,.402],[.281,.084,.407],[.281,.089,.412],
        [.282,.094,.417],[.282,.100,.422],[.282,.105,.426],[.283,.110,.431],
        [.283,.115,.436],[.283,.120,.440],[.283,.125,.444],[.283,.130,.449],
        [.282,.135,.453],[.282,.140,.457],[.282,.145,.461],[.281,.150,.465],
        [.281,.155,.469],[.280,.160,.472],[.280,.165,.476],[.279,.170,.479],
        [.278,.175,.483],[.278,.180,.486],[.277,.185,.489],[.276,.190,.493],
        [.275,.194,.496],[.274,.199,.498],[.273,.204,.501],[.271,.209,.504],
        [.270,.214,.507],[.269,.218,.509],[.267,.223,.512],[.266,.228,.514],
        [.265,.232,.516],[.263,.237,.518],[.262,.242,.520],[.260,.246,.522],
        [.258,.251,.524],[.257,.256,.526],[.255,.260,.528],[.253,.265,.529],
        [.252,.269,.531],[.250,.274,.533],[.248,.278,.534],[.246,.283,.535],
        [.244,.287,.537],[.243,.292,.538],[.241,.296,.539],[.239,.300,.540],
        [.237,.305,.541],[.235,.309,.542],[.233,.313,.543],[.231,.318,.544],
        [.229,.322,.545],[.227,.326,.546],[.225,.330,.547],[.223,.334,.548],
        [.221,.339,.548],[.220,.343,.549],[.218,.347,.550],[.216,.351,.550],
        [.214,.355,.551],[.212,.359,.551],[.210,.363,.552],[.208,.367,.552],
        [.206,.371,.553],[.204,.375,.553],[.203,.379,.553],[.201,.383,.554],
        [.199,.387,.554],[.197,.391,.554],[.195,.395,.555],[.194,.399,.555],
        [.192,.403,.555],[.190,.407,.556],[.188,.410,.556],[.187,.414,.556],
        [.185,.418,.556],[.183,.422,.556],[.182,.426,.557],[.180,.429,.557],
        [.179,.433,.557],[.177,.437,.557],[.175,.441,.557],[.174,.445,.557],
        [.172,.448,.557],[.171,.452,.557],[.169,.456,.558],[.168,.459,.558],
        [.166,.463,.558],[.165,.467,.558],[.163,.471,.558],[.162,.474,.558],
        [.160,.478,.558],[.159,.482,.558],[.157,.485,.558],[.156,.489,.557],
        [.154,.493,.557],[.153,.497,.557],[.151,.500,.557],[.150,.504,.557],
        [.149,.508,.557],[.147,.511,.557],[.146,.515,.556],[.144,.519,.556],
        [.143,.522,.556],[.141,.526,.555],[.140,.530,.555],[.139,.533,.555],
        [.137,.537,.554],[.136,.541,.554],[.135,.544,.554],[.133,.548,.553],
        [.132,.552,.553],[.131,.555,.552],[.129,.559,.551],[.128,.563,.551],
        [.127,.566,.550],[.126,.570,.549],[.125,.574,.549],[.124,.578,.548],
        [.123,.581,.547],[.122,.585,.546],[.121,.589,.545],[.121,.592,.544],
        [.120,.596,.543],[.120,.600,.542],[.119,.603,.541],[.119,.607,.540],
        [.119,.611,.538],[.119,.614,.537],[.119,.618,.536],[.120,.622,.534],
        [.120,.625,.533],[.121,.629,.531],[.122,.633,.530],[.123,.636,.528],
        [.124,.640,.527],[.126,.644,.525],[.128,.647,.523],[.130,.651,.521],
        [.132,.655,.519],[.134,.658,.517],[.137,.662,.515],[.140,.665,.513],
        [.143,.669,.511],[.146,.673,.508],[.150,.676,.506],[.153,.680,.504],
        [.157,.683,.501],[.162,.687,.499],[.166,.690,.496],[.170,.694,.493],
        [.175,.697,.491],[.180,.701,.488],[.185,.704,.485],[.191,.708,.482],
        [.196,.711,.479],[.202,.715,.476],[.208,.718,.472],[.214,.722,.469],
        [.220,.725,.466],[.226,.728,.462],[.232,.732,.459],[.239,.735,.455],
        [.246,.738,.452],[.252,.742,.448],[.259,.745,.444],[.266,.748,.440],
        [.274,.751,.436],[.281,.755,.432],[.288,.758,.428],[.296,.761,.424],
        [.304,.764,.419],[.311,.767,.415],[.319,.770,.411],[.327,.773,.406],
        [.335,.777,.402],[.344,.780,.397],[.352,.783,.392],[.360,.785,.387],
        [.369,.788,.382],[.377,.791,.377],[.386,.794,.372],[.395,.797,.367],
        [.404,.800,.362],[.412,.803,.357],[.421,.805,.351],[.430,.808,.346],
        [.440,.811,.340],[.449,.813,.335],[.458,.816,.329],[.468,.818,.323],
        [.477,.821,.318],[.487,.823,.312],[.496,.826,.306],[.506,.828,.300],
        [.515,.831,.294],[.525,.833,.288],[.535,.835,.281],[.545,.838,.275],
        [.555,.840,.269],[.565,.842,.262],[.575,.844,.256],[.585,.846,.249],
        [.595,.848,.243],[.606,.850,.236],[.616,.852,.230],[.626,.854,.223],
        [.636,.856,.216],[.647,.858,.209],[.657,.860,.203],[.668,.861,.196],
        [.678,.863,.189],[.688,.865,.182],[.699,.867,.175],[.709,.868,.169],
        [.720,.870,.162],[.730,.871,.156],[.741,.873,.149],[.751,.874,.143],
        [.762,.876,.137],[.772,.877,.131],[.783,.879,.125],[.793,.880,.120],
        [.804,.882,.114],[.814,.883,.110],[.824,.884,.106],[.835,.886,.102],
        [.845,.887,.099],[.855,.888,.097],[.866,.889,.095],[.876,.891,.095],
        [.886,.892,.095],[.896,.893,.096],[.906,.894,.098],[.916,.896,.100],
        [.926,.897,.104],[.935,.898,.108],[.945,.899,.112],[.955,.901,.118],
        [.964,.902,.123],[.974,.903,.130],[.983,.904,.136],[.993,.906,.143],
    ])[_discretize(x,256)]


def sweetie16(x):
    """
    https://lospec.com/palette-list/sweetie-16
    """
    return np.array([
        [.101,.109,.172],[.364,.152,.364],[.694,.243,.325],[.937,.490,.341],
        [.999,.803,.458],[.654,.941,.439],[.219,.717,.392],[.145,.443,.474],
        [.160,.211,.435],[.231,.364,.788],[.254,.650,.964],[.450,.937,.968],
        [.956,.956,.956],[.580,.690,.760],[.337,.423,.525],[.2  ,.235,.341],
    ])[x]


# # # UTILITIES


def _discretize(x, n=256):
    """
    float[0,1] x -> int8 i
    """
    return (np.clip(x, 0., 1.) * (n-1)).astype(int)


def _color(s, fg=None, bg=None):
    color_code = _color_code(fg, bg)
    reset_code = "\033[0m" if fg is not None or bg is not None else ""
    return f"{color_code}{s}{reset_code}"


def _color_code(fg=None, bg=None):
    fg_code = f'\033[38;{_color_encode(fg)}m' if fg is not None else ""
    bg_code = f'\033[48;{_color_encode(bg)}m' if bg is not None else ""
    return f"{fg_code}{bg_code}"


def _color_encode(c):
    r, g, b = int(255 * c[0]), int(255 * c[1]), int(255 * c[2])
    return f"2;{r};{g};{b}"


def _braille_encode(a):
    """
    Turns a HxW array of booleans into a (H//4)x(W//2) array of braille
    binary codes (suitable for specifying unicode codepoints, just add
    0x2800).
    
    braille symbol:                 binary digit representation:
                    0-o o-1
                    2-o o-3   ---->     0 b  0 0  0 0 0  0 0 0
                    4-o o-5                  | |  | | |  | | |
                    6-o o-7                  7 6  5 3 1  4 2 0
    """
    r = einops.rearrange(a, '(h h4) (w w2) -> (h4 w2) h w', h4=4, w2=2)
    b = ( r[0]      | r[1] << 3 
        | r[2] << 1 | r[3] << 4 
        | r[4] << 2 | r[5] << 5 
        | r[6] << 6 | r[7] << 7
        )
    return b


