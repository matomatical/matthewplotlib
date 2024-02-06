"""
Terminal-based (live) plotting library.
"""

import numpy as np
import einops
import itertools


# # # PLOTTING


def hist(
    data,
    weights=None,
    density=False,
    lo=None,
    hi=None,
    bins=30,
    height=22,
    labelformat="4.2f",
    countformat="",
):
    """
    Print a histogram plot of the sequence of samples in `data`, binned
    into  boundaries `bins` (if `bins` is an int, then the data are
    separated into `bins` even width bins between `lo` (default: min(data)
    and `hi` (default: max(data)).

    The bin boundaries are shown below each bin using `labelformat` and
    the counts are shown with `countformat`. The bars are colored using
    the colormap `color`. TODO: Allow single color.
    """
    data = np.asarray(data)
    hist, bins = np.histogram(
        a=data,
        bins=bins,
        range=(
            data.min() if lo is None else lo,
            data.max() if hi is None else hi,
        ),
        weights=weights,
        density=density,
    )
    # build the bin labels
    labels = []
    for i, (b1, b2) in enumerate(np.column_stack((bins[:-1], bins[1:]))):
        l1 = format(b1, labelformat)
        l2 = format(b2, labelformat)
        ub = ")" if i == len(bins)-1 else "]"
        labels.append(f"[{l1}, {l2}{ub}")
    # plot the histogram (just a bar chart)
    return bars(
        values=hist,
        labels=labels,
        height=height,
        valueformat=countformat,
        labelformat="",
    )


def hist2d(
    xdata_or_data, #n
    ydata=None,
    weights=None,
    density=False,
    xrange=None,
    yrange=None,
    xbins=30,
    ybins=30,
    labelformat="4.2f",
    countformat="",
    colormap=None,
):
    """
    Print a histogram plot of the sequence of x,y samples in `xdata`, `ydata`
    (or `data`, an n by 2 array),
    binned into boundaries `xbins`, `ybins` (if `bins` is an int, then the
    data are separated into `bins` even width bins between `xlo` (default:
    min(xdata) and `xhi` (default: max(xdata)).
    """
    if ydata is None:
        data = np.asarray(xdata_or_data)
    else:
        data = np.column_stack((xdata_or_data, ydata))
    if xrange is None:
        xrange = (data[:,0].min(), data[:,0].max())
    if yrange is None:
        yrange = (data[:,1].min(), data[:,1].max())
    # generate the histogram
    hist, _xbins, _ybins = np.histogram2d(
        x=data[:,0],
        y=data[:,1],
        bins=(xbins, ybins),
        range=(xrange, yrange),
        weights=weights,
        density=density,
    )
    # TODO: add labels...?
    # plot the histogram (just a heatmap)
    hist = einops.rearrange(hist, 'x y -> y x')[::-1] # flip y
    return image(
        im=hist / hist.max(),
        colormap=colormap,
    )


def hist2d_rgb(
    rdata, #n
    gdata=None,
    bdata=None,
    weights=None,
    density=False,
    xrange=None,
    yrange=None,
    xbins=30,
    ybins=30,
):
    rdata = np.asarray(rdata)
    if gdata is None:
        gdata = np.empty((0, 2))
    if bdata is None:
        bdata = np.empty((0, 2))
    data = np.vstack([rdata, gdata, bdata])
    if xrange is None:
        xrange = (data[:,0].min(), data[:,0].max())
    if yrange is None:
        yrange = (data[:,1].min(), data[:,1].max())
    # generate the histogram
    hist = np.zeros((xbins, ybins, 3))
    hist[:, :, 0], _, _ = np.histogram2d(
        x=rdata[:,0],
        y=rdata[:,1],
        bins=(xbins, ybins),
        range=(xrange, yrange),
        weights=weights,
        density=density,
    )
    hist[:, :, 1], _, _ = np.histogram2d(
        x=gdata[:,0],
        y=gdata[:,1],
        bins=(xbins, ybins),
        range=(xrange, yrange),
        weights=weights,
        density=density,
    )
    hist[:, :, 2], _, _ = np.histogram2d(
        x=bdata[:,0],
        y=bdata[:,1],
        bins=(xbins, ybins),
        range=(xrange, yrange),
        weights=weights,
        density=density,
    )
    # TODO: add labels...?
    # plot the histogram (just a heatmap)
    hist = einops.rearrange(hist, 'x y c -> y x c')[::-1] # flip y
    return image(im=hist / hist.max())


def bars(
    values,
    labels=None,
    height=22,
    valueformat="",
    labelformat="",
):
    """
    Print a bar chart with `values` and options below `labels`. `colors`
    is an optional list of `colors` of the bars. The values are printed
    below each bar with format `valueformat` and the labels are formatted
    with `labelformat`.
    """
    values = np.asarray(values)
    # compute the bar heights
    vmax = values.max()
    heights = height * values / values.max()
    # balance the labels
    if labels is not None:
        labels = [format(l, labelformat) for l in labels]
        lmax = max(len(l) for l in labels)
        labels = [l.rjust(lmax)+" " for l in labels]
    else:
        labels = [""] * len(values)
    # balance the value-labels
    valuelabels = [format(v, valueformat) for v in values]
    vlmax = max(len(vl) for vl in valuelabels)
    valuelabels = [f"({vl.rjust(vlmax)}) " for vl in valuelabels]
    # print the bar chart!
    lines = []
    for lab, vlab, ht in zip(labels, valuelabels, heights):
        bar = "█" * int(ht) + " ▏▎▍▌▋▊▉█"[int(9*(ht % 1))]
        lines.append(Line(*[Char(c) for c in f"{lab}{vlab}{bar}"]))
    return TextBox(*lines)


def image(im, downsample=None, upsample=None, colormap=None):
    im = np.asarray(im)
    # convert to RGB
    if len(im.shape) == 2 and colormap is None:
        im = einops.repeat(im, 'h w -> h w 3') # simplistic
    elif colormap is not None:
        im = colormap(im) # colormap: h w [bw OR r g b] -> h w [r g b]
    # downsample by a factor
    if downsample is not None and downsample > 1:
        im = einops.reduce(
            im,
            "(h h2) (w w2) c -> h w c",
            'mean',
            h2=downsample,
            w2=downsample,
        )
    # upsample by a factor
    if upsample is not None and upsample > 1:
        im = einops.repeat(
            im,
            "h w c -> (h h2) (w w2) c",
            h2=upsample,
            w2=upsample,
        )
    # convert to odd
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
    # hope it's a good size now!
    # stack to fg/bg
    im = einops.rearrange(im, '(h h2) w c -> h w h2 c', h2=2)
    # print the image
    return TextBox(*[
        Line(*[
            Char("▀", fgbg=(tuple(fg), tuple(bg)))
            for fg, bg in row
        ])
        for row in im
    ])


# # PRINT VERSIONS

def print_hist(*args, **kwargs):
    print(hist(*args, **kwargs))

def print_hist2d(*args, **kwargs):
    print(hist2d(*args, **kwargs))

def print_hist2d_rgb(*args, **kwargs):
    print(hist2d_rgb(*args, **kwargs))

def print_bars(*args, **kwargs):
    print(bars(*args, **kwargs))

def print_image(*args, **kwargs):
    print(image(*args, **kwargs))


# # COLOR SCHEMES


def viridis(x):
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
    """https://lospec.com/palette-list/sweetie-16"""
    return np.array([
        [.101,.109,.172],[.364,.152,.364],[.694,.243,.325],[.937,.490,.341],
        [.999,.803,.458],[.654,.941,.439],[.219,.717,.392],[.145,.443,.474],
        [.160,.211,.435],[.231,.364,.788],[.254,.650,.964],[.450,.937,.968],
        [.956,.956,.956],[.580,.690,.760],[.337,.423,.525],[.2  ,.235,.341],
    ])[x]


# # UTILS


def _discretize(x, n=256):
    """
    float[0,1] grat -> int8 i
    """
    return (np.clip(x, 0., 1.) * (n-1)).astype(int)

    
# # # Return Objects

class Char:
    def __init__(self, char, fgbg=None):
        self.char = char
        self.fgbg = fgbg
        self.do_color = fgbg is not None
        self.do_reset = self.do_color

    def __str__(self):
        color_code = _color_code(self.fgbg) if self.do_color else ""
        reset_code = _RESET_CODE if self.do_reset else ""
        return f"{color_code}{self.char}{reset_code}"
    
    def __repr__(self):
        if fgbg:
            return f"Char({self.char!r}, fgbg={self.fgbg!r})"
        else:
            return f"Char({self.char!r})"

def _color_code(fgbg):
    fg, bg = fgbg
    fg_code = f'\033[38;{_color_encode(fg)}m' if fg else ""
    bg_code = f'\033[48;{_color_encode(bg)}m' if bg else ""
    return f"{fg_code}{bg_code}"

def _color_encode(c):
    r, g, b = int(255 * c[0]), int(255 * c[1]), int(255 * c[2])
    return f"2;{r};{g};{b}"

_RESET_CODE = "\033[0m"


class Line:
    def __init__(self, *chars, width=None):
        self.chars = chars
        self.w = width if width is not None else len(chars)
        # color compression
        for ch1, ch2 in zip(self.chars, self.chars[1:]):
            if ch1.do_reset and (ch1.fgbg == ch2.fgbg):
                ch1.do_reset = False
                ch2.do_color = False

    def __str__(self):
        return "".join([str(c) for c in self.chars]).ljust(self.w)

    def __repr__(self):
        args = "".join(repr(c)+"," for c in self.chars)
        return f"Line({args}width={self.width})"

    def __len__(self):
        return self.w


class TextBox:
    def __init__(self, *lines, width=None, height=None):
        self.lines = lines
        self.h = height if height is not None else len(lines)
        self.w = width if width is not None else max(len(l) for l in lines)

    def __str__(self):
        # TODO: respect width
        return "\n".join([str(l) for l in self.lines])


# # # Compositions (TODO: UPDATE TO USE TEXTBOX)


def hstack(*ss):
    return "\n".join(
        "".join(l) for l in itertools.zip_longest(*[
            str(s).splitlines() for s in ss
        ], fillvalue="")
    )


def vstack(*ss):
    return "\n".join([str(s) for s in ss])


# # # DEMO


if __name__ == "__main__":
    x = np.random.normal(size=1000)
    y = x + np.random.normal(scale=0.5, size=1000)
    z = -x + np.random.normal(scale=0.6, size=1000)
    xy = np.stack([x, y]).T
    xz = np.stack([x, z]).T
    print(hstack(
        vstack("duelling histogram:", hist2d_rgb(xy, xz)),
        vstack("2d histogram:", hist2d(x, y, colormap=viridis)),
        vstack("1d histogram:", hist(x, bins=15)),
    ))
