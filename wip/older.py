"""
A Python plotting library that isn't painful.
"""


# # # PLOTTING


class plot:
    def __init__(self, height, width, lines):
        self.height = height
        self.width = width
        self.lines = lines

    def __str__(self):
        return "\n".join(self.lines)

    def __and__(self, other):
        return hstack(self, other)

    def __or__(self, other):
        return vstack(self, other)


class block(plot):
    def __init__(self, height=1, width=1, color=None):
        line = _color("B" * width, fg=color)
        super().__init__(
            height=height,
            width=width,
            lines=[line for _ in range(height)],
        )
        

class hstack(plot):
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


# # # UTILITIES


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

