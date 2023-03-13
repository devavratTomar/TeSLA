############### Radar Factory ###############
"Code from https://github.com/fiveai/LAME/blob/master/src/utils/figures.py"

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.path import Path as plt_path


def radar_plot(cases, methods, scores):

    n_cases = len(cases)
    radar_factory(n_cases, frame='circle')

    fig, ax = plt.subplots(figsize=(20, 13), nrows=1, ncols=1, 
                            subplot_kw=dict(projection='radar'), squeeze=True, sharey=True, sharex=True)
    BG_WHITE = "#fbf9f4"
    BLUE = "#2a475e"
    GREY70 = "#b3b3b3"
    GREY_LIGHT = "#f2efe8"

    angle = 0.5
    ANGLES = [n / n_cases * 2 * np.pi for n in range(n_cases)]
    # ANGLES += ANGLES[:1]
    HANGLES = np.linspace(0, 2 * np.pi)
    l = np.linspace(0.20,0.9,6)
    H = [np.ones(len(HANGLES)) * li for li in l]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Setting lower limit to negative value reduces overlap
    # for values that are 0 (the minimums)
    ax.set_ylim(l[0] - 0.02, l[-1])

    # Set values for the angular axis (x)
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(cases, size=30, y=-0.1)

    # Remove lines for radial axis (y)
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")

    # Add custom lines for radial axis (y) at 0, 0.5 and 1.
    _ = [ax.plot(HANGLES, h, ls=(0, (6, 6)), c=GREY70) for h in H]

    # Add levels -----------------------------------------------------
    # These labels indicate the values of the radial axis
    PAD = 0.005

    size = 20
    _ = [ax.text(angle, li + PAD, f"{int(li * 100)}%", size=size) for li in l]

    # Now fill the area of the circle with radius 1.
    # This create the effect of gray background.
    ax.fill(HANGLES, H[-1], GREY_LIGHT)

    colors = ['b', 'r', 'g','m', 'y', 'c']

    for i, (method, color) in enumerate(zip(methods, colors)):
        ax.plot(ANGLES, scores[i], c=color, linewidth=3, label=method)
        ax.scatter(ANGLES, scores[i],c=color, zorder=10)

    ax.legend(
        loc='center',
        bbox_to_anchor=[1.4, 0.5],       # bottom-right
        ncol=1,
        frameon=False,     # don't put a frame
        prop={'size': 32}
    )
        
    return fig, ax



def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.
    This function creates a RadarAxes projection and registers it.
    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=plt_path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
