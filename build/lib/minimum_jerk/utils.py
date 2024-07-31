"""Collection of useful functions
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib.axes import Axes

def get_steps(a: float, b: float):
    """ Find nr of timesteps

    Args:
        a (double): Time in seconds
        b (double): Timestep in seconds

    Returns:
        int: Number of timesteps in T
    """
    return round(a/b)

def _set_axes_radius_2d(ax, origin, radius) -> None:
    x, y = origin
    ax.set_xlim([x - radius, x + radius])
    ax.set_ylim([y - radius, y + radius])
    
def set_axes_equal_2d(ax: Axes) -> None:
    """Set equal x, y axes
    """
    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius_2d(ax, origin, radius)

def set_axes_format(ax: Axes, x_label: str, y_label: str) -> None:
    """Format the axes
    """
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid()