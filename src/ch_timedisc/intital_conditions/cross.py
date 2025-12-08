

import numpy as np


class Cross2D:
    """Indicator function for a 2D cross centered in the unit square."""

    def __init__(self, width=0.1):
        self.width = width

    def __call__(self, x):
        """Return 1 inside the cross arms and 0 elsewhere.

        Args:
            x (np.ndarray): Array of shape (2, n) with points in the unit square.
        """
        values = np.zeros(x.shape[1])
        cross_width = self.width
        values[
            np.logical_or(
                np.logical_and(
                    (np.abs(x[0] - 0.5) <= cross_width / 2),
                    (np.abs(x[1] - 0.5) <= cross_width),
                ),
                np.logical_and(
                    (np.abs(x[1] - 0.5) <= cross_width / 2),
                    (np.abs(x[0] - 0.5) <= cross_width),
                ),
            )
        ] = 1.0
        return values


class Cross3D:
    """Indicator function for a 3D cross centered in the unit cube."""

    def __init__(self, width=0.1):
        self.width = width

    def __call__(self, x):
        """Return 1 inside the cross arms and 0 elsewhere.

        Args:
            x (np.ndarray): Array of shape (3, n) with points in the unit cube.
        """
        values = np.zeros(x.shape[1])
        cross_width = self.width

        bar_x = np.logical_and.reduce(
            [
                np.abs(x[0] - 0.5) <= cross_width / 2,
                np.abs(x[1] - 0.5) <= cross_width,
                np.abs(x[2] - 0.5) <= cross_width,
            ]
        )
        bar_y = np.logical_and.reduce(
            [
                np.abs(x[1] - 0.5) <= cross_width / 2,
                np.abs(x[0] - 0.5) <= cross_width,
                np.abs(x[2] - 0.5) <= cross_width,
            ]
        )
        bar_z = np.logical_and.reduce(
            [
                np.abs(x[2] - 0.5) <= cross_width / 2,
                np.abs(x[0] - 0.5) <= cross_width,
                np.abs(x[1] - 0.5) <= cross_width,
            ]
        )

        values[np.logical_or.reduce([bar_x, bar_y, bar_z])] = 1.0
        return values