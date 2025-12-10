import numpy as np


class Cross2D:
    """Indicator function for a 2D cross centered in the unit square."""

    def __init__(self, width=0.1):
        self.width = width

    def __call__(self, x):
        """Return 1 inside the cross arms and 0 elsewhere.

        Args:
            x (np.ndarray): Array of shape (2, n) with points
                in the unit square.
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

        # X-axis spike: extends in +/- x direction, square cross-section
        spike_x_pos = np.logical_and.reduce(
            [
                x[0] >= 0.5,
                x[0] <= 0.5 + cross_width,
                np.abs(x[1] - 0.5) <= cross_width / 2,
                np.abs(x[2] - 0.5) <= cross_width / 2,
            ]
        )
        spike_x_neg = np.logical_and.reduce(
            [
                x[0] <= 0.5,
                x[0] >= 0.5 - cross_width,
                np.abs(x[1] - 0.5) <= cross_width / 2,
                np.abs(x[2] - 0.5) <= cross_width / 2,
            ]
        )

        # Y-axis spike: extends in +/- y direction, square cross-section
        spike_y_pos = np.logical_and.reduce(
            [
                x[1] >= 0.5,
                x[1] <= 0.5 + cross_width,
                np.abs(x[0] - 0.5) <= cross_width / 2,
                np.abs(x[2] - 0.5) <= cross_width / 2,
            ]
        )
        spike_y_neg = np.logical_and.reduce(
            [
                x[1] <= 0.5,
                x[1] >= 0.5 - cross_width,
                np.abs(x[0] - 0.5) <= cross_width / 2,
                np.abs(x[2] - 0.5) <= cross_width / 2,
            ]
        )

        # Z-axis spike: extends in +/- z direction, square cross-section
        spike_z_pos = np.logical_and.reduce(
            [
                x[2] >= 0.5,
                x[2] <= 0.5 + cross_width,
                np.abs(x[0] - 0.5) <= cross_width / 2,
                np.abs(x[1] - 0.5) <= cross_width / 2,
            ]
        )
        spike_z_neg = np.logical_and.reduce(
            [
                x[2] <= 0.5,
                x[2] >= 0.5 - cross_width,
                np.abs(x[0] - 0.5) <= cross_width / 2,
                np.abs(x[1] - 0.5) <= cross_width / 2,
            ]
        )

        values[
            np.logical_or.reduce(
                [
                    spike_x_pos,
                    spike_x_neg,
                    spike_y_pos,
                    spike_y_neg,
                    spike_z_pos,
                    spike_z_neg,
                ]
            )
        ] = 1.0
        return values
