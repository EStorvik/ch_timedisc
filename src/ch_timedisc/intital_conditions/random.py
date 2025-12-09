"""Random initial condition generator."""

import numpy as np


class Random:
    """Random initial condition generator for phase field values."""

    def __init__(self, mean=0.5, std=0.1, seed=None):
        """Initialize random initial condition.

        Args:
            mean (float): Mean value for random distribution.
            std (float): Standard deviation for random distribution.
            seed (int, optional): Random seed for reproducibility.
        """
        self.mean = mean
        self.std = std
        self.seed = seed

    def __call__(self, x):
        """Generate random values for given points.

        Args:
            x (np.ndarray): Array of shape (d, n) with n points in d dimensions.

        Returns:
            np.ndarray: Random values clipped to [0, 1].
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        n_points = x.shape[1]
        values = np.random.normal(self.mean, self.std, n_points)
        # Clip to valid phase field range [0, 1]
        return np.clip(values, 0.0, 1.0)
