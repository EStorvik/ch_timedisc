"""Random initial condition generator."""

from typing import Optional

import numpy as np


class Random:
    """Random initial condition generator for phase field values."""

    def __init__(
        self, mean: float = 0.5, std: float = 0.1, seed: Optional[int] = None
    ) -> None:
        """Initialize random initial condition.

        Args:
            mean: Mean value for random distribution. Default: 0.5
            std: Standard deviation for random distribution. Default: 0.1
            seed: Random seed for reproducibility. Default: None
        """
        self.mean: float = mean
        self.std: float = std
        self.seed: Optional[int] = seed

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Generate random values for given points.

        Args:
            x: Array of shape (d, n) with n points in d dimensions.

        Returns:
            Random values clipped to [0, 1].
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        n_points = x.shape[1]
        values = np.random.normal(self.mean, self.std, n_points)
        # Clip to valid phase field range [0, 1]
        return np.clip(values, 0.0, 1.0)
