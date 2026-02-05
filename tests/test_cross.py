# Fix MPI/OFI finalization errors on macOS
import os

os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

import numpy as np
import pytest

try:
    from ch_timedisc.intital_conditions.cross import Cross2D, Cross3D
except ImportError:
    pytest.skip("dolfinx not available", allow_module_level=True)


def test_cross2d_indicator_values():
    cross = Cross2D(width=0.2)

    points = np.array(
        [
            [0.5, 0.5, 0.5, 0.3, 0.9],  # x
            [0.5, 0.6, 0.9, 0.5, 0.1],  # y
        ]
    )

    # Inside cross arms
    # (0.5, 0.6): vertical arm within width/2 in x and within width in y
    # (0.5, 0.5): center
    # (0.3, 0.5): horizontal arm within width/2 in y and within width in x
    # Outside cross arms
    # (0.5, 0.9) and (0.9, 0.1)
    expected = np.array([1.0, 1.0, 0.0, 1.0, 0.0])

    result = cross(points)

    assert np.array_equal(result, expected)


def test_cross3d_indicator_values():
    cross = Cross3D(width=0.2)

    points = np.array(
        [
            [0.6, 0.4, 0.5, 0.8, 0.2, 0.9],  # x
            [0.5, 0.5, 0.6, 0.5, 0.2, 0.1],  # y
            [0.5, 0.5, 0.5, 0.6, 0.5, 0.9],  # z
        ]
    )

    # Inside spikes: (0.6,0.5,0.5), (0.4,0.5,0.5), (0.5,0.6,0.5), (0.8,0.5,0.6)
    # Outside: (0.2,0.2,0.5), (0.9,0.1,0.9)
    expected = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    result = cross(points)

    assert np.array_equal(result, expected)
