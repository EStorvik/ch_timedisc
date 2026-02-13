"""
Example script to load and analyze reference solution numpy arrays.

This demonstrates how to load the saved solution arrays for comparative analysis.
"""

import numpy as np
from pathlib import Path

# Load the reference solution
reference_dir = Path(__file__).resolve().parent.parent.parent / "reference_solution"
data = np.load(reference_dir / "solution_arrays.npz")

# Access the data
times = data["times"]
pf_solutions = data["pf"]  # Shape: (n_times, n_dofs)
mu_solutions = data["mu"]  # Shape: (n_times, n_dofs)

# Metadata
dt = data["dt"]
T = data["T"]
nx = data["nx"]
ny = data["ny"]

print(f"Loaded reference solution:")
print(f"  Number of snapshots: {len(times)}")
print(f"  Time range: [{times[0]:.4f}, {times[-1]:.4f}]")
print(f"  Mesh resolution: {nx} x {ny}")
print(f"  DOFs per field: {pf_solutions.shape[1]}")
print(f"\nOutput times: {times}")

# Example: Access solution at a specific time
time_idx = 3  # Fourth snapshot
print(f"\nSolution at t = {times[time_idx]}:")
print(
    f"  pf range: [{pf_solutions[time_idx].min():.4f}, {pf_solutions[time_idx].max():.4f}]"
)
print(
    f"  mu range: [{mu_solutions[time_idx].min():.4f}, {mu_solutions[time_idx].max():.4f}]"
)

# Example: Compute L2 difference between two solutions
if len(times) > 1:
    pf_diff = np.linalg.norm(pf_solutions[-1] - pf_solutions[0])
    print(f"\nL2 difference (initial to final): {pf_diff:.6e}")
