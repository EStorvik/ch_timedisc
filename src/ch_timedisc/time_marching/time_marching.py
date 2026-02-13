"""Time stepping and simulation control for Cahn-Hilliard equations."""

from typing import TYPE_CHECKING, List, Optional, Union
from tqdm import tqdm
from pathlib import Path
import numpy as np


if TYPE_CHECKING:
    from ch_timedisc.fem import FEMHandler
    from ch_timedisc.parameters import Parameters
    from ch_timedisc.visualization import (
        PyvistaVizualization,
        PyvistaVizualization3D,
    )
    from ch_timedisc.energy import Energy
    from ch_timedisc.time_marching import AdaptiveTimeStep
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.io.utils import XDMFFile


class TimeMarching:
    """Manages time stepping and simulation evolution for Cahn-Hilliard problems.

    This class orchestrates the time-dependent solution process, handling:
    - Time stepping loops
    - Solution updates and time evolution (with potential adaptive time-step sizes)
    - Nonlinear solver
    - Energy tracking
    - Visualization updates

    Attributes:
        femhandler (FEMHandler): Finite element handler with solution variables.
        parameters (Parameters): Simulation parameters (t0, dt, num_time_steps).
        energy (Energy): Energy tracker for monitoring energy evolution.
        problem (NonlinearProblem): The nonlinear problem to solve at each step.
        adaptivive_time_step (AdaptiveTimeStep, Optional): adaptive time step rules.
        verbose (bool): Enable detailed output during time stepping.
        viz (Visualization, optional): Visualization handler for live updates.
    """

    def __init__(
        self,
        femhandler: "FEMHandler",
        parameters: "Parameters",
        energy: "Energy",
        problem: "NonlinearProblem",
        adaptive_time_step: Optional["AdaptiveTimeStep"] = None,
        verbose: bool = False,
        viz: Optional[Union["PyvistaVizualization", "PyvistaVizualization3D"]] = None,
        output_file: Optional["XDMFFile"] = None,
        numpy_output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the time marching controller.

        Args:
            femhandler: Finite element handler with spaces and functions.
            parameters: Simulation parameters object.
            energy: Energy tracker object.
            problem: Nonlinear problem from FEniCSx.
            adaptive_time_step: Adaptive time stepping controller. Defaults to None.
            verbose: Print convergence info. Defaults to False.
            viz: Visualization object for updates. Defaults to None (no visualization).
            output_file: Output file handler for saving results. Defaults to None.
            numpy_output_dir: Directory to save numpy arrays of solutions. Defaults to None.
        """
        self.femhandler: "FEMHandler" = femhandler
        self.parameters: "Parameters" = parameters
        self.energy: "Energy" = energy
        self.problem: "NonlinearProblem" = problem
        self.adaptive_time_step: Optional["AdaptiveTimeStep"] = adaptive_time_step
        self.verbose: bool = verbose
        self.viz: Optional[Union["PyvistaVizualization", "PyvistaVizualization3D"]] = (
            viz
        )
        self.time_vec: List[float] = []
        self.output_file: Optional["XDMFFile"] = output_file
        self.numpy_output_dir: Optional[Path] = (
            Path(numpy_output_dir) if numpy_output_dir is not None else None
        )

        # Storage for numpy arrays at output times
        self.solution_arrays: dict = {}  # {time: pf_array}
        self.mu_arrays: dict = {}  # {time: mu_array}

    def __call__(self) -> List[float]:
        """Execute the time stepping loop and return time vector.

        Performs the main simulation loop:
        - Evolves the solution from t0 to t_final = t0 + num_time_steps * dt
        - Solves the nonlinear system at each time step
        - Tracks energy evolution
        - Updates visualization if provided
        - Monitors solver convergence

        Returns:
            Time values at each time step.
        """
        # Time stepping
        t = self.parameters.t0
        self.time_vec.append(t)

        i = 0

        # Track which output times have been written
        output_times_written = set()

        # Write initial condition if output_file exists and either no output_times specified
        # or t0 is in the output_times list
        if self.output_file is not None:
            if self.parameters.output_times is None:
                # Write at every step (original behavior)
                pf_out, _ = self.femhandler.xi.split()
                self.output_file.write_function(pf_out, t)
            elif any(
                abs(t - output_t) < 1e-10 for output_t in self.parameters.output_times
            ):
                # Write initial condition if it's in output_times
                pf_out, _ = self.femhandler.xi.split()
                self.output_file.write_function(pf_out, t)
                output_times_written.add(t)

        # Save initial condition as numpy array if requested
        if self.numpy_output_dir is not None:
            if self.parameters.output_times is None or any(
                abs(t - output_t) < 1e-10 for output_t in self.parameters.output_times
            ):
                pf_out, mu_out = self.femhandler.xi.split()
                self.solution_arrays[t] = pf_out.x.array[:].copy()
                self.mu_arrays[t] = mu_out.x.array[:].copy()

        # Initialize progress bar if verbose
        if self.verbose:
            pbar = tqdm(
                total=self.parameters.T - self.parameters.t0,
                desc="Time evolution",
                unit="s",
                bar_format=(
                    "{l_bar}{bar}| {n:.4e}/{total:.4e} "
                    "[{elapsed}<{remaining}, dt={postfix}]"
                ),
            )
            pbar.set_postfix_str(f"{self.parameters.dt:.4e}")

        while t < self.parameters.T:
            i += 1
            # Copy current solution to old for time stepping
            self.femhandler.xi_old.x.array[:] = self.femhandler.xi.x.array
            self.femhandler.xi_old.x.scatter_forward()

            n, converged = self.problem.solve()
            if not converged:
                print(f"WARNING: Newton solver did not converge at time step {i}")

            if self.verbose and not self.adaptive_time_step:
                print(f"Used {n} Newton iterations to converge at time step {i}.")

            if self.adaptive_time_step is not None:
                while self.adaptive_time_step.criterion() == "decrease":
                    self.problem = self.adaptive_time_step.update_dt("decrease", i)
                    n, converged = self.problem.solve()
                    if not converged:
                        print(
                            f"WARNING: Newton solver did not converge at time step {i} within adaptive time step decrease."
                        )
                    if self.verbose:
                        pbar.set_postfix_str(f"{self.parameters.dt:.4e}")

                if self.adaptive_time_step.criterion() == "increase":
                    self.problem = self.adaptive_time_step.update_dt("increase", i)
                    if self.verbose:
                        pbar.set_postfix_str(f"{self.parameters.dt:.4e}")

            # Update and track energy
            self.energy()

            # Increment time
            t += self.parameters.dt
            self.time_vec.append(t)

            # Update progress bar
            if self.verbose:
                pbar.update(self.parameters.dt)

            # Update visualization if provided
            if self.viz is not None:
                self.viz.update(self.femhandler.xi.sub(0), t)

            # Write output at predetermined times or every step
            if self.output_file is not None:
                if self.parameters.output_times is None:
                    # Write at every time step (original behavior)
                    pf_out, _ = self.femhandler.xi.split()
                    self.output_file.write_function(pf_out, t)
                else:
                    # Check if current time crosses any output checkpoint
                    for output_time in self.parameters.output_times:
                        # Check if this time hasn't been written yet and current time crosses it
                        if (
                            output_time not in output_times_written
                            and t - self.parameters.dt < output_time <= t + 1e-10
                        ):
                            pf_out, _ = self.femhandler.xi.split()
                            self.output_file.write_function(pf_out, output_time)
                            output_times_written.add(output_time)
                            break  # Only write once per checkpoint

            # Save numpy arrays at predetermined times
            if self.numpy_output_dir is not None:
                if self.parameters.output_times is None:
                    # Save at every time step
                    pf_out, mu_out = self.femhandler.xi.split()
                    self.solution_arrays[t] = pf_out.x.array[:].copy()
                    self.mu_arrays[t] = mu_out.x.array[:].copy()
                else:
                    # Check if current time crosses any output checkpoint
                    for output_time in self.parameters.output_times:
                        if (
                            output_time not in output_times_written
                            and t - self.parameters.dt < output_time <= t + 1e-10
                        ):
                            pf_out, mu_out = self.femhandler.xi.split()
                            self.solution_arrays[output_time] = pf_out.x.array[:].copy()
                            self.mu_arrays[output_time] = mu_out.x.array[:].copy()
                            break

        # Close progress bar
        if self.verbose:
            pbar.close()

        # Save numpy arrays to file if requested
        if self.numpy_output_dir is not None and len(self.solution_arrays) > 0:
            self._save_numpy_arrays()

        return self.time_vec

    def _save_numpy_arrays(self) -> None:
        """Save collected solution arrays to numpy .npz file."""
        if self.numpy_output_dir is None:
            return

        # Create output directory if it doesn't exist
        self.numpy_output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data for saving
        times = sorted(self.solution_arrays.keys())
        pf_arrays = [self.solution_arrays[t] for t in times]
        mu_arrays = [self.mu_arrays[t] for t in times]

        # Save to .npz file
        output_file = self.numpy_output_dir / "solution_arrays.npz"
        np.savez(
            output_file,
            times=np.array(times),
            pf=np.array(pf_arrays),
            mu=np.array(mu_arrays),
            dt=self.parameters.dt,
            T=self.parameters.T,
            nx=self.parameters.nx,
            ny=self.parameters.ny,
        )

        if self.verbose:
            print(f"\nSaved {len(times)} solution snapshots to {output_file}")
