"""Time stepping and simulation control for Cahn-Hilliard equations."""

from typing import TYPE_CHECKING, List, Optional, Union
from tqdm import tqdm


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

        if self.output_file is not None:
            pf_out, _ = self.femhandler.xi.split()
            self.output_file.write_function(pf_out, t)

        # Initialize progress bar if verbose
        if self.verbose:
            pbar = tqdm(
                total=self.parameters.T - self.parameters.t0,
                desc="Time evolution",
                unit="s",
                bar_format="{l_bar}{bar}| {n:.4e}/{total:.4e} [{elapsed}<{remaining}, dt={postfix}]",
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
                print(f"Used {n} newton iterations to converge at time step {i}.")

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

            if self.output_file is not None:
                pf_out, _ = self.femhandler.xi.split()
                self.output_file.write_function(pf_out, t)

        # Close progress bar
        if self.verbose:
            pbar.close()

        return self.time_vec
