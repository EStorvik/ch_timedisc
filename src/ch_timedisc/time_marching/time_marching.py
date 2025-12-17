"""Time stepping and simulation control for Cahn-Hilliard equations."""


class TimeMarching:
    """Manages time stepping and simulation evolution for Cahn-Hilliard problems.

    This class orchestrates the time-dependent solution process, handling:
    - Time stepping loops
    - Solution updates and time evolution
    - Nonlinear solver control
    - Energy tracking
    - Visualization updates
    - Verbosity control

    Attributes:
        femhandler (FEMHandler): Finite element handler with solution variables.
        parameters (Parameters): Simulation parameters (t0, dt, num_time_steps).
        energy (Energy): Energy tracker for monitoring energy evolution.
        problem (NonlinearProblem): The nonlinear problem to solve at each step.
        verbose (bool): Enable detailed output during time stepping.
        viz (Visualization, optional): Visualization handler for live updates.
    """

    def __init__(
        self,
        femhandler,
        parameters,
        energy,
        problem,
        verbose=False,
        viz=None,
    ):
        """Initialize the time marching controller.

        Args:
            femhandler (FEMHandler): Finite element handler with spaces and functions.
            parameters (Parameters): Simulation parameters object.
            energy (Energy): Energy tracker object.
            problem (NonlinearProblem): Nonlinear problem from FEniCSx.
            verbose (bool, optional): Print convergence info. Defaults to False.
            viz (Visualization, optional): Visualization object for updates.
                Defaults to None (no visualization).
        """
        self.femhandler = femhandler
        self.parameters = parameters
        self.energy = energy
        self.problem = problem
        self.verbose = verbose
        self.viz = viz
        self.time_vec = []

    def __call__(self):
        """Execute the time stepping loop and return time vector.

        Performs the main simulation loop:
        - Evolves the solution from t0 to t_final = t0 + num_time_steps * dt
        - Solves the nonlinear system at each time step
        - Tracks energy evolution
        - Updates visualization if provided
        - Monitors solver convergence

        Returns:
            list: Time values at each time step.
        """
        # Time stepping
        t = self.parameters.t0
        self.time_vec.append(t)

        for i in range(self.parameters.num_time_steps):
            # Copy current solution to old for time stepping
            self.femhandler.xi_old.x.array[:] = self.femhandler.xi.x.array
            self.femhandler.xi_old.x.scatter_forward()

            # Increment time
            t += self.parameters.dt

            # Solve the nonlinear system at current time step
            n, converged = self.problem.solve()

            if not converged:
                print(f"WARNING: Newton solver did not converge at time step {i}")

            if self.verbose:
                print(f"Used {n} newton iterations to converge at time step {i}.")

            self.time_vec.append(t)

            # Update visualization if provided
            if self.viz is not None:
                self.viz.update(self.femhandler.xi.sub(0), t)

            # Track and print energy
            print(self.energy(self.femhandler.pf, self.femhandler.mu))

