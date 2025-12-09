"""PyVista-based visualization for FEniCSx solutions."""

import pyvista as pv
import pyvistaqt as pvqt
from dolfinx import plot

if pv.OFF_SCREEN:
    pv.start_xvfb(wait=0.5)


class PyvistaVizualization:

    def __init__(self, V, xi, t0, name="phase-field") -> None:
        """
        Initialize the visualization object.

        Args:
            V (FunctionSpace): Function space. Provide the function
                space of the solution that you want to plot. F.ex. use
                .subs(0) to get the function space of the first
                component of the solution.
            xi (Function): The solution to visualize
            t0 (float): The initial time
            name (str): The name of the scalar field to plot
        """
        self.V0, self.dofs = V.collapse()
        self.name = name

        # Create a VTK 'mesh' with 'nodes' at the function dofs
        self.topology, self.cell_types, self.x = plot.vtk_mesh(self.V0)
        self.grid = pv.UnstructuredGrid(self.topology, self.cell_types, self.x)

        # Set output data
        self.grid.point_data[name] = xi.x.array[self.dofs].real
        self.grid.set_active_scalars(name)
        self.p = pvqt.BackgroundPlotter(title=self.name, auto_update=True)
        self.p.add_mesh(self.grid, clim=[0, 1])
        self.p.view_xy(True)
        self.p.add_text(f"time: {t0}", font_size=12, name="timelabel")

    def update(self, xi, t):
        """
        Update the visualization with the new solution xi at time t.

        Args:
            xi (Function): The new solution
            t (float): The new time
        """
        self.p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        self.grid.point_data[self.name] = xi.x.array[self.dofs].real
        self.p.app.processEvents()

    # Update ghost entries and plot
    def final_plot(self, xi):
        """
        Update the visualization with the final solution.

        Args:
            xi (Function): The final solution
        """

        xi.x.scatter_forward()
        self.grid.point_data[self.name] = xi.x.array[self.dofs].real

        screenshot = None
        if pv.OFF_SCREEN:
            screenshot = self.name + ".png"
        pv.plot(self.grid, show_edges=True, screenshot=screenshot)


class PyvistaVizualization3D:
    """3D visualization with isosurfaces, volume rendering, and slicing."""

    def __init__(self, V, xi, t0, name="phase-field", mode="isosurface") -> None:
        """
        Initialize the 3D visualization object.

        Args:
            V (FunctionSpace): Function space.
            xi (Function): The solution to visualize
            t0 (float): The initial time
            name (str): The name of the scalar field to plot
            mode (str): Visualization mode - "isosurface", "volume", "slices", or "combined"
        """
        self.V0, self.dofs = V.collapse()
        self.name = name
        self.mode = mode

        # Create a VTK 'mesh' with 'nodes' at the function dofs
        self.topology, self.cell_types, self.x = plot.vtk_mesh(self.V0)
        self.grid = pv.UnstructuredGrid(self.topology, self.cell_types, self.x)

        # Set output data
        self.grid.point_data[name] = xi.x.array[self.dofs].real
        self.grid.set_active_scalars(name)

        # Create plotter
        self.p = pvqt.BackgroundPlotter(title=f"{self.name} - 3D", auto_update=True)
        self._setup_visualization()
        self.p.add_text(f"time: {t0}", font_size=12, name="timelabel")

    def _setup_visualization(self):
        """Setup the visualization based on mode."""
        if self.mode == "isosurface":
            # Show interface at pf=0.5
            contours = self.grid.contour([0.5])
            self.p.add_mesh(contours, color='red', opacity=0.8, name="interface")
        elif self.mode == "volume":
            # Volume rendering with opacity mapping
            self.p.add_volume(
                self.grid,
                opacity=[0, 0, 0.3, 0.6, 0.3, 0, 0],
                cmap="coolwarm",
                name="volume"
            )
        elif self.mode == "slices":
            # Orthogonal slice planes
            slices = self.grid.slice_orthogonal()
            self.p.add_mesh(slices, cmap="viridis", opacity=0.8, name="slices")
        elif self.mode == "combined":
            # Interface + semi-transparent volume
            contours = self.grid.contour([0.5])
            self.p.add_mesh(contours, color='red', opacity=0.9, name="interface")
            self.p.add_volume(
                self.grid,
                opacity=[0, 0, 0.2, 0.4, 0.2, 0, 0],
                cmap="coolwarm",
                name="volume"
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.p.reset_camera()

    def update(self, xi, t):
        """
        Update the visualization with the new solution xi at time t.

        Args:
            xi (Function): The new solution
            t (float): The new time
        """
        self.p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        self.grid.point_data[self.name] = xi.x.array[self.dofs].real

        # Remove old visualization
        if self.mode == "isosurface":
            self.p.remove_actor("interface")
            contours = self.grid.contour([0.5])
            self.p.add_mesh(contours, color='red', opacity=0.8, name="interface")
        elif self.mode == "volume":
            self.p.remove_actor("volume")
            self.p.add_volume(
                self.grid,
                opacity=[0, 0, 0.3, 0.6, 0.3, 0, 0],
                cmap="coolwarm",
                name="volume"
            )
        elif self.mode == "slices":
            self.p.remove_actor("slices")
            slices = self.grid.slice_orthogonal()
            self.p.add_mesh(slices, cmap="viridis", opacity=0.8, name="slices")
        elif self.mode == "combined":
            self.p.remove_actor("interface")
            self.p.remove_actor("volume")
            contours = self.grid.contour([0.5])
            self.p.add_mesh(contours, color='red', opacity=0.9, name="interface")
            self.p.add_volume(
                self.grid,
                opacity=[0, 0, 0.2, 0.4, 0.2, 0, 0],
                cmap="coolwarm",
                name="volume"
            )

        self.p.app.processEvents()

    def final_plot(self, xi):
        """
        Create a final static plot with the solution.

        Args:
            xi (Function): The final solution
        """
        xi.x.scatter_forward()
        self.grid.point_data[self.name] = xi.x.array[self.dofs].real

        plotter = pv.Plotter()

        if self.mode == "isosurface":
            contours = self.grid.contour([0.5])
            plotter.add_mesh(contours, color='red', opacity=0.8)
        elif self.mode == "volume":
            plotter.add_volume(
                self.grid,
                opacity=[0, 0, 0.3, 0.6, 0.3, 0, 0],
                cmap="coolwarm"
            )
        elif self.mode == "slices":
            slices = self.grid.slice_orthogonal()
            plotter.add_mesh(slices, cmap="viridis", opacity=0.8)
        elif self.mode == "combined":
            contours = self.grid.contour([0.5])
            plotter.add_mesh(contours, color='red', opacity=0.9)
            plotter.add_volume(
                self.grid,
                opacity=[0, 0, 0.2, 0.4, 0.2, 0, 0],
                cmap="coolwarm"
            )

        screenshot = None
        if pv.OFF_SCREEN:
            screenshot = f"{self.name}_3d.png"
        plotter.show(screenshot=screenshot)

