from typing import TYPE_CHECKING, Tuple

import moderngl
import numpy as np

from manimlib.constants import GREY, OUT, TAU
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.three_dimensions import Sphere
from manimlib.mobject.types.surface import Surface
from manimlib.utils.bezier import inverse_interpolate
from manimlib.utils.color import get_color_map

if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence, Tuple

    from manimlib.camera.camera import Camera
    from manimlib.typing import ManimColor, Self, Vect3, Vect3Array


class Plane(Surface):
    def __init__(
        self,
        u_range: Tuple[float, float],
        v_range: Tuple[float, float],
        resolution: Tuple[int, int] = (101, 51),
        **kwargs,
    ):
        super().__init__(
            u_range=u_range, v_range=v_range, resolution=resolution, **kwargs
        )
        s = self.data["point"].shape
        v = np.array([0, 0, 1])
        self.data["d_normal_point"] = self.data["point"] + self.normal_nudge * (
            np.zeros(s) + v[None, ...]
        )

    # def uv_func(self, u: float, v: float) -> np.ndarray:
    #     return (u, v, 0.0)


def complex_to_polar(cx_array: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(cx_array, axis=-1)
    phase = np.atan2(cx_array[:, 1], cx_array[:, 0])
    phase *= 1 / TAU
    phase = phase % 1
    phase *= TAU
    return np.stack((r, phase), axis=-1)


def phase_to_rgb(phase: np.ndarray) -> np.ndarray:
    phase *= 1 / TAU
    phase = phase % 1.0
    mask_small = phase < 1 / 3
    mask_medium = (phase >= 1 / 3) & (phase < 2 / 3)
    mask_large = (phase >= 2 / 3) & (phase <= 1.0)

    output = np.zeros((*phase.shape, 3))
    if np.any(mask_small):
        x = phase[mask_small]
        output[mask_small, 0] = 1.0 - 3 * x
        output[mask_small, 1] = 3 * x

    if np.any(mask_medium):
        x = phase[mask_medium]
        output[mask_medium, 1] = 2.0 - 3 * x
        output[mask_medium, 2] = 3 * x - 1.0

    if np.any(mask_large):
        x = phase[mask_large]
        output[mask_large, 2] = 3.0 - 3 * x
        output[mask_large, 0] = 3 * x - 2.0

    return output


def complex_to_rgba(cx_array: np.ndarray):
    cx_polar_array = complex_to_polar(cx_array)
    rgb = phase_to_rgb(cx_polar_array[:, 1])
    opacity = np.exp(-cx_polar_array[:, :1])
    return np.concat((rgb, opacity), axis=-1)


# TODO Make this work for other surfaces, e.g. a flat circle or a region of the plane.
# This might require turning it into a "mixin", or perhaps just a function which can be applied to any Surface object.
#
# To do a subset of an already-implemented surface (such as a subdomain of the plane), just set the
# uncolored pixels to RGBA values (*, *, *, 0). This is hit in the current settings by setting the magnitude to infinity.
#
#
# TODO Write custom "heatmap" shader functions
#
# TODO Test an evolving planar sine wave, for example.
#
# TODO Write convenience functions which convert the relevant arrays to the correct shape.
class HeatMap(Plane):
    """
    An enrichment of Surface where individual vertices are colored their own colors, according to
    a separately-defined array of values stored up at the Python level.
    """

    # TODO Move the computation of the array values down into the shader folder, built from
    # computational primitives.

    render_primitive: int = moderngl.TRIANGLES
    shader_folder: str = "surface"
    data_dtype: np.dtype = np.dtype(
        [
            ("point", np.float32, (3,)),
            ("d_normal_point", np.float32, (3,)),
            ("rgba", np.float32, (4,)),
        ]
    )
    pointlike_data_keys = ["point", "d_normal_point"]
    arr_vals: np.ndarray

    def update_array(self, arr_vals: np.ndarray):
        """Input is an array of shape (N, 2)"""
        self.arr_vals = arr_vals
        self.update_rgba()

    def update_rgba(self):
        new_rgba = complex_to_rgba(self.arr_vals)
        self.data["rgba"] = new_rgba

    # def __init__(
    #     self,
    #     # data: np.ndarray,  # 2D numpy array - store as instance attribute
    #     # color_map: str = "3b1b_colormap",
    #     # vmin: float | None = None,
    #     # vmax: float | None = None,
    #     # width: float = 4.0,
    #     # height: float = 4.0,
    #     **kwargs,
    # ):
    #     self.u_range: Tuple[float, float] = (-1.0, 1.0)
    #     self.v_range: Tuple[float, float] = (-1.0, 1.0)
    #     self.resolution: Tuple[int, int] = (101, 101)

    #     # # Store the 2D array as an instance attribute
    #     # self._data_array = np.asarray(data)

    #     # # Store other parameters as instance attributes
    #     # self.color_map_name = color_map
    #     # self.color_map = get_color_map(color_map)

    #     # # Set value range for color mapping
    #     # self.vmin = vmin if vmin is not None else self._data_array.min()
    #     # self.vmax = vmax if vmax is not None else self._data_array.max()

    #     # # Store dimensions
    #     # self.width = width
    #     # self.height = height

    #     # # Store grid dimensions
    #     # self.rows, self.cols = self._data_array.shape

    #     # Call parent constructor
    #     super().__init__(**kwargs, color=GREY, shading=(0.3, 0.2, 0.4), depth_test=True)

    # @property
    # def data_array(self) -> np.ndarray:
    #     """Get the 2D data array."""
    #     return self._data_array

    # @data_array.setter
    # def data_array(self, value: np.ndarray):
    #     """Set the 2D data array and update the visualization."""
    #     self._data_array = np.asarray(value)
    #     self.rows, self.cols = self._data_array.shape
    #     self.update_points_and_colors()

    # def init_points(self):
    #     """
    #     Create a grid of points based on the data dimensions.
    #     Each data point becomes a vertex in the grid.
    #     """
    #     # Create grid of points in 3D space
    #     x_range = np.linspace(-self.width / 2, self.width / 2, self.cols)
    #     y_range = np.linspace(-self.height / 2, self.height / 2, self.rows)

    #     # Create meshgrid
    #     X, Y = np.meshgrid(x_range, y_range)

    #     # Flatten and combine into points array
    #     points = np.zeros((self.rows * self.cols, 3))
    #     points[:, 0] = X.flatten()  # x coordinates
    #     points[:, 1] = Y.flatten()  # y coordinates
    #     points[:, 2] = 0  # z coordinates (all 0 for 2D)

    #     # Set the points using the parent class method
    #     self.set_points(points)

    #     # Create triangle indices for rendering
    #     self.compute_triangle_indices()

    #     # Set colors based on data values
    #     self.update_colors()

    # def compute_triangle_indices(self):
    #     """Compute triangle indices with correct winding."""
    #     if self.rows < 2 or self.cols < 2:
    #         self.triangle_indices = np.zeros(0, dtype=int)
    #         return

    #     index_grid = np.arange(self.rows * self.cols).reshape((self.rows, self.cols))
    #     indices = np.zeros(6 * (self.rows - 1) * (self.cols - 1), dtype=int)

    #     # Use the same pattern as Surface class
    #     indices[0::6] = index_grid[:-1, :-1].flatten()  # Top left
    #     indices[1::6] = index_grid[+1:, :-1].flatten()  # Bottom left
    #     indices[2::6] = index_grid[:-1, +1:].flatten()  # Top right

    #     indices[3::6] = index_grid[:-1, +1:].flatten()  # Top right
    #     indices[4::6] = index_grid[+1:, :-1].flatten()  # Bottom left
    #     indices[5::6] = index_grid[+1:, +1:].flatten()  # Bottom right

    #     self.triangle_indices = indices

    # def init_shader_data(self):
    #     """Initialize shader-specific data if needed."""
    #     super().init_shader_data()
    #     # Ensure triangle indices are available
    #     if not hasattr(self, "triangle_indices"):
    #         self.compute_triangle_indices()

    # def get_triangle_indices(self) -> np.ndarray:
    #     """Return triangle indices for rendering."""
    #     if not hasattr(self, "triangle_indices"):
    #         self.compute_triangle_indices()
    #     return self.triangle_indices

    # def update_colors(self):
    #     """
    #     Update colors based on data values using the color map.
    #     This updates the 'rgba' field in self.data.
    #     """
    #     if not self.has_points():
    #         return

    #     # Normalize data values to [0, 1] range
    #     normalized = inverse_interpolate(
    #         self.vmin, self.vmax, self._data_array.flatten()
    #     )
    #     normalized = np.clip(normalized, 0, 1)

    #     # Get colors from color map
    #     colors = self.color_map(normalized)

    #     # Set rgba values for all points in self.data
    #     self.data["rgba"][:] = colors

    #     # Mark that data has changed
    #     self.note_changed_data()

    # def update_points_and_colors(self):
    #     """
    #     Update both points and colors when data changes.
    #     """
    #     self.init_points()
    #     self.update_colors()

    # def set_data(
    #     self, new_data: np.ndarray, vmin: float | None = None, vmax: float | None = None
    # ):
    #     """
    #     Update the heatmap with new data.
    #     """
    #     self._data_array = np.asarray(new_data)
    #     self.rows, self.cols = self._data_array.shape

    #     # Update value range if provided
    #     if vmin is not None:
    #         self.vmin = vmin
    #     if vmax is not None:
    #         self.vmax = vmax

    #     # Update the visualization
    #     self.update_points_and_colors()
    #     return self

    # def set_color_map(self, color_map: str):
    #     """
    #     Change the color map used for the heatmap.
    #     """
    #     self.color_map_name = color_map
    #     self.color_map = get_color_map(color_map)
    #     self.update_colors()
    #     return self

    # def set_value_range(self, vmin: float, vmax: float):
    #     """
    #     Set the value range for color mapping.
    #     """
    #     self.vmin = vmin
    #     self.vmax = vmax
    #     self.update_colors()
    #     return self

    # def get_value_at_position(self, x: float, y: float) -> float | None:
    #     """
    #     Get the data value at a specific position in the heatmap.
    #     Returns None if position is outside the heatmap bounds.
    #     """
    #     # Convert position to grid coordinates
    #     grid_x = (x + self.width / 2) / self.width * (self.cols - 1)
    #     grid_y = (y + self.height / 2) / self.height * (self.rows - 1)

    #     # Check bounds
    #     if (
    #         grid_x < 0
    #         or grid_x >= self.cols - 1
    #         or grid_y < 0
    #         or grid_y >= self.rows - 1
    #     ):
    #         return None

    #     # Bilinear interpolation
    #     x0 = int(grid_x)
    #     y0 = int(grid_y)
    #     x1 = min(x0 + 1, self.cols - 1)
    #     y1 = min(y0 + 1, self.rows - 1)

    #     dx = grid_x - x0
    #     dy = grid_y - y0

    #     # Interpolate
    #     top = self._data_array[y0, x0] * (1 - dx) + self._data_array[y0, x1] * dx
    #     bottom = self._data_array[y1, x0] * (1 - dx) + self._data_array[y1, x1] * dx
    #     return top * (1 - dy) + bottom * dy
