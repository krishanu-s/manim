from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Tuple

import moderngl
import numpy as np

from manimlib.constants import GREY, OUT, TAU
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.three_dimensions import Sphere, Torus
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


# For arrays of shape (*, 2), which represent complex-valued functions
def cx_to_polar(cx_array: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(cx_array, axis=-1)
    phase = np.atan2(cx_array[..., 1], cx_array[..., 0])
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


def cx_to_rgba(cx_array: np.ndarray):
    cx_polar_array = cx_to_polar(cx_array)
    rgb = phase_to_rgb(cx_polar_array[..., 1])
    opacity = np.exp(-cx_polar_array[..., :1])
    return np.concat((rgb, opacity), axis=-1)


# Assumed for all intents and purposes to be at infinity
INFINITY = np.array([1e5, 0])


# For nonnegative real heatmaps -- red color. TODO Figure out a color scheme for negative
def real_to_rgba(arr: np.ndarray):
    return np.stack(
        (np.ones_like(arr), np.zeros_like(arr), np.zeros_like(arr), np.exp(-arr)),
        axis=-1,
    )


# Defining operations on complex numbers, in array form
def cx_add(arr_1: np.ndarray, arr_2: np.ndarray) -> np.ndarray:
    return arr_1 + arr_2


def _cx_mult(z: np.ndarray) -> np.ndarray:
    return np.array([z[0] * z[2] - z[1] * z[3], z[0] * z[3] + z[1] * z[2]])


def cx_mult(arr_1: np.ndarray, arr_2: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(
        func1d=_cx_mult, axis=-1, arr=np.concat((arr_1, arr_2), axis=-1)
    )


def _cx_inv(z: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(z)
    if r == 0:
        return INFINITY
    else:
        return np.array([z[0], -z[1]]) / (r**2)


def cx_inv(arr: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(_cx_inv, axis=-1, arr=arr)


def restrict_domain(
    domain_condition: Callable[[np.ndarray], bool],
    domain_points: np.ndarray,
    cx_array: np.ndarray,
) -> np.ndarray:
    domain_mask = np.invert(
        np.apply_along_axis(domain_condition, axis=-1, arr=domain_points.copy())
    )
    new_arr = cx_array.copy()
    if np.any(domain_mask):
        new_arr[domain_mask] = INFINITY
    return new_arr


# TODO How to constrict the domain to a disc? We need an interface for specifying
#

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
# def set_rgba_from_cx_array(mobj: Surface, arr: np.ndarray):
#     """Sets the colors of individual vertices of Surface according to a given array of shape (N, 2)."""
#     mobj.data["rgba"] = cx_to_rgba(arr)


# def set_rgba_from_real_array(mobj: Surface, arr: np.ndarray):
#     """Sets the colors of individual vertices of Surface according to a given array of shape (N,)."""
#     mobj.data["rgba"] = real_to_rgba(arr)


# Note the two classes below are identical, but are separately named for programmer clarity
@dataclass
class ComplexHeatMap:
    """A function f: U -> C, where U is a subset of C."""

    points: np.ndarray  # Array of shape (N, 2) containing the points in the domain
    vals: np.ndarray  # Array of shape (N, 2) containing the values
    domain_condition: Callable[[np.ndarray], bool]

    @classmethod
    def new(
        cls,
        xlims: Tuple[float, float],
        ylims: Tuple[float, float],
        resolution: Tuple[int, int],
    ) -> ComplexHeatMap:
        """Initializes a new ComplexHeatMap with the identity function."""
        xmin, xmax = xlims
        ymin, ymax = ylims
        nx, ny = resolution
        re, im = np.meshgrid(np.linspace(ymin, ymax, ny), np.linspace(xmin, xmax, nx))
        points = np.stack((np.ravel(re), np.ravel(im)), axis=-1)
        vals = np.stack((np.ravel(re), np.ravel(im)), axis=-1)
        return ComplexHeatMap(points, vals, lambda z: True)

    def copy(self) -> ComplexHeatMap:
        """Copies the object."""
        return ComplexHeatMap(self.points, self.vals, self.domain_condition)

    def set_vals(self, vals: np.ndarray):
        """Sets the function values manually from an input array."""
        self.vals = vals

    def set_f(self, f: Callable[[np.ndarray], np.ndarray] | None = None):
        """Sets the function values according to a computable function.
        By default, the identity function is used."""
        if f is None:
            f = lambda x: x

        self.vals = f(self.points)

    def set_domain(self, domain_condition: Callable[[np.ndarray], bool] | None):
        """Sets the domain of the function."""
        if domain_condition is None:
            self.domain_condition = lambda z: True
        else:
            self.domain_condition = domain_condition

    def get_rgba(self) -> np.ndarray:
        """Get RGBA heatmap"""
        mask = np.invert(
            np.apply_along_axis(self.domain_condition, axis=-1, arr=self.points.copy())
        )
        vals = self.vals.copy()
        if np.any(mask):
            vals[mask] = INFINITY

        return cx_to_rgba(vals)


@dataclass
class RealHeatMap:
    """A function f: U -> R, where U is a subset of R^2."""

    points: np.ndarray  # Array of shape (N, 2) containing the points in the domain
    vals: np.ndarray  # Array of shape (N,) containing the values

    @classmethod
    def new(
        xlims: Tuple[float, float],
        ylims: Tuple[float, float],
        resolution: Tuple[int, int],
    ) -> RealHeatMap:
        xmin, xmax = xlims
        ymin, ymax = ylims
        nx, ny = resolution
        re, im = np.meshgrid(np.linspace(ymin, ymax, ny), np.linspace(xmin, xmax, nx))
        points = np.stack((np.ravel(re), np.ravel(im)), axis=-1)
        vals = np.stack(np.zeros((nx * ny,)), axis=-1)
        return RealHeatMap(points, vals)

    def copy(self) -> RealHeatMap:
        return RealHeatMap(self.points, self.vals)

    def restrict_domain(self, domain_condition: Callable[[np.ndarray], bool]):
        mask = np.invert(
            np.apply_along_axis(domain_condition, axis=-1, arr=self.points.copy())
        )
        if np.any(mask):
            self.vals[mask] = INFINITY

    def set_vals(self, vals: np.ndarray):
        """Set output values manually"""
        self.vals = vals

    def set_vals_by_f(self, f: Callable[[np.ndarray], np.ndarray]):
        self.vals = f(self.points)

    def to_rgba(self) -> np.ndarray:
        return real_to_rgba(self.vals)


class HeatMapMixin(Surface):
    """
    An enrichment of Surface where individual vertices are given their own colors, according to
    a separately-defined array of values stored up at the Python level.
    """

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
    heatmap: ComplexHeatMap | RealHeatMap

    # TODO Find a way to animate changing the domain of the current array values: whether that's
    # extending or contracting. Probably can do this when the domain is expanding or contracting
    # from a single point.
    #
    # TODO Make some method which stores the fixed function and all its values underneath, and reveals
    # only the array values on a given domain

    @Mobject.affects_data
    def init_heatmap(self):
        self.heatmap = ComplexHeatMap.new(self.u_range, self.v_range, self.resolution)
        self.update_rgba()
        return self

    @Mobject.affects_data
    def set_f(self, f: Callable[[np.ndarray], np.ndarray] | None):
        """Sets the function values according to a computable function."""
        self.heatmap.set_f(f)
        self.update_rgba()
        return self

    @Mobject.affects_data
    def set_domain(self, domain_condition: Callable[[np.ndarray], bool] | None):
        self.heatmap.set_domain(domain_condition)
        self.update_rgba()
        return self

    @Mobject.affects_data
    def set_vals(self, vals: np.ndarray):
        """Sets the function values manually from an input array."""
        self.heatmap.set_vals(vals)
        self.update_rgba()
        return self

    @Mobject.affects_data
    def update_rgba(self):
        """Updates colors of the MObject."""
        self.data["rgba"] = self.heatmap.get_rgba()
        # if domain_condition:
        #     mask = np.invert(
        #         np.apply_along_axis(
        #             domain_condition, axis=-1, arr=self.heatmap.points.copy()
        #         )
        #     )
        #     vals = self.heatmap.vals.copy()
        #     if np.any(mask):
        #         vals[mask] = INFINITY
        #     # TODO Make this work for real case
        #     self.data["rgba"] = cx_to_rgba(vals)
        # else:
        #     self.data["rgba"] = cx_to_rgba(self.heatmap.vals)


class PlaneHeatMap(HeatMapMixin, Plane):
    pass


class SphereHeatMap(HeatMapMixin, Sphere):
    pass


class TorusHeatMap(HeatMapMixin, Torus):
    pass


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
