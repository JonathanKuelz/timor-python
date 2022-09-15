from __future__ import annotations

from functools import reduce
from typing import Iterable, List, Optional, Union, Tuple, Collection

import numpy as np
import pinocchio as pin

from timor.utilities import spatial

_TransformationConvertable = Union[
    List[Union[List[float], Tuple[float, float, float, float]]],
    Tuple[Union[List[float], Tuple[float, float, float, float]],
          Union[List[float], Tuple[float, float, float, float]],
          Union[List[float], Tuple[float, float, float, float]],
          Union[List[float], Tuple[float, float, float, float]]],
    np.ndarray,
    List[np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pin.SE3
]
TransformationConvertable = Union[_TransformationConvertable, Iterable[_TransformationConvertable]]


class Projection:
    """A namespace for transformation projections"""

    def __init__(self, transformation: Transformation):
        """Projections are directly defined on transformations."""
        self.transformation = transformation

    @property
    def axis_angles(self) -> np.ndarray:
        """Returns the rotation part of the placement as axis angle representation [n_x, n_y, n_z, theta_R]."""
        return spatial.rot_mat2axis_angle(self.transformation[:3, :3])

    @property
    def cartesian(self) -> np.ndarray:
        """Returns the x, y, z coordinates in a cartesian coordinate system"""
        return self.transformation[:3, 3]

    @property
    def cylindrical(self) -> np.ndarray:
        """Returns the position in cylindrical coordinates [r, phi, z]."""
        return spatial.cartesian2cylindrical(self.cartesian)

    @property
    def spherical(self) -> np.ndarray:
        """Returns the position in spherical coordinates [r, theta, phi]."""
        return spatial.cartesian2spherical(self.cartesian)


class Transformation:
    """A transformation describes a translation and rotation in cartesian space."""

    shape: Tuple[int, int] = (4, 4)

    def __init__(self, transformation: Union[Transformation, TransformationConvertable]):
        """Initialize a spatial transform with a 4x4 matrix which must be a homogeneous transformation.

        :param transformation: A 4x4 homogeneous matrix representing the transformation. Can also be a sequence of
          (4x4) transformations, in which this parameter will be interpreted as the resulting transformation when
          performing all of them after one another.
        """
        if isinstance(transformation, Transformation):
            # The __init__ basically becomes a shallow copy
            transformation = transformation.homogeneous
        elif isinstance(transformation, pin.SE3):
            transformation = transformation.homogeneous
        transformation = np.asarray(transformation, dtype=float)
        if len(transformation.shape) == 3:
            # Unpack a sequence of transformations by multiplying them from left to right
            self.homogeneous = reduce(np.matmul, transformation)
        else:
            self.homogeneous: np.ndarray = transformation

    def multiply_from_left(self, other: np.ndarray) -> Transformation:
        """Multiply a placement from the left.

        Multiplication (matmul) from the right side with a numpy array is easy, as it can be done with this @ other.
        However, other @ this fails due to numpys matmul implementation which cannot be overridden. Use this method
        to multiply from left with a numpy array.
        :param other: A numpy array to multiply.
        :return: A new placement.
        """
        return Transformation(other) @ self  # For transformations, this is implemented as __matmul__

    @classmethod
    def from_translation(cls, p: Collection[float]) -> Transformation:
        """Create a placement from a point with default orientation."""
        T = np.eye(4, dtype=float)
        T[:3, 3] = np.asarray(p, dtype=float)
        return cls(T)

    @classmethod
    def from_rotation(cls, R: Collection[Collection[float]]):
        """Create a placement from a rotation/orientation in the origin."""
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.asarray(R, dtype=float)
        return cls(T)

    @classmethod
    def neutral(cls) -> Transformation:
        """Returns a neutral placement."""
        return cls(np.eye(4))

    @property
    def crok_description(self) -> List[List[float]]:
        """Returns a serialized description if the nominal placement."""
        return self.homogeneous.tolist()

    @property
    def homogeneous(self) -> np.ndarray:
        """The nominal placement as a 4x4 matrix. Alias for placement."""
        return self._transformation

    @property
    def inv(self) -> Transformation:
        """The inverse of the placement."""
        return self.__class__(spatial.inv_homogeneous(self.homogeneous))

    @homogeneous.setter
    def homogeneous(self, T: np.ndarray):
        """Makes sure the placement is a valid homogeneous transformation and make it read-only"""
        self._transformation = T
        self._transformation.setflags(write=False)

    @property
    def projection(self) -> Projection:
        """
        Returns a projection object that can be used to get projections to different conventions for this placement.
        """
        return Projection(self)

    @property
    def translation(self) -> np.ndarray:
        """The translation part of the nominal placement as a 3x1 vector."""
        return self[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        """The rotation part of the nominal placement as a 3x3 matrix."""
        return self[:3, :3]

    def __eq__(self, other):
        """Allows comparison of transformations"""
        if type(other) is type(self):
            return np.allclose(self.homogeneous, other.homogeneous)
        elif isinstance(other, np.ndarray):
            return np.allclose(self.homogeneous, other)
        return NotImplemented

    def __getitem__(self, item):
        """Allows indexing a placement without accessing placement.placement"""
        return self.homogeneous[item]

    def __matmul__(self, other):
        """Allows matrix multiplication with a placement

        :note: Matmul is only possible if the placement is on the left side. Numpy does raise a ValueError instead of
          returning NotImplemented on __matmul__ with placement, so __rmatmul__ would never be called if implemented.
        """
        if isinstance(other, Transformation):
            return self.__class__(self.homogeneous @ other.homogeneous)
        elif isinstance(other, np.ndarray):
            return self.__class__(self.homogeneous @ other)
        return NotImplemented

    def __str__(self):
        """Defaults to numpy"""
        return self.homogeneous.__str__()

    def visualize(self, viz: 'pin.visualize.MeshcatVisualizer', name: str,
                  scale: float = 1., text: Optional[str] = None):
        """
        Draws this placement inside the visualizer object

        :param viz: Visualizer to add placement to
        :param name: label to add geometry to viewer under (overwrites existing!)
        :param scale: Size in meter of each axis of placement
        :param text: Optional descriptive text put next to placement
        """
        import timor.utilities.visualization

        geom = timor.utilities.visualization.drawable_coordinate_system(self.homogeneous, scale=scale)
        viz.viewer[name].set_object(geom)

        if text:
            text_geom = timor.utilities.visualization.scene_text(text, font_size=100, size=scale)
            viz.viewer[f"{name}_text"].set_object(text_geom)
            viz.viewer[f"{name}_text"].set_transform(self.homogeneous)


TransformationLike = Union[Transformation, TransformationConvertable]
