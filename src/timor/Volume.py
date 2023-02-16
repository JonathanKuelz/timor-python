import abc
import itertools
import math
from typing import Sequence, Union

import numpy as np

from timor.utilities import spatial
from timor.utilities.transformation import Transformation, TransformationLike


class Volume(abc.ABC):
    """
    A Volume is a connected sub-set of a three-dimensional euclidean space.
    """

    _rounding_error: float = 1e-12  # Valid rounding error for numerical stability

    def __init__(self, limits: np.ndarray, offset: TransformationLike = Transformation.neutral()):
        """
        Initialize a volume with a transformation offset for every sample.

        :param limits: Limits for defining the shape of the volume -> (3,2)
        :param offset: Transformation offset of the volume with shape -> (4,4)
        """
        self._limits: np.ndarray = np.asarray(limits)
        self._offset: Transformation = Transformation(offset)

    @property
    @abc.abstractmethod
    def limit_rectangular_cuboid(self) -> np.ndarray:
        """
        Get the edge vertex with the extreme value in each dimension for a hyperrectangle over-approximating the volume.

        :return: 2x3 array with the first element for minimal vertex and the second for maximal vertex
        """

    @property
    @abc.abstractmethod
    def sample_uniform(self) -> np.ndarray:
        """
        Uniformly sample a point inside the volume.

        :return: A point as a (3,) np array
        """

    @property
    @abc.abstractmethod
    def vol(self) -> float:
        """Calculate the actual volume of the sampling space"""

    @abc.abstractmethod
    def _projection(self, point: np.ndarray) -> np.ndarray:
        """Maps a point to the projection space of the limits"""
        pass

    def contains(self, r_point: Union[Sequence[float], Transformation]) -> bool:
        """Check whether a given point (3,) locates within the volume"""
        if isinstance(r_point, Transformation):
            r_point = r_point.translation
        else:
            r_point = np.asarray(r_point)

        # transform the point from local based system to global based system
        transformed_point = (spatial.inv_homogeneous(self._offset.homogeneous) @ np.insert(r_point, 3, 1))[:3]
        diff = self._projection(transformed_point)
        return ((diff >= self._limits[:, 0] - self._rounding_error).all()
                and (diff <= self._limits[:, 1] + self._rounding_error).all())

    def discretize(self, grid_size: Sequence[int]) -> np.ndarray:
        """
        Discretize points in grid form inside this volume.

        It should be noticed that a hyperrectangle must surround a sphere or cylinder, which also means there could be
        many points existing in the complementary space between the volume and hyperrectangle. So a contains check is
        a must for the volume subclass. Those not contained in the volume will be given "nan" value.

        :param grid_size: the number of grid along each side of the hyperbox, should be an integer array.
        :return: A (grid_size x grid_size x grid_size x 3) array as a point representing the center of the subgrid.
                 Notice: There could also be "nan" for sphere and cylinder for those fail to pass the contains check.
        """
        grid_size = np.asarray(grid_size)
        grid: np.ndarray = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3))
        grid_len: np.ndarray = (self.limit_rectangular_cuboid[:, 1] - self.limit_rectangular_cuboid[:, 0]) / grid_size

        for (i, j, k) in itertools.product(range(grid_size[0]), range(grid_size[1]), range(grid_size[2])):
            vol_based = np.array((self.limit_rectangular_cuboid[:, 0] + .5 * grid_len)) + np.array((i, j, k)) * grid_len
            global_based = self._offset[:3, :3] @ vol_based + self._offset[:3, 3]
            if self.contains(global_based):
                grid[i, j, k, :] = global_based
            else:
                grid[i, j, k, :] = math.nan
        return grid


class SphereVolume(Volume):
    """
    A spherical volume is an expansion in the three-dimensional euclidean space forming a sphere.
    """

    def __init__(self, limits: np.ndarray, offset: TransformationLike = Transformation.neutral()):
        r"""
        The basic building block of a spherical volume.

        :param limits: Defined by :math:(r, \theta, \phi) with shape (3,2),
                       where, :math:`\theta` is in the range of :math:`(0, \pi).`
                              :math:`\phi` is in the range of :math:`(-\pi, \pi).`
                       --> np.array((lower_bound_r, upper_bound_r),
                                    (lower_bound_theta, upper_bound_theta),
                                    (lower_bound_phi, upper_bound_phi))
        :param offset: Transformation offset of the volume, inherited from super class
        """
        super().__init__(limits, offset)
        if any((limits[1, 0] < 0, limits[1, 1] > np.pi, limits[2, 0] < -np.pi, limits[2, 1] > np.pi)):
            raise ValueError("The given parameter exceeds the boundary of a sphere")

    @classmethod
    def from_radius(cls, r):
        """The limits for the full sphere with radius equal to r"""
        limits = np.array(((0, r), (0, np.pi), (-np.pi, np.pi)))
        return cls(limits)

    @property
    def lim_r(self) -> np.ndarray:
        """The radial limits"""
        return self._limits[0]

    @property
    def lim_theta(self) -> np.ndarray:
        """The azimuthal limits"""
        return self._limits[1]

    @property
    def lim_phi(self) -> np.ndarray:
        """The polar limits"""
        return self._limits[2]

    @property
    def vol(self) -> float:
        """The actual volume of the partial sampling space"""
        return 4 / 3 * np.pi * \
            (self.lim_r[1] ** 3 - self.lim_r[0] ** 3) * \
            (self.lim_theta[1] - self.lim_theta[0]) / np.pi * \
            (self.lim_phi[1] - self.lim_phi[0]) / (2 * np.pi)

    @property
    def limit_rectangular_cuboid(self) -> np.ndarray:
        """
        Return the edge vertices for sphere with the minimal and maximal value in each dimension.

        It should be noticed that this could be quite a bit tighter if the allowed angles are restricted.
        """
        min_hyperrectangle_sphere = -np.ones((3,)) * self._limits[0, 1]
        max_hyperrectangle_sphere = np.ones((3,)) * self._limits[0, 1]
        return np.array((min_hyperrectangle_sphere, max_hyperrectangle_sphere)).T

    def sample_uniform(self) -> np.ndarray:
        """
        Sample points within the partial sphere which are defined by [r, theta, phi] uniformly.

        :source: stats.stackexchange.com/questions/8021
        :return: A point which is sampled uniformly inside the volume
        """
        # r³ is uniformly distributed for a point inside the ball.
        # The cubic root of a Uniform Variable has the same distribution.
        r = np.random.uniform(self.lim_r[0] ** 3, self.lim_r[1] ** 3) ** (1 / 3)

        # The probability that a point lies in a spherical cone is defined by (1-cos(theta))/2.
        # Minus the arc-cosine of a Uniform Variable has the same distribution.
        cos_theta = np.random.uniform(-1 + 2 * (np.pi - self.lim_theta[1]) / np.pi, 1 - 2 * self.lim_theta[0] / np.pi)
        theta = np.arccos(cos_theta)

        # phi is uniformly distributed in (-np.pi, np.pi)
        phi = np.random.uniform(self.lim_phi[0], self.lim_phi[1])

        # transform points from spherical to cartesian coordinate system
        sample_init = spatial.spherical2cartesian(np.array((r, theta, phi)))
        return self._offset.rotation @ sample_init + self._offset.translation

    def _projection(self, point: np.ndarray) -> np.ndarray:
        """Map cartesian to spherical coordinates"""
        return spatial.cartesian2spherical(point)


class CylinderVolume(Volume):
    """
    A cylindrical volume is an expansion in the three-dimensional euclidean space forming a cylinder.
    """

    def __init__(self, limits: np.ndarray, offset: TransformationLike = Transformation.neutral()):
        r"""
        The basic building block of a cylindrical volume.

        :param limits: Defined by :math:(r, \phi, \z) with shape (3,2),
                       where, :math:`\phi` is in the range of :math:`(-\pi, \pi).`
                       --> np.array((lower_bound_r, upper_bound_r),
                                    (lower_bound_phi, upper_bound_phi),
                                    (lower_bound_z, upper_bound_z))
        :param offset: Transformation offset of the volume, inherited from super class
        """
        super().__init__(limits, offset)
        if (limits[1, 0] < -np.pi) or (limits[1, 1] > np.pi):
            raise ValueError("The given parameter exceeds the boundary of a cylinder.")

    @classmethod
    def from_radius_height(cls, r, z):
        """The limits for the full cylinder with radius = r and height = z"""
        limits = np.array(((0, r), (-np.pi, np.pi), (-z / 2, z / 2)))
        return cls(limits)

    @property
    def lim_r(self) -> np.ndarray:
        """The radial limits"""
        return self._limits[0]

    @property
    def lim_phi(self) -> np.ndarray:
        """The azimuthal limits"""
        return self._limits[1]

    @property
    def lim_z(self) -> np.ndarray:
        """The height limits"""
        return self._limits[2]

    @property
    def vol(self) -> float:
        """The actual volume of the partial sampling space"""
        return np.pi * (self.lim_r[1] ** 2 - self.lim_r[0] ** 2) \
            * (self.lim_z[1] - self.lim_z[0]) * (self.lim_phi[1] - self.lim_phi[0]) / (2 * np.pi)

    @property
    def limit_rectangular_cuboid(self) -> np.ndarray:
        """
        Return the edge vertices for cylinder with the minimal and maximal value in each dimension
        """
        min_hyperrectangle_cylinder = np.array((-self._limits[0, 1], -self._limits[0, 1], self._limits[2, 0]))
        max_hyperrectangle_cylinder = np.array((self._limits[0, 1], self._limits[0, 1], self._limits[2, 1]))
        return np.array((min_hyperrectangle_cylinder, max_hyperrectangle_cylinder)).T

    def sample_uniform(self) -> np.ndarray:
        """
        Sample points within the partial cylinder which are defined by [r, theta, height] uniformly.

        :return: A point which is sampled uniformly inside the volume
        """
        # r² is uniformly distributed for a point inside the circle.
        # The square root of a Uniform Variable has the same distribution.
        r = np.random.uniform(self.lim_r[0] ** 2, self.lim_r[1] ** 2) ** (1 / 2)

        # phi is uniformly distributed.
        phi = np.random.uniform(self.lim_phi[0], self.lim_phi[1])

        # z is uniformly distributed along the height.
        z = np.random.uniform(self.lim_z[0], self.lim_z[1])

        # transform points from cylindrical to cartesian coordinate system
        sample_init = spatial.cylindrical2cartesian(np.array((r, phi, z)))
        return self._offset.rotation @ sample_init + self._offset.translation

    def _projection(self, point: np.ndarray) -> np.ndarray:
        """Map cartesian to cylindrical coordinates"""
        return spatial.cartesian2cylindrical(point)


class BoxVolume(Volume):
    """
    A box volume is an expansion in the three-dimensional euclidean space forming a box.
    """

    def __init__(self, limits: np.ndarray, offset: TransformationLike = Transformation.neutral()):
        """
        The basic building block of a box volume.

        :param limits: Defined by math (x, y, z) with shape (3,2).
                       --> np.array((lower_bound_x, upper_bound_x),
                                    (lower_bound_y, upper_bound_y),
                                    (lower_bound_z, upper_bound_z))
        :param offset: Transformation offset of the volume, inherited from super class
        """
        super().__init__(limits, offset)

    @property
    def lim_x(self) -> np.ndarray:
        """Exposes the limits in x-direction"""
        return self._limits[0]

    @property
    def lim_y(self) -> np.ndarray:
        """Exposes the limits in y-direction"""
        return self._limits[1]

    @property
    def lim_z(self) -> np.ndarray:
        """Exposes the limits in z-direction"""
        return self._limits[2]

    @property
    def vol(self) -> float:
        """The actual volume of the partial sampling space"""
        return abs(np.prod(self._limits[:, 1] - self._limits[:, 0]))

    @property
    def limit_rectangular_cuboid(self) -> np.ndarray:
        """
        Return the edge vertices for box with the minimal and maximal value in each dimension.
        """
        min_hyperrectangle_box = np.array((self._limits[0, 0], self._limits[1, 0], self._limits[2, 0]))
        max_hyperrectangle_box = np.array((self._limits[0, 1], self._limits[1, 1], self._limits[2, 1]))
        return np.array((min_hyperrectangle_box, max_hyperrectangle_box)).T

    def sample_uniform(self) -> np.ndarray:
        """
        Sample points within the partial box which are defined by [x, y, z] uniformly.

        :return: A point which is sampled uniformly inside the volume
        """
        sample_init = np.random.uniform(self._limits[:, 0], self._limits[:, 1], size=3)
        return self._offset.rotation @ sample_init + self._offset.translation

    def _projection(self, point: np.ndarray) -> np.ndarray:
        """Keep cartesian coordinates"""
        return point
