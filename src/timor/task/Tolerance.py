from __future__ import annotations

import abc
import itertools
import logging
import re
from typing import Collection, Dict, List, Optional, Set, Tuple, Union
import uuid
from warnings import warn

import meshcat
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

from timor import Volume as TimorVolume
from timor.utilities import spatial
from timor.utilities.transformation import Transformation, TransformationLike


class ToleranceBase(abc.ABC):
    """
    An interface on numeric tolerances between goals and states that can be expressed as np-arrays
    """

    _visual_color = 0x00ff00
    _visual_material = meshcat.geometry.MeshBasicMaterial(color=_visual_color, opacity=0.2, transparent=True)

    @classmethod
    @abc.abstractmethod
    def default(cls):
        """Default tolerance, e.g. for a cartesian tolerance that is 0.001 for all dimensions"""
        pass

    @staticmethod
    def from_projection(projection: str,
                        values: Union[List[float], Tuple[float, float], np.ndarray]) -> ToleranceBase:
        """
        Creates a tolerance from a projection descriptor and an absolute tolerance value.

        :param projection: The projection descriptor. Valid values: {"x", "y", "z", "r_cyl", "r_sph", "theta", "phi",
            "R", "P", "Y", "Alpha", "Beta", "Gamma", "N_x", "N_y", "N_z", "Theta_R", "A", "B", "C", "D"}
        :param values: The lower and upper bound for the tolerance
        :return: The tolerance class instance
        """
        if projection in {"x", "y", "z"}:
            # Cartesian tolerance in any of the three dimensions, unlimited in the other two
            proj_idx = ['x', 'y', 'z'].index(projection)
            xyz_tol = np.ones((3, 2), float) * np.array([-np.inf, np.inf])
            xyz_tol[proj_idx, :] = values
            return CartesianXYZ(*xyz_tol)
        elif projection == "r_cyl":
            return CartesianCylindrical(r=values)
        elif projection == "phi_cyl":
            return CartesianCylindrical(r=[0, np.inf], phi_cyl=values)
        elif projection == "r_sph":
            return CartesianSpheric(r=values)
        elif projection == 'theta':
            return CartesianSpheric(r=[0, np.inf], theta=values)
        elif projection == 'phi_sph':
            return CartesianSpheric(r=[0, np.inf], phi_sph=values)
        elif re.match('N_[xyz]', projection):
            default = {'n_x': [-1, 1], 'n_y': [-1, 1], 'n_z': [-1, 1], 'theta_r': [-np.pi, np.pi]}
            arg = default.copy()
            arg.update({projection.lower(): values})
            return RotationAxisAngle(**arg)
        elif projection == 'Theta_R':
            return RotationAxisAngle.max_abs_rotation(values)
        else:
            raise NotImplementedError(f"Projection {projection} not implemented")

    @abc.abstractmethod
    def _add_same(self, other: ToleranceBase) -> ToleranceBase:
        """Used when two tolerances of the same class are used together.

        Interface to add a tolerance of the same class to this tolerance.
        This method is to be called by __add__ and can simplify the combination of multiple tolerances.
        """
        pass

    @abc.abstractmethod
    def to_projection(self) -> Dict[str, Tuple[Union[str, np.ndarray], ...]]:
        """Describe a tolerance as a projection dictionary

        Returns a dictionary that can be dumped in a json to describe the tolerance
        Inverse operation to from_projection
        """
        pass

    @abc.abstractmethod
    def valid(self,
              desired: Union[np.ndarray, 'TransformationLike'],
              real: Union[np.ndarray, 'TransformationLike']) -> bool:
        """Returns true if <real> is within <desired>'s tolerance (defined by self)"""
        pass

    @property
    def _visual_geometry(self) -> (meshcat.geometry.Geometry, Transformation):
        """The geometry representing the tolerance in a visualization and the relative offset to the nominal position"""
        return None, None

    def visualize(self,
                  viz: MeshcatVisualizer,
                  transform: np.ndarray,
                  name: Optional[str] = None,
                  custom_material: Optional[meshcat.geometry.Material] = None):
        """
        Visualize the tolerance in a MeshcatVisualizer

        :param viz: A meshcat visualizer instance to display the tolerance in.
        :param transform: The nominal position for which the tolerance is specified
        :param name: A display name for the tolerance. Will be set to a random name if not given.
        :param custom_material: Tolerances are displayed in transparent green -- a custom material can change the
          default appearance
        """
        geometry, offset = self._visual_geometry
        if geometry is None:
            logging.debug(f"{self.__class__} does not implement a visualization.")
            return
        if name is None:
            name = f'tol_{self.__class__.__name__}_{uuid.uuid4().hex}'
        if custom_material is None:
            custom_material = self._visual_material
        if hasattr(geometry, 'material'):
            viz.viewer[name].set_object(geometry)
        else:
            viz.viewer[name].set_object(geometry, self._visual_material)
        viz.viewer[name].set_transform(transform @ offset.homogeneous)

    def __add__(self, other: ToleranceBase) -> ToleranceBase:
        """Combine tolerances (logical AND operation)

        Adding two tolerances always yields a stronger constraint than either of the summands
        """
        if not isinstance(other, ToleranceBase):
            return NotImplemented
        if type(other) is type(self):
            return self._add_same(other)
        else:
            return Composed((self, other), simplify_combinations=False)

    def __eq__(self, other):
        """Explicitly fail if not implemented"""
        raise NotImplementedError

    def __hash__(self):
        """Explicitly fail if not implemented"""
        raise NotImplementedError


class AlwaysValidTolerance(ToleranceBase):
    """A tolerance that always evaluates to true"""

    @classmethod
    def default(cls):
        """AlwaysValidTolerance default is always valid - who would've thought?"""
        return cls()

    def _add_same(self, other: AlwaysValidTolerance) -> AlwaysValidTolerance:
        """This should not be necessary. Raising because it would be a bug if this method would ever be called"""
        raise NotImplementedError()

    def to_projection(self) -> Dict[str, Tuple[Union[str, np.ndarray], ...]]:
        """AlwaysValidTolerance is always valid, so no description is needed"""
        return {}

    def valid(self,
              desired: Union[np.ndarray, 'TransformationLike'],
              real: Union[np.ndarray, 'TransformationLike']) -> bool:
        """You don't say"""
        return True

    def visualize(self,
                  viz: MeshcatVisualizer,
                  transform: np.ndarray,
                  name: Optional[str] = None,
                  custom_material: Optional[meshcat.geometry.Material] = None):
        """AlwaysValidTolerance is always valid, so no visualization is needed"""
        pass

    def __add__(self, other: ToleranceBase) -> ToleranceBase:
        """
        AlwaysValidTolerance is always weaker than any other tolerance
        """
        if not isinstance(other, ToleranceBase):
            return NotImplemented
        return other

    def __eq__(self, other):
        """AlwaysValidTolerance is always equal to itself"""
        return isinstance(other, AlwaysValidTolerance)

    def __hash__(self):
        """There's just one instance of this tolerance"""
        return 0


class Composed(ToleranceBase):
    """Holds any number of different tolerances and evaluates to true iff all internal tolerances do"""

    def __new__(cls, tolerances: Collection[ToleranceBase] = None, *args, **kwargs):
        """Custom new method to allow returning the element in tolerances if it only contains one."""
        if tolerances is None:  # tolerances None has to be possible to allow unpickling a Tolerance using __getstate__
            return super().__new__(cls)
        tolerances = tuple(tolerances)  # Make sure not to exhaust a generator and have a __len__ method
        if len(tolerances) == 1:
            return tolerances[0]
        return super().__new__(cls)

    def __init__(self, tolerances: Collection[ToleranceBase], sample_time_limits: float = 10.,
                 simplify_combinations: bool = True):
        """
        A tolerance combining multiple tolerances of various types

        :param tolerances: Any number of tolerance instances
        :param sample_time_limits: The maximum time to find a random valid sample for the composed tolerance
        :param simplify_combinations: If true, compatible tolerances (e.g. 2 of the same type) will be merged
        """
        self._internal = set()
        for tolerance in tolerances:
            self.add(tolerance, simplify_combinations=simplify_combinations)
        self.time_limit = sample_time_limits

    @classmethod
    def default(cls):
        """This cannot be done"""
        raise ValueError("There is no default composed tolerance.")

    @property
    def tolerances(self) -> Set[ToleranceBase]:
        """The internal tolerances"""
        return self._internal

    def _add_same(self, other: Composed) -> Composed:
        """Combine two composed tolerances"""
        return self.__class__(itertools.chain(self._internal, other._internal))

    def to_projection(self) -> Dict[str, Tuple[Union[str, np.ndarray], ...]]:
        """Projects all internal tolerances"""
        internals = [tol.to_projection() for tol in self._internal]
        return {
            'toleranceProjection': tuple(itertools.chain.from_iterable(t['toleranceProjection'] for t in internals)),
            'tolerance': tuple(itertools.chain.from_iterable(t['tolerance'] for t in internals))
        }

    def add(self, other: ToleranceBase, simplify_combinations: bool = True) -> None:
        """
        Add a new tolerance to this composed tolerance. Other than tha add operator (+), this changes the instance

        :param other: The tolerance to add
        :param simplify_combinations: If true, compatible tolerances (e.g. 2 of the same type) will be merged
        """
        if isinstance(other, AlwaysValidTolerance):
            return

        if simplify_combinations:
            for previous in self._internal:
                combined = previous + other
                # This happens for same or compatible tolerances (e.g. CartesianCylindrical, CartesianXYZ)
                if not isinstance(combined, Composed):
                    self._internal.remove(previous)
                    self._internal.add(combined)
                    return

        # The break was not encountered, so the other tolerances cannot be combined with an existing internal
        self._internal.add(other)

    def valid(self,
              desired: Union[np.ndarray, TransformationLike],
              real: Union[np.ndarray, TransformationLike]) -> bool:
        """Is valid if all internal tolerances are valid"""
        return all(tol.valid(desired, real) for tol in self._internal)

    def visualize(self,
                  viz: MeshcatVisualizer,
                  transform: np.ndarray,
                  name: Optional[str] = None,
                  custom_material: Optional[meshcat.geometry.Material] = None):
        """Visualize all internal tolerances, with a slightly different color than the default to distinguish them"""
        if custom_material is None:
            random_color = np.random.randint(0, 0xFFFFFF)
            default_color = self._visual_material.color
            color = (random_color + 19 * default_color) / 20
            custom_material = meshcat.geometry.MeshBasicMaterial(color=color, opacity=0.35, transparent=True)
        if name is None:
            name = f'tol_{self.__class__.__name__}_{uuid.uuid4().hex}'

        for i, tol in enumerate(self._internal):
            tol.visualize(viz, transform, name=f'{name}_{i}', custom_material=custom_material)

    def __eq__(self, other):
        """Two composed tolerances are equal if they contain the same tolerances"""
        if len(self.tolerances) == 1:
            return tuple(self.tolerances)[0] == other
        if not isinstance(other, Composed):
            return NotImplemented
        return self._internal == other._internal

    def __hash__(self):
        """The class is defined by the tolerance bounds that can be hashed"""
        return sum(hash(tol) for tol in self._internal)


class Spatial(ToleranceBase, abc.ABC):
    """Any tolerance that tolerates spatial attributes (placements)."""

    _max_possible: np.ndarray  # Tolerance lower and upper boundaries
    _rounding_error: float = 1e-12  # Valid rounding error for numerical stability

    def __init__(self, *args):
        """Sanity checks"""
        if not all(tol.shape == (2,) for tol in args):
            raise ValueError("Tolerances need to define an interval.")
        if not all(tol[0] <= tol[1] for tol in args):
            raise ValueError("Your lower bounds seem to be larger than the upper bounds.")

    def to_projection(self) -> Dict[str, Tuple[Union[str, np.ndarray], ...]]:
        """
        A dictionary that can be used to specify this tolerance as a tolerated pose projection.

        Format: Dictionary with

            * projection: (proj_0, ..., proj_n) <<Tuple of strings>>,
            * tolerance: (tol_0, ..., tol_n) <<Tuple of 2-tuples of tolerances>>
        """
        ret = self._pose_projection_data
        assert ret['tolerance'].shape == (len(ret['toleranceProjection']), 2)
        tol = ret['tolerance']
        keep = np.where(np.logical_or(tol[:, 0] > self._max_possible[:, 0],
                                      tol[:, 1] < self._max_possible[:, 1]))[0]
        return {'toleranceProjection': tuple(ret['toleranceProjection'][i] for i in keep),
                'tolerance': tuple(ret['tolerance'][i].tolist() for i in keep)}

    def valid(self, desired: TransformationLike, real: TransformationLike) -> bool:
        """Public interface, makes sure child classes only work on (3,) shaped points"""
        desired, real = map(Transformation, (desired, real))
        diff = self._projection(desired.inv @ real).round(16)  # Tries to prevent precision errors close to 0
        return (all(self.stacked[:, 0] - self._rounding_error <= diff)
                and all(self.stacked[:, 1] + self._rounding_error >= diff))

    def _add_same(self, other: Spatial) -> Spatial:
        """Combine two pose tolerances"""
        mask_lower = self.stacked[:, 0] > other.stacked[:, 0]
        mask_upper = self.stacked[:, 1] < other.stacked[:, 1]
        mask = np.vstack((mask_lower, mask_upper)).T
        tolerance = other.stacked
        tolerance[mask] = self.stacked[mask]
        return self.__class__(*tolerance)

    @property
    def valid_random_deviation(self) -> 'Transformation':
        """Used to generate a random but still valid transformation.

        Returns a random deviation/distortion that would still satisfy the tolerance, distributed uniformly in the
        solution space. Useful for sampling valid positions.
        Always returns a Transformation.
        """
        diff = np.mean(np.random.random(self.stacked.shape) * self.stacked, axis=1)
        return Transformation(self._inv_projection(diff))

    @abc.abstractmethod
    def _projection(self, nominal: Transformation) -> np.ndarray:
        """Maps a placement to the projection space of the tolerance"""
        pass

    @abc.abstractmethod
    def _inv_projection(self, projected: np.ndarray) -> Transformation:
        """Inverse to _projection

        Maps a projected element from the tolerances projection space to a placement, being offset from the origin by
        the values defined in the projection space.
        """
        pass

    @property
    @abc.abstractmethod
    def stacked(self) -> np.ndarray:
        """The tolerances stacked in an array of dimensions (num_projections, 2)"""

    @property
    @abc.abstractmethod
    def _pose_projection_data(self) -> Dict[str, Union[Tuple[str], np.ndarray]]:
        """Generate a projection that can be used to describe a tolerance.

        To be implemented by children - returns the data for the projection of the pose.
        The tolerance values should be np arrays though, not lists.
        """

    def __eq__(self, other):
        """Two cartesian tolerances are equal if they have the same tolerance values and dtype"""
        if type(other) != type(self):
            return NotImplemented
        return (self.stacked == other.stacked).all()

    def __hash__(self):
        """The class is defined by the tolerance bounds that can be hashed"""
        return sum(hash(tol) for tol in self.stacked.flat)


class Cartesian(Spatial, abc.ABC):
    """Cartesian Space Tolerances (on positions)"""

    # The maximum limits of the tolerance class
    _max_possible = np.array(
        [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]
    )

    def __init__(self,
                 a: Union[List[float], Tuple[float, float], np.ndarray],
                 b: Union[List[float], Tuple[float, float], np.ndarray],
                 c: Union[List[float], Tuple[float, float], np.ndarray]):
        """Tolerance is valid if all proj(desired-real) is in the interval [tolerances[:, 0], tolerances[:, 1]]"""
        a, b, c = map(np.squeeze, map(np.asarray, (a, b, c)))
        super().__init__(a, b, c)
        self._a, self._b, self._c = a, b, c

    @property
    def stacked(self) -> np.ndarray:
        """The tolerances stacked in a 3x2 array"""
        return np.vstack([self._a, self._b, self._c])


class CartesianXYZ(Cartesian):
    """Axis-aligned cartesian tolerance."""

    def __init__(self,
                 x: Union[List[float], Tuple[float, float], np.ndarray],
                 y: Union[List[float], Tuple[float, float], np.ndarray],
                 z: Union[List[float], Tuple[float, float], np.ndarray]):
        """Defines a box-shaped tolerance

        :param x: Tolerances for x. Usually expected to be of format [negative, positive]
        :param y: Tolerances for y. Usually expected to be of format [negative, positive]
        :param z: Tolerances for z. Usually expected to be of format [negative, positive]
        """
        super().__init__(x, y, z)
        if any(tol[0] > 0 for tol in (x, y, z)):
            logging.warning("Got a positive lower tolerance - are you sure you want to define a positive lower bound?")

    def __add__(self, other: ToleranceBase) -> ToleranceBase:
        """Custom add method to deal with cartesian/cylinder combinations

        Overwrite behavior if the other class is a cartesian with a z-tolerance only: Then it can be seen
        as cylindrical as well.
        """
        if isinstance(other, CartesianCylindrical):
            if np.all(other._max_possible[:2, :] == other.stacked[:2, :]):
                return self._add_same(self.__class__(x=(-np.inf, np.inf), y=(-np.inf, np.inf), z=other.stacked[2, :]))

        return super().__add__(other)

    @classmethod
    def default(cls):
        """Returns a CartesianXYZ with default tolerances"""
        return cls(*[[-1e-3, 1e-3]] * 3)

    def _projection(self, nominal: Transformation) -> np.ndarray:
        """CartesianXYZ works in default cartesian coordinates"""
        return nominal.projection.cartesian

    def _inv_projection(self, projected: np.ndarray) -> Transformation:
        """CartesianXYZ works in default cartesian coordinates"""
        return Transformation.from_translation(projected)

    @property
    def _pose_projection_data(self) -> Dict[str, Union[Tuple[str], np.ndarray]]:
        return {
            'toleranceProjection': ('x', 'y', 'z'),
            'tolerance': self.stacked
        }

    @property
    def _visual_geometry(self) -> (meshcat.geometry.Geometry, Transformation):
        """That's a box"""
        offset = Transformation.from_translation(np.mean(self.stacked, axis=1))
        return meshcat.geometry.Box((self.stacked[:, 1] - self.stacked[:, 0]).tolist()), offset


class CartesianCylindrical(Cartesian):
    """Cylindrical space tolerances (on positions).

    The desired placement must be within a space defined by cylinder coordinates r, z and phi.
    https://en.wikipedia.org/wiki/Cylindrical_coordinate_system
    """

    _max_possible = np.array(
        [[0, np.inf], [-np.pi, np.pi], [-np.inf, np.inf]]
    )

    def __init__(self,
                 r: Union[List[float], Tuple[float, float], np.ndarray],
                 phi_cyl: Union[List[float], Tuple[float, float], np.ndarray] = (-np.pi, np.pi),
                 z: Union[List[float], Tuple[float, float], np.ndarray] = (-np.inf, np.inf)
                 ):
        """Defines a (partial) cylinder as tolerance.

        :param r: Lower and upper tolerance for the radius.
        :param phi_cyl: Lower and upper tolerance for angle of cylinder coordinates between -pi and pi (radian)
        :param z: lower and upper tolerance for the cylinder z coordinates.
        """
        super().__init__(r, phi_cyl, z)
        if not all(np.array(r) >= 0):
            raise ValueError("Can only accept positive radius in cylinder coordinates")
        if all(np.array(z) > 0) or all(np.array(z) < 0):
            warn("Your z tolerances in cylinder coordinates have the same sign. Are you sure you want this?")
        if not phi_cyl[0] >= -np.pi and phi_cyl[1] <= np.pi:
            raise ValueError("Phi must be between -pi and pi")

    def __add__(self, other: ToleranceBase) -> ToleranceBase:
        """Custom add method to deal with cartesian/cylinder combinations

        Overwrite behavior if the other class is a cartesian with a z-tolerance only: Then it can be seen
        as cylindrical as well.
        """
        if isinstance(other, CartesianXYZ):
            if np.all(other._max_possible[:2, :] == other.stacked[:2, :]):
                return self._add_same(self.__class__(r=(0, np.inf), z=other.stacked[2, :]))

        return super().__add__(other)

    @classmethod
    def default(cls):
        """Returns a CartesianCylindrical with default tolerances"""
        return cls(r=[0, 1e-3], phi_cyl=[-np.pi, np.pi], z=[-1e-3, 1e-3])

    def _projection(self, nominal: Transformation) -> np.ndarray:
        """Map cartesian to cylindrical coordinates"""
        return nominal.projection.cylindrical

    def _inv_projection(self, projected: np.ndarray) -> Transformation:
        """Map cylindrical to cartesian coordinates"""
        return Transformation.from_translation(spatial.cylindrical2cartesian(projected))

    @property
    def _pose_projection_data(self) -> Dict[str, Union[Tuple[str], np.ndarray]]:
        return {
            'toleranceProjection': ('r_cyl', 'phi_cyl', 'z'),
            'tolerance': self.stacked
        }

    @property
    def _visual_geometry(self) -> (meshcat.geometry.Geometry, Transformation):
        """Visualize (partial) cylinder"""
        if self._a[0] != 0 or self._b[1] - self._b[0] != 2 * np.pi:
            logging.debug("Visualizing cylindric tolerance as point cloud.")
            N = 10000
            v = TimorVolume.CylinderVolume(limits=self.stacked)
            points = np.stack([v.sample_uniform() for _ in range(N)]).T
            hex_color = '0' * (8 - len(hex(self._visual_color))) + hex(self._visual_color)[2:]
            rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            geometry = meshcat.geometry.PointCloud(points, np.stack((rgb,) * N, axis=1) / 255)
            offset = Transformation.neutral()
        else:
            offset = Transformation(spatial.rotX(np.pi / 2))
            geometry = meshcat.geometry.Cylinder(float(self._c[1] - self._c[0]), float(self._a[1]))
        return geometry, offset


class CartesianSpheric(Cartesian):
    """https://en.wikipedia.org/wiki/Spherical_coordinate_system"""

    _max_possible = np.array(
        [[0, np.inf], [-np.pi, np.pi], [-np.pi, np.pi]]
    )

    def __init__(self,
                 r: Union[List[float], Tuple[float, float], np.ndarray],
                 theta: Union[List[float], Tuple[float, float], np.ndarray] = (0, np.pi),
                 phi_sph: Union[List[float], Tuple[float, float], np.ndarray] = (-np.pi, np.pi)
                 ):
        """Defines a potentially cut out spherical-shaped tolerance

        :param r: Lower and upper tolerance for the radius.
        :param theta: lower and upper tolerance for the inclination angle theta between 0 and 180 degree (in radian).
        :param phi_sph: Lower and upper tolerance for the polar angle phi between -180 and 180 degrees (in radian).
        """
        super().__init__(r, theta, phi_sph)
        if not all(np.array(r) >= 0):
            raise ValueError("Can only accept positive radius in spherical coordinates")
        if not theta[0] >= 0 and theta[1] <= np.pi:
            raise ValueError("Theta must be between 0 and pi")
        if not phi_sph[0] >= -np.pi and phi_sph[1] <= np.pi:
            raise ValueError("Phi must be between -pi and pi")

    @classmethod
    def default(cls):
        """Returns a CartesianSpheric with default tolerances"""
        return cls([0, 1e-3])

    def _projection(self, nominal: Transformation) -> np.ndarray:
        """Map cartesian to spherical coordinates"""
        return nominal.projection.spherical

    def _inv_projection(self, projected: np.ndarray) -> Transformation:
        """Map spherical to cartesian coordinates"""
        return Transformation.from_translation(spatial.spherical2cartesian(projected))

    @property
    def _pose_projection_data(self) -> Dict[str, Union[Tuple[str], np.ndarray]]:
        return {
            'toleranceProjection': ('r_sph', 'theta', 'phi_sph'),
            'tolerance': self.stacked
        }

    @property
    def _visual_geometry(self) -> (meshcat.geometry.Geometry, Transformation):
        """Visualize a (partial) sphere. Meshcat has no support for this, so we use a Mesh to realize it"""
        if self._a[0] != 0 or self._b[1] - self._b[0] != np.pi or self._c[1] - self._c[0] != 2 * np.pi:
            logging.debug("Visualizing spheric tolerance as point cloud.")
            N = 10000
            v = TimorVolume.SphereVolume(limits=self.stacked)
            points = np.stack([v.sample_uniform() for _ in range(N)]).T
            hex_color = '0' * (8 - len(hex(self._visual_color))) + hex(self._visual_color)[2:]
            rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            geometry = meshcat.geometry.PointCloud(points, np.stack((rgb,) * N, axis=1) / 255)
        else:
            geometry = meshcat.geometry.Sphere(float(self._a[1]))
        offset = Transformation.neutral()
        return geometry, offset


class Rotation(Spatial, abc.ABC):
    """Rotation tolerances based on projections of a 4x4 placement - implements projections."""

    _visual_material = meshcat.geometry.PointsMaterial()

    def __init__(self,
                 a: Union[List[float], Tuple[float, float], np.ndarray],
                 b: Union[List[float], Tuple[float, float], np.ndarray],
                 c: Union[List[float], Tuple[float, float], np.ndarray],
                 d: Union[List[float], Tuple[float, float], np.ndarray]):
        """In general, four values suffice to define a rotation tolerance

        :param a: Lower and upper tolerance for the value for the first projection dimension of the rotation.
        :param b: Lower and upper tolerance for the value for the second projection dimension of the rotation.
        :param c: Lower and upper tolerance for the value for the third projection dimension of the rotation.
        :param d: Lower and upper tolerance for the value for the fourth projection dimension of the rotation.
        """
        a, b, c, d = map(np.squeeze, map(np.asarray, (a, b, c, d)))
        super().__init__(a, b, c, d)
        self._a, self._b, self._c, self._d = a, b, c, d

    @property
    def stacked(self) -> np.ndarray:
        """Stacks all tolerances into a single array."""
        return np.vstack((self._a, self._b, self._c, self._d))

    def __eq__(self, other):
        """Two rotation tolerances are equal if they have the same tolerance values"""
        if type(other) != type(self):
            return NotImplemented
        return (self.stacked == other.stacked).all()

    def __hash__(self):
        """The class is defined by the tolerance bounds that can be hashed"""
        return sum(hash(tol) for tol in self.stacked.flat)


class RotationAxisAngle(Rotation):
    """Tolerance on rotations defined in thee axis-angle representation.

    https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    """

    _max_possible = np.array([[-1, 1], [-1, 1], [-1, 1], [-np.pi, np.pi]])

    def __init__(self,
                 n_x: Union[List[float], Tuple[float, float], np.ndarray],
                 n_y: Union[List[float], Tuple[float, float], np.ndarray],
                 n_z: Union[List[float], Tuple[float, float], np.ndarray],
                 theta_r: Union[List[float], Tuple[float, float], np.ndarray]):
        """The axis angle tolerance is defined by four parameters:

        :param n_x: Lower and upper tolerance for the x-part of the unit rotation axis.
        :param n_y: Lower and upper tolerance for the y-part of the unit rotation axis.
        :param n_z: Lower and upper tolerance for the z-part of the unit rotation axis.
        :param theta_r: Lower and upper tolerance for the total rotation in the interval [-pi, pi]. Having theta in the
            interval [0, pi] is sufficient to define every single rotation - however, enforcing it makes it hard to
            define tolerances around theta=0, so we allow negative thetas which leaves us with multiple possible
            solutions to define a desired rotation tolerance.
        """
        n_x, n_y, n_z, theta_r = map(np.asarray, (n_x, n_y, n_z, theta_r))
        mask = abs(theta_r) != np.pi
        theta_r[mask] = ((theta_r + np.pi) % (2 * np.pi) - np.pi)[mask]  # Ensure theta_r is in [-pi, pi]
        if theta_r[0] < 0 and theta_r[1] < 0:
            logging.warning("RotationAxis Tolerances that can be defined using positive theta only should do so.")
        for tolerance in itertools.chain(n_x, n_y, n_z):
            if abs(tolerance) > 1:
                raise ValueError("Cannot interpret a unit vector element tolerance larger than 1.")
        super().__init__(n_x, n_y, n_z, theta_r)

    @classmethod
    def default(cls):
        """Default rotation tolerance:

        The default tolerance is only on theta_r, leading to a maximum tolerance deviation around an
        arbitrary axis.
        """
        half_degree = (np.pi / 180) / 2  # Total tolerance of one degree, 1/2 of a degree in both directions
        return cls.max_abs_rotation(np.array([-half_degree, half_degree]))

    @classmethod
    def max_abs_rotation(cls, theta_r: np.ndarray) -> 'RotationAxisAngle':
        """A utility method to easily create a tolerance on theta_r and an arbitrary axis."""
        return cls((-1, 1), (-1, 1), (-1, 1), theta_r)

    def valid(self, desired: TransformationLike, real: TransformationLike) -> bool:
        """
        Override the default valid check to prevent bugs due to rounding and negative theta_r:

        As long as theta ~=0, the rotation axis is not reliable and therefore not evaluated.
        The calculated diff will always have a positive theta_r - however, this class allows setting it negative aswell,
        so in case theta is negative, we need to check if diff is valid if written as R(-n, -theta), which is the same
        rotation.
        """
        desired, real = map(Transformation, (desired, real))
        diff = self._projection(desired.inv @ real).round(16)  # Tries to prevent precision errors close to 0
        if np.isclose(diff[-1], 0, rtol=0, atol=self._rounding_error):
            # theta_r is almost 0, so the axis is not reliable, and we only check if theta=0 is valid after all
            return self.stacked[-1, 0] <= .0 <= self.stacked[-1, 1]
        valid = (all(self.stacked[:, 0] - self._rounding_error <= diff)
                 and all(self.stacked[:, 1] + self._rounding_error >= diff))
        if not valid and self.stacked[-1, 0] < 0:  # Lower bound for theta_r is negative
            diff = -diff  # R(n, theta) = R(-n, -theta)
            valid = (all(self.stacked[:, 0] - self._rounding_error <= diff)
                     and all(self.stacked[:, 1] + self._rounding_error >= diff))
        return valid

    def _projection(self, nominal: Transformation) -> np.ndarray:
        """4x1 axis-angle representation."""
        projection = nominal.projection.axis_angles
        return projection

    def _inv_projection(self, projected: np.ndarray) -> Transformation:
        """4x1 axis-angle representation -> placement."""
        return Transformation.from_rotation(spatial.axis_angle2rot_mat(projected))

    @property
    def _pose_projection_data(self) -> Dict[str, Union[Tuple[str], np.ndarray]]:
        return {
            'toleranceProjection': ('N_x', 'N_y', 'N_z', 'Theta_R'),
            'tolerance': self.stacked
        }


class Abs6dPoseTolerance(ToleranceBase):
    """An absolute 6d-Vector Tolerance

    Tolerance on the absolute difference between spatial and angular velocities or accelerations. (6D vector)
    This makes no distinction between (d)dx, (d)dy, (d)dz and (d)d_alpha, (d)d_beta, (d)d_gamma - so use with care!
    """

    def __init__(self, abs_tolerance: float):
        """A tolerance for a vector valued array

        :param abs_tolerance: The tolerance value (symmetric)
        """
        self.tolerance = abs_tolerance

    @classmethod
    def default(cls):
        """A default absolute tolerance of 0.001."""
        return cls(1e-3)

    def _stratify(self, v: np.ndarray) -> np.ndarray:
        """Make sure the array is 1-dimensional."""
        return v.reshape((6,))

    def _add_same(self, other: 'Abs6dPoseTolerance') -> 'Abs6dPoseTolerance':
        """Add two tolerances together."""
        return self.__class__(min(self.tolerance, other.tolerance))

    def to_projection(self) -> Dict[str, Tuple[Union[str, np.ndarray], ...]]:
        """There is no tolerance projection for this tolerance yet"""
        raise NotImplementedError("Projection not defined")

    def valid(self, desired: np.ndarray, real: np.ndarray) -> bool:
        """Is valid if the absolute difference between the desired and real is within the tolerance."""
        desired = self._stratify(desired)
        real = self._stratify(real)
        return np.max(abs(real - desired)) <= self.tolerance

    def __eq__(self, other):
        """Two absolute pose tolerances are equal if they have the same tolerance values"""
        if type(other) != type(self):
            return NotImplemented
        return self.tolerance == other.tolerance

    def __hash__(self):
        """The tolerance value fully determines this class"""
        return hash(self.tolerance)


class VectorDistance(ToleranceBase):
    """A tolerance of the LN-Distance between two vectors. Default: L2 / Cartesian distance"""

    def __init__(self, tolerance: float, order: int = 2):
        """
        Create tolerance limiting the distance between two vectors

        :param tolerance: Allowed distance between vectors under given norm
        :param order: Which order of norm to use (default 2 for Cartesian / L2 distance, see "ord" in numpy.linalg.norm)
        """
        if tolerance < 0:
            raise ValueError("VectorDistance needs tolerance >= 0 otherwise would always fail")
        self.tolerance = tolerance
        self.order = order

    @classmethod
    def default(cls):
        """Default is Euclidean / L2 distance with 1e-4 acceptable distance"""
        return cls(tolerance=1e-4)

    def _add_same(self, other: ToleranceBase) -> ToleranceBase:
        if self == other:
            return self
        if isinstance(other, VectorDistance) and ord == other.order:
            return VectorDistance(min(self.tolerance, other.tolerance), self.order)
        return Composed((self, other))

    def to_projection(self) -> Dict[str, Tuple[Union[str, np.ndarray], ...]]:
        """Create CoBRA conformant dict describing this constraint"""
        raise NotImplementedError("Not even specified explicitly for CoBRA")

    def valid(self, desired: Union[np.ndarray], real: Union[np.ndarray]) -> bool:
        """
        Return whether desired and real are at most tolerance apart.

        :param desired: Vector to compare to
        :param real: Vector to judge
        """
        return np.linalg.norm(desired - real, ord=self.order) <= self.tolerance

    def __eq__(self, other):
        """
        Two Vector distances are equal if they use the same norm and acceptable tolerance

        :param other: The other Tolerance to compare to
        """
        if not isinstance(other, VectorDistance):
            raise NotImplementedError(f"Cannot compare to other of type {type(other)}")
        return self.tolerance == other.tolerance and self.order == other.order

    def __hash__(self):
        """Vector distance is unique due to order of norm and acceptable tolerance"""
        return hash((self.order, self.tolerance))


DEFAULT_SPATIAL = Composed((CartesianXYZ.default(), RotationAxisAngle.default()))
