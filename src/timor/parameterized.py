from __future__ import annotations

import abc
import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import networkx as nx
import numpy as np
import pinocchio as pin

from timor.Bodies import Body, BodyBase, Connector, ConnectorSet, Gender
from timor.Geometry import Box, ComposedGeometry, Cylinder, Geometry, GeometryType, Sphere, _geometry_type2class
from timor.Joints import Joint, JointSet, TimorJointType
from timor.Module import AtomicModule, ModuleBase, ModuleHeader
from timor.utilities import logging, spatial
from timor.utilities.transformation import Transformation


class ParameterizableBody(BodyBase, abc.ABC):
    """A body that has a geometry which is subject to change depending on an arbitrary number of parameters."""

    parameter_order: Dict[GeometryType, Tuple[str, ...]] = {  # Expected order for the parameters given for different geometry types  # noqa: E501
        GeometryType.BOX: ('x', 'y', 'z'),
        GeometryType.CYLINDER: ('r', 'z'),
        GeometryType.SPHERE: ('r',),
    }
    num_parameters: int  # The number of parameters for this body's geometry
    _parameters: List[float]  # The parameters of the body as a private attribute s.t. setting is safe
    _parameter_limits: np.array = None  # The lower and upper bounds for each parameter
    __geometry_type: GeometryType = None  # The type of the underlying geometry

    def __init__(self,
                 body_id: str,
                 parameters: Iterable[float],
                 parameter_limits: Optional[Sequence[Sequence[float]]] = None,
                 mass_density: float = 1.0,
                 connectors: Iterable[Connector] = (),
                 geometry_placement: Transformation = Transformation.neutral(),
                 in_module: ModuleBase = None
                 ):
        """
        Construct a parameterizable body from a geometry reference and parameters.

        A parameterizable body has a fixed geometry type -- depending on this type, there is a fixed number of
        parameters that define the expansion of the geometry in the respective direction. For example, a sphere
        has a single parameter that defines the radius, a cylinder has two parameters that define the radius and
        the length, and a box has three parameters that define the expansion in x, y, and z direction.

        :param body_id: The unique ID of the body
        :param parameters: The number of parameters depends on the geometry type. Their (positive) value defines the
            expansion of the geometry in the respective direction. For a sphere, this is the radius, for a cylinder,
            this is the radius and the length, and for a box, this is the expansion in x, y, and z direction.
        :param parameter_limits: The lower and upper bounds for each parameter. If None is given (default), the bounds
            are set to [0, inf] for all parameters. Bounds for geometric attributes cannot be negative.
        :param mass_density: The mass density of the body (in kg/m^3), implicitly assuming a homogeneous material.
        :param connectors: The connectors of the body - their placement has to be externally and dynamically updated.
            The connector_placements_valid attribute is used to indicate whether the position of the connectors was
            updated since the parameters have been changed last time.
        :param geometry_placement: Parametrized bodies cannot contain composed geometries, so we need to define a
            relative geometry placement to the origin in order to combine multiple parameterized bodies in a module.
            This parameter defines the relative transformation between geometry origin and body frame.
        :param in_module: The module this body is part of. Will be automatically assigned if the body is defined and
            added to a module later on.
        """
        self._id: str = str(body_id)
        self.mass_density = mass_density

        self._geometry_transformation: Transformation = geometry_placement  # Static part of the geometry placement

        if parameter_limits is None:
            parameter_limits = np.asarray([(0, np.inf) for _ in range(self.num_parameters)])
        self.parameter_limits: np.array = np.array(parameter_limits, dtype=float)

        self.parameters: List[float] = list(parameters)  # Unpacking first in case iterable can be exhausted
        if len(self.parameters) != self.num_parameters:
            raise ValueError(f"Invalid number of parameters {len(self.parameters)}"
                             f" for geometry type {self.geometry_type}!")

        self.in_module: ModuleBase = in_module
        self.connectors: ConnectorSet = ConnectorSet(connectors)
        for connector in self.connectors:
            if connector.parent not in (None, self):
                raise ValueError("Cannot set a new parent for a connector!")
            connector.parent = self

        self.connector_placements_valid: bool = False

    @property
    def geometry_type(self) -> GeometryType:
        """The geometry type of the body. Ensured to be in the set of valid geometry types."""
        return self.__geometry_type

    @property
    def mass(self) -> float:
        """The mass needs to be dynamically calculated, depending on the geometry expansion."""
        return self.mass_density * self.collision.volume

    @property
    def parameters(self) -> Tuple[float, ...]:
        """The parameters of the geometry, returned as a tuple to avoid external changes."""
        return tuple(self._parameters)

    @parameters.setter
    def parameters(self, parameters: Iterable[float]):
        """Set the parameters of the geometry."""
        parameters = list(parameters)  # Unpacking first in case iterable can be exhausted
        if len(parameters) != self.num_parameters:
            raise ValueError(f"Invalid number of parameters {len(parameters)}"
                             f" for geometry type {self.geometry_type}!")
        if np.any(np.asarray(parameters) < self.parameter_limits[:, 0]) \
                or np.any(np.asarray(parameters) > self.parameter_limits[:, 1]):
            raise ValueError(f"Invalid parameters {parameters} for limits {self.parameter_limits}!")
        self._parameters = list(parameters)
        self.connector_placements_valid = False  # Parameters have changed, so the old connector positions are invalid

    @property
    def parameter_limits(self) -> np.array:
        """Returns a copy of the parameter limits of the body."""
        return self._parameter_limits.copy()

    @parameter_limits.setter
    def parameter_limits(self, parameter_limits: Sequence[Sequence[float]]):
        """Set the parameter limits of the geometry."""
        parameter_limits = np.array(parameter_limits, dtype=float)
        if np.any(parameter_limits < 0):  # Enforcing positive limits implicitly enforces positive parameters
            raise ValueError("Geometry parameters cannot be negative, nor can their limits!")
        if parameter_limits.shape != (self.num_parameters, 2):
            raise ValueError(f"Invalid shape of parameter limits {parameter_limits.shape}"
                             f" for geometry type {self.geometry_type}!")
        if np.any(parameter_limits[:, 0] > parameter_limits[:, 1]):
            raise ValueError(f"Invalid parameter limits {parameter_limits}"
                             f" for geometry type {self.geometry_type}!")
        if '_parameters' in self.__dict__:
            if np.any(np.asarray(self._parameters) < parameter_limits[:, 0]) \
                    or np.any(np.asarray(self._parameters) > parameter_limits[:, 1]):
                raise ValueError(f"Invalid limits {parameter_limits} for already set parameters {self._parameters}!")
        self._parameter_limits = parameter_limits

    def freeze(self) -> Body:
        """Freeze the body, i.e. make it immutable by transforming it to an unparameterized body."""
        if not self.connector_placements_valid:
            raise ValueError("Cannot freeze a body before the connector placement has been updated/validated!")
        connectors = [copy.copy(con) for con in self.connectors]
        for con in connectors:
            con._id += '_frozen'
        return Body(self._id,
                    collision=self.collision,
                    visual=None,
                    connectors=connectors,
                    inertia=self.inertia,
                    in_module=self.in_module)

    def to_json_data(self) -> Dict[str, any]:
        """Freezes the current body with the currently set parameters and returns the according JSON"""
        return self.freeze().to_json_data()

    @property
    @abc.abstractmethod
    def collision(self) -> Geometry:
        """
        Returns the momentary collision geometry of the body taking the current parameters into account.

        For parameterizable bodies, the collision geometry is always the same as the visual geometry. As the visual
        defaults to collision anyways, we just need to implement the collision geometry here. We implicitly assume
        that the body frame coincides with the geometry center-of-mass frame.
        """

    @property
    @abc.abstractmethod
    def inertia(self) -> pin.Inertia:
        """The inertia needs to be dynamically calculated, depending on the geometry."""

    def __copy__(self):
        """Parameterizable bodies are actually quite easy to copy as geometries are calculated on-the-fly."""
        return self.__class__(self._id, self.parameters, self.parameter_limits, self.mass_density, in_module=None,
                              connectors=[copy.copy(con) for con in self.connectors])

    def __eq__(self, other):
        """Parameterizable body equality is given by the original parameters - no need to compute the geometry."""
        if not isinstance(other, ParameterizableBody):
            return NotImplemented
        this_attributes = (self._id, tuple(self.parameters), self.mass_density, self._geometry_transformation)
        other_attributes = (other._id, tuple(other.parameters), other.mass_density, other._geometry_transformation)
        return this_attributes == other_attributes

    def __hash__(self):
        """Hashing is equal to hashes of regular bodies - ID should be unique."""
        return super().__hash__()

    def __getstate__(self):
        """For parameterized bodies, we can rely on the python builtin actually"""
        return self.__dict__

    def __setstate__(self, state):
        """For parameterized bodies, we can rely on the python builtin actually"""
        self.__dict__ = state


class ParameterizableSphereBody(ParameterizableBody):
    """A module body composed of a sphere with a parameterizable radius"""

    num_parameters: int = 1
    __geometry_type: GeometryType = GeometryType.SPHERE

    @property
    def r(self) -> float:
        """The radius of the sphere."""
        return self.parameters[0]

    @r.setter
    def r(self, r: float):
        """Set the radius of the sphere."""
        self.parameters = [r]

    @property
    def collision(self) -> Sphere:
        """The collision geometry of the body."""
        return Sphere({'r': self.r}, pose=self._geometry_transformation)

    @property
    def inertia(self) -> pin.Inertia:
        """Assuming a homogeneous material, compute the inertia tensor of self for the current parameters"""
        return pin.Inertia.FromSphere(self.mass, self.r).se3Action(pin.SE3(self._geometry_transformation.homogeneous))


class ParameterizableCylinderBody(ParameterizableBody):
    """A module body composed of a cylinder with a parameterizable radius and length"""

    num_parameters: int = 2
    __geometry_type: GeometryType = GeometryType.CYLINDER

    @property
    def r(self) -> float:
        """The radius of the cylinder."""
        return self.parameters[0]

    @r.setter
    def r(self, r: float):
        """Set the radius of the cylinder."""
        self.parameters = [r, self.l]

    @property
    def l(self) -> float:  # noqa: E741, E743
        """The length of the cylinder."""
        return self.parameters[1]

    @l.setter
    def l(self, l: float):  # noqa: E741, E743
        """Set the length of the cylinder."""
        self.parameters = [self.r, l]

    @property
    def collision(self) -> Cylinder:
        """The collision geometry of the body."""
        return Cylinder({'r': self.r, 'z': self.l}, pose=self._geometry_transformation)

    @property
    def inertia(self) -> pin.Inertia:
        """Assuming a homogeneous material, compute the inertia tensor of self for the current parameters"""
        return pin.Inertia.FromCylinder(self.mass, self.r, self.l).se3Action(
            pin.SE3(self._geometry_transformation.homogeneous))


class ParameterizableBoxBody(ParameterizableBody):
    """A module body composed of a box with parameterizable dimensions"""

    geometry_type: GeometryType = GeometryType.BOX
    num_parameters: int = 3

    @property
    def x(self) -> float:
        """The x dimension of the box."""
        return self.parameters[0]

    @x.setter
    def x(self, x: float):
        """Set the x dimension of the box."""
        self.parameters = [x, self.y, self.z]

    @property
    def y(self) -> float:
        """The y dimension of the box."""
        return self.parameters[1]

    @y.setter
    def y(self, y: float):
        """Set the y dimension of the box."""
        self.parameters = [self.x, y, self.z]

    @property
    def z(self) -> float:
        """The z dimension of the box."""
        return self.parameters[2]

    @z.setter
    def z(self, z: float):
        """Set the z dimension of the box."""
        self.parameters = [self.x, self.y, z]

    @property
    def collision(self) -> Box:
        """The collision geometry of the body."""
        return Box({'x': self.x, 'y': self.y, 'z': self.z}, pose=self._geometry_transformation)

    @property
    def inertia(self) -> pin.Inertia:
        """Assuming a homogeneous material, compute the inertia tensor of self for the current parameters"""
        return pin.Inertia.FromBox(self.mass, self.x, self.y, self.z).se3Action(
            pin.SE3(self._geometry_transformation.homogeneous))


class ParameterizableMultiBody(ParameterizableBody):
    """A parameterizable multi body is a module body made from multiple generic geometries."""

    geometry_type_to_class: Dict[GeometryType, Type[ParameterizableBody]] = {
        GeometryType.SPHERE: ParameterizableSphereBody,
        GeometryType.CYLINDER: ParameterizableCylinderBody,
        GeometryType.BOX: ParameterizableBoxBody
    }
    geometry_type: GeometryType = GeometryType.COMPOSED
    _geometry_transformation: Tuple[Transformation]

    def __init__(self,
                 body_id: str,
                 geometry_types: Sequence[GeometryType],
                 parameters: Iterable[float],
                 parameter_limits: Optional[Sequence[Sequence[float]]] = None,
                 mass_density: float = 1.0,
                 connectors: Iterable[Connector] = (),
                 geometry_placements: Iterable[Transformation] = None,
                 in_module: 'ModuleBase' = None
                 ):
        """
        Initialize a parameterizable body composed of multiple geometries that are placed relative to self's origin.

        A ParameterizableMultiBody can contain an arbitrary number of geometries that are allowed for a
        ParameterizableBody. Each one of them has a placement transformation relative to the origin of the origin. The
        inertia, mass and volume of the body are calculated as the sum of the corresponding values of the geometries.

        :param body_id: The id of the body
        :param geometry_types: The types of the geometries composing this body.
        :param parameters: The parameters of the geometries, flattened and in the order the geometries are given
        :param parameter_limits: The limits of the parameters, flattened and in the same order as parameters (n x 2)
        :param mass_density: The mass density of the body - uniform for all geometries
        :param geometry_placements: The placement transformations between body frame and center of the geometries. They
            are given in the order of geometry_types. Placements do not automatically change when geometry parameters
            do.
        """
        self.composing_geometry_types: Tuple[GeometryType] = tuple(geometry_types)
        self.num_geometries = len(self.composing_geometry_types)
        super().__init__(body_id, parameters, parameter_limits, mass_density, connectors, in_module=in_module)
        if geometry_placements is None:
            geometry_placements = [Transformation.neutral()] * self.num_geometries
        self.geometry_placements = geometry_placements

    @staticmethod
    def get_num_parameters(gt: GeometryType) -> int:
        """Returns the number of parameters necessary to define the given geometry type."""
        return ParameterizableMultiBody.geometry_type_to_class[gt].num_parameters

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters necessary to define all geometries for this body."""
        return sum(self.geometry_type_to_class[geometry_type].num_parameters for geometry_type in
                   self.composing_geometry_types)

    @property
    def collision(self) -> ComposedGeometry:
        """Returns the collision geometry of the body."""
        references: List[Union[Type[Box], Type[Cylinder], Type[Sphere]]] = list()
        params: List[Dict[str, float]] = list()
        for gt in self.composing_geometry_types:
            references.append(_geometry_type2class(gt))
            num_params = sum(len(d) for d in params)
            class_parameters = self.parameters[num_params:num_params + self.get_num_parameters(gt)]
            params.append({name: param for name, param in zip(references[-1].parameter_names, class_parameters)})
        composing = (class_ref(params[i], self._origin_transformation[i]) for i, class_ref in enumerate(references))
        return ComposedGeometry(composing)

    @property
    def geometry_placements(self) -> Tuple[Transformation]:
        """Returns the placement transformations between body frame and geometry reference frames."""
        return tuple(self._origin_transformation)

    @geometry_placements.setter
    def geometry_placements(self, geometry_placements: Sequence[Transformation]):
        """Sets the placement transformations between body frame and geometry reference frames."""
        geometry_placements = tuple(geometry_placements)
        if len(geometry_placements) != self.num_geometries:
            raise ValueError(f"Invalid number of geometry placements {len(geometry_placements)} for "
                             f"{self.num_geometries} geometries!")
        self._origin_transformation = geometry_placements

    @property
    def inertia(self) -> pin.Inertia:
        """The inertia is calculated as the sum of the inertia of the composing geometries."""
        inertia = pin.Inertia.Zero()
        i = 0
        for gt, position in zip(self.composing_geometry_types, self._origin_transformation):
            num_params = self.get_num_parameters(gt)
            intermediate = self.geometry_type_to_class[gt](body_id='intermediate',
                                                           parameters=self.parameters[i:i + num_params],
                                                           mass_density=self.mass_density,
                                                           geometry_placement=position)
            inertia = inertia + intermediate.inertia
            i += num_params
        return inertia

    def print_parameter_info(self):
        """A helper method to show the current parameters and how they are mapped to the underlying geometries."""
        mentioned = 0
        for cg in self.composing_geometry_types:
            num_params = self.get_num_parameters(cg)
            info = ', '.join(name + '=' + str(value) for name, value in
                             zip(self.parameter_order[cg], self.parameters[mentioned:mentioned + num_params]))
            print(f"{cg.name} with parameters ({', '.join(map(str, range(mentioned, mentioned+num_params)))}): {info}")
            mentioned += num_params

    def __copy__(self):
        """Parameterizable bodies are actually quite easy to copy as geometries are calculated on-the-fly."""
        return self.__class__(self._id, self.composing_geometry_types, self.parameters, self.parameter_limits,
                              self.mass_density, [copy.copy(con) for con in self.connectors],
                              self._origin_transformation, in_module=None)


class ParameterizableModule(ModuleBase, abc.ABC):
    """
    A parameterizable module may contain bodies that are subject to change and is non-static until frozen.

    Parameterizable modules can be dynamically resized. The automatic calculation of dynamic properties happens on the
    basis of the inherent parameterizable bodies -- however, as the connector placement must also dynamically be
    re-computed with every resizing, only simple geometries are supported.
    """

    def __init__(self,
                 header: ModuleHeader,
                 bodies: Iterable[BodyBase],
                 joints: Iterable[Joint],
                 parameters: Tuple[float, ...],
                 ):
        """
        Creates a parameterized module.

        :param header: The module header for the generated link module
        :param bodies: The bodies of this module
        :param joints: The joints of this module
        :param parameters: The extension-defining parameters of this module
        """
        super().__init__(header, bodies, joints)
        del self._module_graph  # This has to be dynamic for parameterized modules
        self.resize(parameters)

    @staticmethod
    def _generate_connectors(ids: Tuple[str, str] = ('proximal', 'distal'),
                             genders: Tuple[Gender, Gender] = (Gender.female, Gender.male),
                             types: Tuple[type, type] = ('', ''),
                             sizes: Tuple[float, float] = ((), ())
                             ) -> Tuple[Connector, ...]:
        """Generates the connectors for this module."""
        return tuple(Connector(ids[i], gender=genders[i], connector_type=types[i], size=sizes[i]) for i in range(2))

    @property
    def module_graph(self) -> nx.DiGraph:
        """The module graph for a parameterizable module has to be recomputed with every resize."""
        return self._build_module_graph()

    def copy(self, suffix: str) -> ModuleBase:
        """Copies the module and returns the copy."""
        new_header = ModuleHeader(
            ID=self.id + suffix,
            name=self.name + suffix,
            author=self.header.author, email=self.header.email, affiliation=self.header.affiliation
        )
        return self.__class__(new_header, *self._get_copy_args())

    def freeze(self) -> AtomicModule:
        """Returns a static module representation of this module."""
        frozen_bodies = {body.freeze() for body in self.bodies if isinstance(body, ParameterizableBody)}
        frozen_bodies.update({body for body in self.bodies if not isinstance(body, ParameterizableBody)})
        return AtomicModule(self.header, frozen_bodies, self.joints)

    def to_json_data(self) -> Dict:
        """Freezes the parameters of this module to serialize it."""
        logging.info(f"Serializing parameterizable module {self.id}. Deserializing will return an AtomicModule.")
        return super().to_json_data()

    @property
    @abc.abstractmethod
    def parameters(self) -> Tuple[float, ...]:
        """Returns the parameters of this module."""

    @abc.abstractmethod
    def resize(self, parameters: Tuple[float, ...]) -> None:
        """Resizes the module according to the given parameters and updates the connector placements."""

    @abc.abstractmethod
    def _get_copy_args(self) -> Tuple:
        """Returns the arguments for the copy constructor."""

    @abc.abstractmethod
    def _update_connector_placements(self):
        """Based on the current body parameters, updates the placements of the connectors."""

    def __copy__(self):
        """Deactivates the builtin copy method."""
        super().__copy__()

    def __getstate__(self):
        """For a parameterizable module, we can rely on the default pickling."""
        return self.__dict__

    def __setstate__(self, state):
        """For a parameterizable module, we can rely on the default pickling."""
        self.__dict__.update(state)


class ParameterizedCylinderLink(ParameterizableModule):
    """This is a module in cylinder shape with connectors at both ends that can be used to construct a robot."""

    def __init__(self,
                 header: ModuleHeader,
                 length: float = 1.,
                 radius: float = 1.,
                 mass_density: float = 1.,
                 limits: Optional[Sequence[Sequence[float]]] = None,
                 connector_arguments: Optional[Dict[str, any]] = None,
                 ):
        """
        Creates a parameterized module with one, i-shaped link.

        The link has a varying length and radius. Connectors are placed at x=0, y=0 and z chosen s.t. they are aligned
        with the cylinder's flat surfaces. The body coordinate system is placed in the center of the cylinder.

        :param header: The module header for the generated link module
        :param length: The length of the cylinder
        :param radius: The radius of the cylinder
        :param mass_density: The mass density of the link's material (kg/m^3) -- assumed to be uniform
        :param limits: Optional lower and upper bounds on the link's length and radius
        :param connector_arguments: Mapping from keyword to 2-tuples of values for gender, size, type.
        """
        self._length: float = length
        self._radius: float = radius
        if connector_arguments is None:
            connector_arguments = dict()
        connectors = self._generate_connectors(**connector_arguments)
        self.__connector_arguments = connector_arguments  # store for later use
        link = ParameterizableCylinderBody(body_id=f'{header.ID}_link', parameters=(radius, length),
                                           parameter_limits=limits, mass_density=mass_density, connectors=connectors)
        self.link: ParameterizableCylinderBody = link
        super().__init__(header, bodies=(link,), parameters=(radius, length), joints=())

    @property
    def parameters(self) -> Tuple[float, float]:
        """Returns the parameters of this module."""
        return self.radius, self.length

    @property
    def length(self) -> float:
        """The length of the link."""
        return self._length

    @length.setter
    def length(self, value: float):
        """Sets the length of the link and updates the underlying body."""
        self.resize((self.radius, value))

    @property
    def radius(self) -> float:
        """The radius of the link."""
        return self._radius

    @radius.setter
    def radius(self, value: float):
        """Sets the radius of the link and updates the underlying body."""
        self.resize((value, self.length))

    def resize(self, parameters: Tuple[float, float]) -> None:
        """
        The interface to change the link's length or radius and adapt the connector placements accordingly.

        :param parameters: A tuple of (radius, length)
        """
        self._radius, self._length = parameters
        self.link.parameters = (self._radius, self._length)
        self._update_connector_placements()

    def _get_copy_args(self) -> Tuple:
        """Returns the arguments for the copy constructor."""
        return self.length, self.radius, self.link.mass_density, self.link.parameter_limits, self.__connector_arguments

    def _update_connector_placements(self):
        """Makes sure the connectors are always placed on the bottom/top and centered in the link."""
        dz = self._length / 2.
        connectors = {con.own_id: con for con in self.link.connectors}
        connectors['proximal'].body2connector = Transformation.from_translation((0., 0., -dz)) @ spatial.rotX(np.pi)
        connectors['distal'].body2connector = Transformation.from_translation((0., 0., dz))
        self.link.connector_placements_valid = True


class ParameterizedOrthogonalLink(ParameterizableModule):
    """This is a module in L-shape with connectors at both ends that can be used to construct a robot."""

    def __init__(self,
                 header: ModuleHeader,
                 lengths: Iterable[float] = (1., 1.),
                 radius: float = 1.,
                 mass_density: float = 1.,
                 limits: Sequence[Sequence[float]] = ((0, float('inf')), (0, float('inf')), (0, float('inf'))),
                 connector_arguments: Optional[Dict[str, any]] = None,
                 ):
        """
        Creates a parameterized module with one, L-shaped link.

        The link has a varying lengths and radius. Connectors are placed at the flat surfaces of the cylinders. The body
        coordinate systems share origins with the connectors -- body coordinate systems are pointing inwards though.

        :param header: The module header for the generated link module
        :param lengths: The lengths of the cylinders
        :param radius: The radius of the cylinder
        :param mass_density: The mass density of the link's material (kg/m^3) -- assumed to be uniform
        :param limits: Optional lower and upper bounds on the link's lengths and radius (r, l1, l2)
        :param connector_arguments: Mapping from keyword to 2-tuples of values for gender, size, type.
        """
        self._length1, self._length2 = lengths
        self._radius: float = radius
        limits = np.asarray(limits)
        if connector_arguments is None:
            connector_arguments = dict()
        connectors = self._generate_connectors(**connector_arguments)
        self.__connector_arguments = connector_arguments  # store for later use

        self.link: ParameterizableMultiBody = ParameterizableMultiBody(
            body_id=f'{header.ID}_link_1',
            geometry_types=(GeometryType.CYLINDER, GeometryType.CYLINDER),
            parameters=(radius, self.l1, radius, self.l2),
            parameter_limits=(limits[0], limits[1], limits[0], limits[2]),
            mass_density=mass_density,
            connectors=connectors,
            in_module=self
        )
        super().__init__(header, bodies=(self.link,), parameters=(radius, *lengths), joints=())

    @property
    def parameters(self) -> Tuple[float, float, float]:
        """Returns the parameters of this module."""
        return self.radius, self.l1, self.l2

    @property
    def l1(self) -> float:
        """The length of the first link."""
        return self._length1

    @l1.setter
    def length(self, value: float):
        """Sets the length of the link and updates the underlying body."""
        self.resize((self.radius, value, self.l2))

    @property
    def l1_position(self) -> Transformation:
        """Describes the relative transformation between body origin and the geometry frame of the first geometry."""
        return self.link.geometry_placements[0]

    @property
    def l2(self) -> float:
        """The length of the second (perpendicular) link."""
        return self._length2

    @l2.setter
    def l2(self, value: float):
        """Sets the length of the link and updates the underlying body."""
        self.resize((self.radius, self.l1, value))

    @property
    def l2_position(self) -> Transformation:
        """Describes the relative transformation between body origin and the geometry frame of the second geometry."""
        return self.link.geometry_placements[1]

    @property
    def radius(self) -> float:
        """The radius of the link."""
        return self._radius

    @radius.setter
    def radius(self, value: float):
        """Sets the radius of the link and updates the underlying body."""
        self.resize((value, self.l1, self.l2))

    def resize(self, parameters: Tuple[float, float, float]) -> None:
        """
        The interface to change the link's length or radius and adapt the connector placements accordingly.

        :param parameters: A tuple of (radius, length)
        """
        self._radius, self._length1, self._length2 = parameters
        self.link.parameters = (self.radius, self.l1, self.radius, self.l2)
        self._update_geometry_placements()
        self._update_connector_placements()

    def _get_copy_args(self) -> Tuple:
        """Returns the arguments for the copy constructor."""
        return ((self.l1, self.l2), self.radius, self.link.mass_density, self.link.parameter_limits,
                self.__connector_arguments)

    def _update_connector_placements(self):
        """Makes sure the connectors are always placed on the bottom/top and centered in the link."""
        connectors = {con.own_id: con for con in self.link.connectors}
        connectors['proximal'].body2connector = Transformation.from_translation((0., 0., 0)) @ spatial.rotX(np.pi)
        distal = Transformation.from_translation([0, 0, self.l1]) @ spatial.rotX(-np.pi / 2) @ \
            Transformation.from_translation([0, 0, self.l2])
        connectors['distal'].body2connector = distal
        self.link.connector_placements_valid = True

    def _update_geometry_placements(self):
        """Makes sure the two cylinders composing this link are placed correctly relative to each other.

        Assumes the reference frame for this module is aligned with the proximal connector. The frame for the first
        geometry is centered in the first cylinder, l/2 above the module reference frame. The frame for the second
        geometry is placed accordingly in the second cylinder.
        """
        p1 = Transformation.from_translation([0, 0, self.l1 / 2])
        p2 = Transformation.from_translation([0, 0, self.l1]) @ spatial.rotX(-np.pi / 2) @ \
            Transformation.from_translation([0, 0, self.l2 / 2])
        self.link.geometry_placements = (p1, p2)


class ParameterizedStraightJoint(ParameterizableModule):
    """This is a module in cylinder shape with connectors at both ends and a joint in the middle."""

    def __init__(self,
                 header: ModuleHeader,
                 length: float = 1.,
                 radius: float = 1.,
                 mass_density: float = 1.,
                 limits: Sequence[Sequence[float]] = ((0, float('inf')), (0, float('inf'))),
                 joint_type: TimorJointType = TimorJointType.revolute,
                 joint_parameters: Dict[str, any] = None,
                 connector_arguments: Optional[Dict[str, any]] = None,
                 ):
        r"""
        Creates a parameterized module with two cylinders, connected by a joint.

        The module has a varying length and radius. Connectors are placed at the flat surfaces of the cylinders that are
        not attached to the joint.

        :param header: The module header for the generated module
        :param length: The overall length of the module. Will be divided equally between the two bodies.
        :param radius: The radius of the cylinders
        :param mass_density: The mass density of the bodies' material (kg/m^3) -- assumed to be uniform
        :param limits: Optional lower and upper bounds on the link's length and radius
        :param joint_type: The type of joint to use :math:`\in` (revolute, prismatic)
        :param joint_parameters: Additional arguments for timor.Joint such as limits or gear ratio
        :param connector_arguments: Mapping from keyword to 2-tuples of values
        """
        if joint_parameters is None:
            joint_parameters = dict()
        self.__joint_parameters = joint_parameters  # store for copy later on

        self._length: float = length
        self._radius: float = radius

        if connector_arguments is None:
            connector_arguments = dict()
        connectors = self._generate_connectors(**connector_arguments)
        self.__connector_arguments = connector_arguments  # store for later use
        proximal_link = ParameterizableCylinderBody(body_id=f'{header.ID}_proximal', parameters=(radius, length / 2),
                                                    parameter_limits=limits, mass_density=mass_density,
                                                    connectors=(connectors[0],))
        self.proximal_link: ParameterizableCylinderBody = proximal_link
        distal_link = ParameterizableCylinderBody(body_id=f'{header.ID}_distal', parameters=(radius, length / 2),
                                                  parameter_limits=limits, mass_density=mass_density,
                                                  connectors=(connectors[1],))
        self.distal_link: ParameterizableCylinderBody = distal_link
        self.joint = Joint(joint_id=f'{header.ID}_joint', joint_type=joint_type, parent_body=proximal_link,
                           child_body=distal_link, **joint_parameters)
        super().__init__(header, bodies=(proximal_link, distal_link), joints=(self.joint,), parameters=(radius, length))

    @property
    def parameters(self) -> Tuple[float, float]:
        """The parameters of the module."""
        return self.radius, self.length

    @property
    def length(self) -> float:
        """The length of the module (2 * individual body length)."""
        return self._length

    @length.setter
    def length(self, value: float):
        """Sets the length of the module and updates the underlying bodies."""
        self.resize((self.radius, value))

    @property
    def radius(self) -> float:
        """The radius of the bodies."""
        return self._radius

    @radius.setter
    def radius(self, value: float):
        """Sets the radius of the bodies and updates the underlying bodies."""
        self.resize((value, self.length))

    def resize(self, parameters: Tuple[float, float]) -> None:
        """
        The interface to change both of the bodies' length and/or radius and adapt the connector and joint placements.

        :param parameters: A tuple of (radius, length)
        """
        self._radius, self._length = parameters
        self.proximal_link.parameters = (self.radius, self.length / 2)
        self.distal_link.parameters = (self.radius, self.length / 2)
        self._update_connector_placements()
        self._update_joint_placement()

    def _get_copy_args(self) -> Tuple:
        """Returns the arguments to use when copying this module."""
        return (self.length, self.radius, self.proximal_link.mass_density,
                np.concatenate((self.proximal_link.parameter_limits, self.distal_link.parameter_limits)),
                self.joint.type, self.__joint_parameters, self.__connector_arguments)

    def _update_connector_placements(self):
        """Makes sure the connectors are always placed on the bottom/top and centered in the link."""
        dz = self._length / 4.  # half the length of one of the bodies
        connectors = {con.own_id: con for con in self.proximal_link.connectors | self.distal_link.connectors}
        connectors['proximal'].body2connector = Transformation.from_translation((0., 0., -dz)) @ spatial.rotX(np.pi)
        connectors['distal'].body2connector = Transformation.from_translation((0., 0., dz))
        self.proximal_link.connector_placements_valid = True
        self.distal_link.connector_placements_valid = True

    def _update_joint_placement(self):
        """Makes sure the joint is always placed in the middle of the module."""
        self.joint.parent2joint = Transformation.from_translation((0., 0., self._length / 4.))
        self.joint.joint2child = Transformation.from_translation((0., 0., self._length / 4.))

    def __getstate__(self):
        """A custom getstate allows pickling of the module even though there are circular references."""
        state = self.__dict__.copy()
        state['joint'] = self.joint.to_json_data()
        del state['_joints']
        return state

    def __setstate__(self, newstate):
        """The setstate method ensures pickling works with the custom getstate."""
        newstate['joint'] = Joint.from_json_data(newstate['joint'], {b._id: b for b in newstate['_bodies']})
        newstate['_joints'] = JointSet({newstate['joint']})
        self.__dict__ = newstate


class ParameterizedOrthogonalJoint(ParameterizableModule):
    """This is a module in L-shape with connectors at both ends and a joint in the 'elbow'."""

    def __init__(self,
                 header: ModuleHeader,
                 lengths: Sequence[float, float] = (1., 1.),
                 radius: float = 1.,
                 mass_density: float = 1.,
                 limits: Sequence[Sequence[float]] = ((0, float('inf')), (0, float('inf')), (0, float('inf'))),
                 joint_type: TimorJointType = TimorJointType.revolute,
                 joint_parameters: Dict[str, any] = None,
                 connector_arguments: Optional[Dict[str, any]] = None,
                 ):
        r"""
        Creates a parameterized module with two cylinders being orthogonal to each other.

        The first cylinder is placed along the z-Axis. The symmetry axes of body 2 run through z=l1, where the z-Axis
        of the second cylinder is aligned with the y-axis of the first cylinder for q_joint=0. Both have varying length
        and the same, varying radius. At their connection is a joint, which can be revolute or prismatic. For q_joint=0,
        the surface of the second cylinder is aligned with z=0. The joint axis is the initial z-Axis.

        :param header: The module header for the generated module
        :param lengths: (l1, l2) for the two bodies in this module
        :param radius: The radius of the cylinders
        :param mass_density: The mass density of the bodies' material (kg/m^3) -- assumed to be uniform
        :param limits: Optional lower and upper bounds on the link's length and radius (r, l1, l2)
        :param joint_type: The type of joint to use :math:`\in` (revolute, prismatic)
        :param joint_parameters: Additional arguments for timor.Joint such as limits or gear ratio
        :param connector_arguments: Mapping from keyword to 2-tuples of values
        """
        if joint_parameters is None:
            joint_parameters = dict()
        self.__joint_parameters = joint_parameters  # store for copy later on

        self._length1, self._length2 = lengths
        self._radius: float = radius

        if connector_arguments is None:
            connector_arguments = dict()
        connectors = self._generate_connectors(**connector_arguments)
        self.__connector_arguments = connector_arguments  # store for later use
        proximal_link = ParameterizableCylinderBody(body_id=f'{header.ID}_proximal', parameters=(radius, self.l1),
                                                    parameter_limits=(limits[0], limits[1]), mass_density=mass_density,
                                                    connectors=(connectors[0],))
        self.proximal_link: ParameterizableCylinderBody = proximal_link
        distal_link = ParameterizableCylinderBody(body_id=f'{header.ID}_distal', parameters=(radius, self.l2),
                                                  parameter_limits=(limits[0], limits[2]), mass_density=mass_density,
                                                  connectors=(connectors[1],))
        self.distal_link: ParameterizableCylinderBody = distal_link
        self.joint = Joint(joint_id=f'{header.ID}_joint', joint_type=joint_type, parent_body=proximal_link,
                           child_body=distal_link, **joint_parameters)
        super().__init__(header, bodies=(proximal_link, distal_link), joints=(self.joint,),
                         parameters=(radius, self.l1, self.l2))

    @property
    def parameters(self) -> Tuple[float, float, float]:
        """The parameters of the module."""
        return self.radius, self.l1, self.l2

    @property
    def l1(self) -> float:
        """The length of the proximal body."""
        return self._length1

    @l1.setter
    def l1(self, value: float):
        """Sets the length of the proximal body."""
        self.resize((self.radius, value, self.l2))

    @property
    def l2(self) -> float:
        """The length of the distal body."""
        return self._length1

    @l1.setter
    def l1(self, value: float):
        """Sets the length of the distal body."""
        self.resize((self.radius, self.l1, value))

    @property
    def radius(self) -> float:
        """The radius of the bodies."""
        return self._radius

    @radius.setter
    def radius(self, value: float):
        """Sets the radius of the bodies and updates the underlying bodies."""
        self.resize((value, self.l1, self.l2))

    def resize(self, parameters: Tuple[float, float, float]) -> None:
        """
        The interface to change both of the bodies' length and/or radius and adapt the connector and joint placements.

        :param parameters: A tuple of (radius, l1, l2)
        """
        self._radius, self._length1, self._length2 = parameters
        self.proximal_link.parameters = (self.radius, self.l1)
        self.distal_link.parameters = (self.radius, self.l2)
        self._update_connector_placements()
        self._update_joint_placement()

    def _get_copy_args(self) -> Tuple:
        """Returns the arguments to use when copying this module."""
        return ((self.l1, self.l2), self.radius, self.proximal_link.mass_density,
                np.concatenate((self.proximal_link.parameter_limits, self.distal_link.parameter_limits)),
                self.joint.type, self.__joint_parameters, self.__connector_arguments)

    def _update_connector_placements(self):
        """Makes sure the connectors are always placed on the bottom/top and centered in the link."""
        connectors = {con.own_id: con for con in self.proximal_link.connectors | self.distal_link.connectors}
        connectors['proximal'].body2connector = Transformation.from_translation((0., 0., -self.l1 / 2.)) @ \
            spatial.rotX(np.pi)
        connectors['distal'].body2connector = Transformation.from_translation((0., 0., self.l2 / 2.))
        self.proximal_link.connector_placements_valid = True
        self.distal_link.connector_placements_valid = True

    def _update_joint_placement(self):
        """Makes sure the joint is always placed in the middle of the module."""
        self.joint.parent2joint = Transformation.from_translation((0., 0., self.l1 / 2.))
        self.joint.joint2child = Transformation.from_translation((0., 0., self.l2 / 2.)).multiply_from_left(
            spatial.rotX(-np.pi / 2))

    def __getstate__(self):
        """A custom getstate allows pickling of the module even though there are circular references."""
        state = self.__dict__.copy()
        state['joint'] = self.joint.to_json_data()
        del state['_joints']
        return state

    def __setstate__(self, newstate):
        """The setstate method ensures pickling works with the custom getstate."""
        newstate['joint'] = Joint.from_json_data(newstate['joint'], {b._id: b for b in newstate['_bodies']})
        newstate['_joints'] = JointSet({newstate['joint']})
        self.__dict__ = newstate
