from __future__ import annotations

import abc
from enum import Enum, EnumMeta
from inspect import isclass
import itertools
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

from hppfcl import hppfcl
import meshcat
import numpy as np
import pinocchio as pin

from timor.utilities.spatial import rotX
from timor.utilities.transformation import TransformationLike, Transformation


class HppfclEnumMeta(EnumMeta):
    """Custom Metaclass to alter Enum behavior

    Overriding __getitem__ here allows custom getitem use which makes it possible to map hppfcl geometries on enum
    values.
    """

    def __getitem__(self, item):
        """This getitem is commonly used for enums (when they are not instantiated before use)"""
        if isclass(item) and issubclass(item, hppfcl.CollisionGeometry):
            return {  # Note: This does only work in combination with a subclass that has the required attributes.
                hppfcl.Box: self.BOX,
                hppfcl.Cylinder: self.CYLINDER,
                hppfcl.Sphere: self.SPHERE,
                hppfcl.BVHModelOBBRSS: self.MESH
            }[item]
        return super().__getitem__(item)


class GeometryType(Enum, metaclass=HppfclEnumMeta):
    """Wraps GeometryTapes in an Enum.

    Also contains the custom feature that it can be indexed by hppfcl Geometries due to inheriting from HppfclEnumMeta
    """

    BOX = 'box'
    box = 'box'

    CYLINDER = 'cylinder'
    cylinder = 'cylinder'

    SPHERE = 'sphere'
    sphere = 'sphere'

    MESH = 'mesh'
    mesh = 'mesh'

    COMPOSED = 'composed'
    composed = 'composed'

    EMPTY = 'empty'
    empty = 'empty'


def _geometry_type2class(geometry_type: GeometryType) -> Type[Geometry]:
    """Returns the according Geometry class to represent a GeometryType.

    This needs to be wrapped as a function (cannot be class attribute or similar) because we need external access.

    :param geometry_type: Any of the GeometryType Enum types.
    """
    mapping = {
        GeometryType.BOX: Box,
        GeometryType.CYLINDER: Cylinder,
        GeometryType.SPHERE: Sphere,
        GeometryType.MESH: Mesh,
        GeometryType.COMPOSED: ComposedGeometry,
        GeometryType.EMPTY: EmptyGeometry
    }
    return mapping[GeometryType[geometry_type]]


class Geometry(abc.ABC):
    """A class describing geometries as specified in crok.

    A geometry is defined by a type (see GeometryType), an expansion (defined by custom parameters) and a
    placement, defining its spatial position and orientation.
    """

    type: GeometryType  # Must be overridden by child classes
    hppfcl_representation: hppfcl.CollisionGeometry  # used for visualization and collision checks
    placement: Transformation = Transformation.neutral()  # defines where in the world coordinates this geometry is

    def __init__(self,
                 parameters: Dict[str, Union[float, str]],
                 pose: TransformationLike = Transformation.neutral(),
                 hppfcl_representation: Optional[hppfcl.CollisionGeometry] = None
                 ):
        """
        The geometry class is the default way to describe a volume in space.

        :param parameters: Class-specific parameters defining the volume of the geometry. Will be unpacked in child
          classes
        :param pose: Transformation in world coordinates of this geometry
        :param hppfcl_representation: A hppfcl collision geometry representing this geometry in collision checks. If
          not provided, will be created. If provided, there will be no checks whether it fits the given parameters!
        """
        assert self.type is not None, "Class {} needs to define a GeometryType".format(self.__class__)
        self.parameters = parameters
        self.placement = Transformation(pose)
        if hppfcl_representation is None:
            self.collision_geometry = self._make_collision_geometry()
        else:
            self.collision_geometry = hppfcl_representation

    @classmethod
    def from_crok_description(cls, description: Union[List, Dict[str, any]],
                              package_dir: Optional[Path] = None) -> Geometry:
        """
        Takes a crok geometry specification and returns the according Geometry instance.

        :param description: A crok Geometry as described in the crok documentation
        :param package_dir: The top directory to which the mesh file paths are relative to. Necessary for meshes (only).
        """
        if isinstance(description, (list, tuple, set)):
            if len(description) == 1:
                description = description[0]
            else:
                return ComposedGeometry((cls.from_crok_description(d, package_dir) for d in description))

        if cls is not Geometry:
            raise AttributeError("Class {} does not support this method. Call it on the Geometry class.".format(cls))
        desc = description.copy()
        class_ref = _geometry_type2class(desc.pop('type'))

        if class_ref is Mesh:
            if package_dir is None:
                raise ValueError("If your geometry contains a mesh, you need to provide the package_dir")
            desc['parameters']['package_dir'] = package_dir
            # The "package://" prefix might be given, but is not mandatory.
            desc['parameters']['file'] = re.sub('package://', '', desc['parameters']['file'])
            if not (package_dir / Path(desc['parameters']['file'])).exists():
                raise ValueError("Mesh file {} does not exist".format(desc['parameters']['file']))

        return class_ref(**desc)

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """
        A helper function that transforms a generic hppfcl collision object to an according Geometry instance.

        :param fcl: The hppfcl collision object
        :return: An instance of the according Obstacle class, based on the kind of collision object handed over.
        """
        if cls is not Geometry:  # Enforce inheritance by children
            raise NotImplementedError("from_hppfcl not implemented for class {}".format(cls))

        return _geometry_type2class(GeometryType[type(fcl)]).from_hppfcl(fcl)

    @property
    def collision_data(self) -> Tuple[Tuple[np.ndarray, hppfcl.CollisionGeometry], ...]:
        """Returns a tuple of tuples: (placement_as_np, geometry) needed to perform collision checks with hppfcl"""
        cd = ((self.placement.homogeneous, self.collision_geometry),)  # default is one geometry only
        return cd

    @property
    def crok(self) -> List[Dict[str, any]]:
        """Creates a crok-compliant dictionary of a Geometry"""
        return [{
            'type': self.type.value,
            'parameters': self.parameters,  # This is a dict itself
            'pose': self.placement.crok_description
        }]

    @property
    @abc.abstractmethod
    def parameters(self) -> Dict[str, Union[float, str]]:
        """Returns the class parameters in a crok json compliant format."""

    @parameters.setter
    @abc.abstractmethod
    def parameters(self, parameters: Dict[str, Union[float, str]]):
        """Stores the geometry specific parameters in class attributes"""

    @property
    @abc.abstractmethod
    def urdf_properties(self) -> Tuple[str, Dict[str, Union[float, np.ndarray]]]:
        """Should return a tuple of (urdf_geometry_type, urdf_properties)"""

    @property
    @abc.abstractmethod
    def viz_object(self) -> Tuple[meshcat.geometry.Geometry, Transformation]:
        """Returns a visual representation that can be used by meshcats 'set_object' and the according transform"""

    @abc.abstractmethod
    def _make_collision_geometry(self) -> hppfcl.CollisionGeometry:
        """Creates a hppfcl collision object aligned with the parameters of the Geometry"""

    def __deepcopy__(self, memodict={}):
        """Must be implemented for Geometries, as hppfcl does not natively support it."""
        return self.__class__(self.parameters, self.placement)


class Box(Geometry):
    """A simple geometry with a box extending in x, y and z direction."""

    type: GeometryType = GeometryType.BOX

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """Wraps a hppfcl Box in a Box instance."""
        x, y, z = [2 * hs for hs in fcl.collisionGeometry().halfSide]
        return cls({'x': x, 'y': y, 'z': z}, fcl, fcl.getTransform())

    def _make_collision_geometry(self) -> hppfcl.CollisionGeometry:
        """A hppfcl box is centered in the symmetry axes of the box defined by width, height, depth"""
        return hppfcl.Box(self.x, self.y, self.z)

    @property
    def parameters(self) -> Dict[str, float]:
        """Box parameters are fully described by x, y and z, each one being a float: {'x': f, 'y': f, 'z': f}"""
        return {'x': self.x, 'y': self.y, 'z': self.z}

    @parameters.setter
    def parameters(self, parameters: Dict[str, float]):
        """Unpacks to x, y, z"""
        self._x = parameters['x']
        self._y = parameters['y']
        self._z = parameters['z']

    @property
    def urdf_properties(self) -> Tuple[str, Dict[str, List[float]]]:
        """A box in URDF is described by [x, y, z]"""
        return 'box', {'size': [self.x, self.y, self.z]}

    @property
    def viz_object(self) -> Tuple[meshcat.geometry.Geometry, Transformation]:
        """Returns a meshcat box"""
        return meshcat.geometry.Box([self.x, self.y, self.z]), self.placement

    @property
    def x(self) -> float:
        """Keep access to these attributes private"""
        return self._x

    @property
    def y(self) -> float:
        """Keep access to these attributes private"""
        return self._y

    @property
    def z(self) -> float:
        """Keep access to these attributes private"""
        return self._z


class Cylinder(Geometry):
    """A simple geometry with a cylinder extending in x-y plane (r) and z direction."""

    type: GeometryType = GeometryType.CYLINDER

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """Wraps a hppfcl Cylinder in a Cylinder instance."""
        r = fcl.collisionGeometry().radius
        z = fcl.collisionGeometry().halfLength * 2
        return cls({'r': r, 'z': z}, fcl, fcl.getTransform())

    def _make_collision_geometry(self) -> hppfcl.CollisionGeometry:
        """
        A hppfcl cylinder is centered in the symmetry axes of the cylinder defined by radius and height.
        """
        return hppfcl.Cylinder(self.r, self.z)

    @property
    def parameters(self) -> Dict[str, float]:
        """Cylinder parameters are fully described by r and z: {'r': float, 'z': float}"""
        return {'r': self.r, 'z': self.z}

    @parameters.setter
    def parameters(self, parameters: Dict[str, float]):
        """Unpacks to r, z"""
        self._r = parameters['r']
        self._z = parameters['z']

    @property
    def urdf_properties(self) -> Tuple[str, Dict[str, float]]:
        """A cylinder is specified with a radius and an extension in z-direction with origin in the symmetry axes"""
        return 'cylinder', {'radius': self.r, 'length': self.z}

    @property
    def viz_object(self) -> Tuple[meshcat.geometry.Geometry, Transformation]:
        """The meshcat cylinder is y-aligned while we expect it to be z-aligned"""
        return meshcat.geometry.Cylinder(height=self.z, radius=self.r), self.placement @ rotX(np.pi / 2)

    @property
    def r(self) -> float:
        """Keep access to these attributes private"""
        return self._r

    @property
    def z(self) -> float:
        """Keep access to these attributes private"""
        return self._z


class Sphere(Geometry):
    """A simple geometry with a full sphere."""

    type: GeometryType = GeometryType.SPHERE

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """Wraps a hppfcl Sphere in a Sphere instance."""
        r = fcl.collisionGeometry().radius
        return cls({'r': r}, fcl, fcl.getTransform())

    def _make_collision_geometry(self) -> hppfcl.CollisionGeometry:
        """A hppfcl sphere is centered in the symmetry axes of the sphere defined by radius"""
        return hppfcl.Sphere(self.r)

    @property
    def parameters(self) -> Dict[str, float]:
        """A sphere is defined by its radius only, given as {'r': float}"""
        return {'r': self.r}

    @parameters.setter
    def parameters(self, parameters: Dict[str, float]):
        """Unpacks to r"""
        self._r = parameters['r']

    @property
    def urdf_properties(self) -> Tuple[str, Dict[str, float]]:
        """A sphere is completely described by its radius"""
        return 'sphere', {'radius': self.r}

    @property
    def viz_object(self) -> Tuple[meshcat.geometry.Geometry, Transformation]:
        """Returns a meshcat sphere"""
        return meshcat.geometry.Sphere(self.r), self.placement

    @property
    def r(self) -> float:
        """Keep access to these attributes private"""
        return self._r


class Mesh(Geometry):
    """Meshes can be any geometry represented by a set of triangles"""

    type: GeometryType = GeometryType.MESH
    _filepath: Path
    _scale: Union[float, np.ndarray]
    package_dir: Optional[Path]

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """Wraps a hppfcl Mesh in a Mesh instance."""
        return cls({'file': 0}, fcl, fcl.getTransform())  # TODO: File from hppfcl!

    def _make_collision_geometry(self) -> hppfcl.CollisionGeometry:
        """A hppfcl box is centered in the symmetry axes of the box defined by width, height, depth"""
        mesh_loader = hppfcl.MeshLoader()
        return mesh_loader.load(str(self.abs_filepath), self.scale)

    @property
    def abs_filepath(self) -> Path:
        """The absolute filepath is the combination of package path and relative path of the mesh file."""
        if self.package_dir is None:
            raise AttributeError("package_dir not set")

        return self.package_dir / self.filepath

    @property
    def parameters(self) -> Dict[str, Union[str, float]]:
        """The mesh geometry is the only one that does not return the full parameters: Package dir is not returned!

        The parameters are given as {'file': str, 'scale': [float, array]}, where scale is optional and can be boat,
        a single float or an array of (x, y, z)-scale.
        """
        return {'file': str(self.filepath), 'scale': self.scale.tolist()}  # tolist to allow serialization

    @parameters.setter
    def parameters(self, parameters: Dict[str, any]):
        """Unpacks to filepath and scale - also expects a package_dir to be able to load the mesh."""
        self.package_dir = parameters['package_dir']
        self._filepath = Path(parameters['file'])
        self._scale = np.asarray(parameters.get('scale', 1.0))

    @property
    def urdf_properties(self) -> Tuple[str, Dict[str, Union[str, List[float]]]]:
        """Can only be saved as a dictionary if the mesh was loaded from a file and the file location is still known."""
        return 'mesh', {'filename': 'package://' + str(self.filepath), 'scale': self.scale.tolist()}

    @property
    def viz_object(self) -> Tuple[meshcat.geometry.Geometry, Transformation]:
        """Returns a meshcat mesh"""
        return pin.visualize.meshcat_visualizer.loadMesh(self.collision_geometry), self.placement

    @property
    def filepath(self) -> Path:
        """Keep access to these attributes private"""
        return self._filepath

    @property
    def scale(self) -> np.ndarray:
        """Keep access to these attributes private and make sure scale is always a 3x1 vector (x, y, z) scale"""
        if isinstance(self._scale, float) or self._scale.size == 1:
            return np.array([self._scale, self._scale, self._scale])
        return self._scale


class ComposedGeometry(Geometry):
    """A composed geometry is a collection of geometries that are combined in one instance."""

    type: GeometryType = GeometryType.COMPOSED

    def __init__(self, geometries: Iterable[Geometry]):
        """
        Composed geometries are fundamentally different from basic geometries.

        They are not loaded from a urdf or a single description but rather wrappers around a set of geometries.

        :param geometries: An iterable of geometries that are combined in this composed geometry.
        """
        self._composing_geometries: List[Geometry] = list(g for g in geometries if not isinstance(g, ComposedGeometry))
        self._composing_geometries += list(itertools.chain.from_iterable(
            g.composing_geometries for g in geometries if isinstance(g, ComposedGeometry)))
        self.collision_geometry = None

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """hppfcl does not know the concept of composed geometries."""
        raise NotImplementedError("There are no composed hppfcl geometries.")

    def _make_collision_geometry(self):
        """This method is usually only called in the init, which is overwritten for this class."""
        raise NotImplementedError("ComposedGeometries collision objects are the sum of the internal geometries ones.")

    @property
    def collision_data(self) -> Tuple[Tuple[np.ndarray, hppfcl.CollisionGeometry], ...]:
        """Composed Geometries are the only ones with more than one element in the collision data"""
        return tuple((g.placement.homogeneous, g.collision_geometry) for g in self.composing_geometries)

    @property
    def composing_geometries(self) -> Tuple[Geometry]:
        """This attribute should be read-only, therefore the leading underscore and the return as tuple."""
        return tuple(self._composing_geometries)

    @property
    def crok(self) -> List[Dict[str, any]]:
        """Creates a crok-compliant dictionary of a Geometry"""
        return [{
            'type': g.type.value,
            'parameters': g.parameters,
            'pose': g.placement.crok_description
        } for g in self.composing_geometries]

    @property
    def urdf_properties(self) -> Tuple[str, Dict[str, Union[float, np.ndarray]]]:
        """urdf does not know the concept of composed geometries."""
        raise NotImplementedError("ComposedGeometries cannot be saved as urdf.")

    @property
    def parameters(self) -> Dict[str, Union[float, str]]:
        """This class cannot be instantiated from parameters, so this attribute is also not necessary."""
        raise NotImplementedError("ComposedGeometries don't have their own parameters.")

    @property
    def viz_object(self) -> Tuple[meshcat.geometry.Geometry, Transformation]:
        """meshcat geometries can only represent a single geometry, so this method is not implemented."""
        raise NotImplementedError("ComposedGeometries cannot be visualized.")


class EmptyGeometry(Geometry):
    """This is a dummy geometry to being able to prevent setting a geometry to None."""

    type: GeometryType = GeometryType.EMPTY

    @property
    def parameters(self) -> Dict[str, Union[float, str]]:
        """No Geometry, no parameters."""
        return {}

    @parameters.setter
    def parameters(self, parameters: Dict[str, any]):
        """No Geometry, no parameters."""
        pass

    @property
    def collision_data(self) -> Tuple[Tuple[np.ndarray, hppfcl.CollisionGeometry], ...]:
        """No Geometry, no collision object."""
        return ()

    @property
    def urdf_properties(self) -> Tuple[str, Dict[str, Union[float, np.ndarray]]]:
        """Omit this object when serializing."""
        raise NotImplementedError()

    @property
    def viz_object(self) -> Tuple[meshcat.geometry.Geometry, Transformation]:
        """Omit this object when serializing."""
        raise NotImplementedError()

    def _make_collision_geometry(self) -> hppfcl.CollisionGeometry:
        """No Geometry, no collision object."""
        pass
