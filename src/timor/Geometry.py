from __future__ import annotations

import abc
from enum import Enum, EnumMeta
import importlib.util
from inspect import isclass
import io
import itertools
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

from hppfcl import hppfcl
import meshcat
import numpy as np
import pinocchio as pin

from timor.utilities import logging
from timor.utilities.jsonable import JSONable_mixin
from timor.utilities.spatial import rotX
from timor.utilities.transformation import Transformation, TransformationLike


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
    """Wraps GeometryTypes in an Enum.

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
    if isinstance(geometry_type, GeometryType):
        return mapping[geometry_type]
    else:
        return mapping[GeometryType[geometry_type]]


class Geometry(abc.ABC, JSONable_mixin):
    """A class describing geometries.

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

    def to_json_string(self) -> str:
        """Returns a json string representation of this geometry."""
        return json.dumps(self.serialized, indent=2)

    @classmethod
    def from_json_data(cls, d: Union[List, Dict[str, any]], package_dir: Optional[Path] = None,
                       *args, **kwargs) -> Geometry:
        """
        Takes a serialized geometry specification and returns the according Geometry instance.

        :param d: A serialized geometry
        :param package_dir: The top directory to which the mesh file paths are relative to. Necessary for meshes (only).
        """
        if isinstance(d, (list, tuple, set)):
            if len(d) == 1:
                d = d[0]
            else:
                return ComposedGeometry((cls.from_json_data(d, package_dir) for d in d))

        if cls is not Geometry:
            raise AttributeError("Class {} does not support this method. Call it on the Geometry class.".format(cls))
        desc = d.copy()
        class_ref = _geometry_type2class(desc.pop('type'))

        if class_ref is Mesh:
            if package_dir is not None:
                # The "package://" prefix might be given, but is not mandatory.
                desc['parameters']['file'] = re.sub('package://', '', desc['parameters']['file'])
                desc['parameters']['file'] = package_dir / Path(desc['parameters']['file'])

            if not (Path(desc['parameters']['file'])).exists():
                raise FileNotFoundError("Mesh file {} does not exist".format(desc['parameters']['file']))

        return class_ref(**desc)

    @classmethod
    def from_json_string(cls, description: str, package_dir: Optional[Path] = None) -> Geometry:
        """
        Takes a serialized geometry specification and returns the according Geometry instance.

        :param description: A serialized geometry
        :param package_dir: The top directory to which the mesh file paths are relative to. Necessary for meshes (only).
        """
        return cls.from_json_data(json.loads(description), package_dir)

    @classmethod
    @abc.abstractmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """
        A helper function that transforms a generic hppfcl collision object to an according Geometry instance.

        :param fcl: The hppfcl collision object
        :return: An instance of the according Obstacle class, based on the kind of collision object handed over.
        """

    @classmethod
    def from_pin_geometry(cls, geo: pin.GeometryObject) -> Geometry:
        """
        A helper function that transforms a generic pinocchio GeometryObject to an according Geometry instance.

        :param geo: The pinocchio GeometryObject
        :return: An instance of the according Obstacle class, based on the kind of collision object handed over.
        """
        if cls is not Geometry:
            raise ValueError("Call from_pin_geometry on the Geometry class, not on a subclass.")
        fcl = geo.geometry
        will_be_instance_of = _geometry_type2class(GeometryType[type(fcl)])
        intermediate = will_be_instance_of.from_hppfcl(hppfcl.CollisionObject(fcl, geo.placement.rotation,
                                                                              geo.placement.translation))
        if isinstance(intermediate, Mesh):
            logging.debug("Restoring mesh file information from pinocchio geometry object")
            params = intermediate.parameters
            params['file'] = geo.meshPath
            params['scale'] = geo.meshScale
            intermediate.parameters = params  # Trigger setter
        return intermediate

    @property
    def as_hppfcl_collision_object(self) -> Tuple[hppfcl.CollisionObject, ...]:
        """Creates hppfcl CollisionObject(s) at the current pose of self."""
        return hppfcl.CollisionObject(self.collision_geometry, self.placement.as_transform3f()),

    @property
    def collision_data(self) -> Tuple[Tuple[np.ndarray, hppfcl.CollisionGeometry], ...]:
        """Returns a tuple of tuples: (placement_as_np, geometry) needed to perform collision checks with hppfcl"""
        cd = ((self.placement.homogeneous, self.collision_geometry),)  # default is one geometry only
        return cd

    @property
    def serialized(self) -> List[Dict[str, any]]:
        """Creates a serialization of a Geometry that can be stored as json"""
        return [{
            'type': self.type.value,
            'parameters': self.parameters,  # This is a dict itself
            'pose': self.placement.to_json_data()
        }]

    @property
    @abc.abstractmethod
    def parameters(self) -> Dict[str, Union[float, str]]:
        """Returns the class parameters in a json compliant format."""

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

    @property
    @abc.abstractmethod
    def volume(self) -> float:
        """Returns the volume of the geometry"""

    @abc.abstractmethod
    def _make_collision_geometry(self) -> hppfcl.CollisionGeometry:
        """Creates a hppfcl collision object aligned with the parameters of the Geometry"""

    def __deepcopy__(self, memodict={}):
        """Must be implemented for Geometries, as hppfcl does not natively support it."""
        return self.__class__(self.parameters, self.placement)

    def __eq__(self, other: Geometry):
        """Compares two Geometry instances based on their type, parameters and placement."""
        if isinstance(other, Geometry) and not isinstance(other, Mesh):
            this_parameters = tuple(sorted(((k, v) for k, v in self.parameters.items()), key=lambda x: x[0]))
            other_parameters = tuple(sorted(((k, v) for k, v in other.parameters.items()), key=lambda x: x[0]))
            return self.type == other.type and this_parameters == other_parameters and self.placement == other.placement
        else:
            return NotImplemented

    def __hash__(self):
        """Hashes the Geometry instance based on its type, parameters and placement."""
        return hash((self.type, tuple(hash(p) for p in self.parameters.items()), self.placement.to_json_string()))


class Box(Geometry):
    """A simple geometry with a box extending in x, y and z direction."""

    parameter_names: Tuple[str, str, str] = ('x', 'y', 'z')
    type: GeometryType = GeometryType.BOX

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """Wraps a hppfcl Box in a Box instance."""
        x, y, z = [2 * hs for hs in fcl.collisionGeometry().halfSide]
        return cls({'x': x, 'y': y, 'z': z}, fcl.getTransform(), fcl.collisionGeometry())

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
    def volume(self) -> float:
        """Returns the volume of the box"""
        return self.x * self.y * self.z

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

    parameter_names: Tuple[str, str] = ('r', 'z')
    type: GeometryType = GeometryType.CYLINDER

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """Wraps a hppfcl Cylinder in a Cylinder instance."""
        r = fcl.collisionGeometry().radius
        z = fcl.collisionGeometry().halfLength * 2
        return cls({'r': r, 'z': z}, fcl.getTransform(), fcl.collisionGeometry())

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
        try:
            self._r = parameters['r']
        except KeyError:
            self._r = parameters['radius']
        try:
            self._z = parameters['z']
        except KeyError:
            self._z = parameters['length']

    @property
    def urdf_properties(self) -> Tuple[str, Dict[str, float]]:
        """A cylinder is specified with a radius and an extension in z-direction with origin in the symmetry axes"""
        return 'cylinder', {'radius': self.r, 'length': self.z}

    @property
    def viz_object(self) -> Tuple[meshcat.geometry.Geometry, Transformation]:
        """The meshcat cylinder is y-aligned while we expect it to be z-aligned"""
        return meshcat.geometry.Cylinder(height=self.z, radius=self.r), self.placement @ rotX(np.pi / 2)

    @property
    def volume(self) -> float:
        """Returns the volume of the cylinder"""
        return np.pi * self.r ** 2 * self.z

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

    parameter_names: Tuple[str] = ('r',)
    type: GeometryType = GeometryType.SPHERE

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """Wraps a hppfcl Sphere in a Sphere instance."""
        r = fcl.collisionGeometry().radius
        return cls({'r': r}, fcl.getTransform(), fcl.collisionGeometry())

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
    def volume(self) -> float:
        """Returns the volume of the sphere"""
        return 4 / 3 * np.pi * self.r ** 3

    @property
    def r(self) -> float:
        """Keep access to these attributes private"""
        return self._r


class Mesh(Geometry):
    """Meshes can be any geometry represented by a set of triangles"""

    type: GeometryType = GeometryType.MESH
    _filepath: Path
    _scale: Union[float, np.ndarray]
    _colors: Optional[np.ndarray] = None

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """Wraps a hppfcl Mesh in a Mesh instance."""
        logging.debug("Creating mesh from hppfcl, will lose possible file location information")
        return cls({'file': ''}, fcl.getTransform(), fcl.collisionGeometry())

    @staticmethod
    def read_wrl(file: Path) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        Reads each sub-mesh from a wrl file into separate vertex and face arrays.

        The WRL format describes triangulated meshes and can contain multiple sub meshes that are parsed separately.
        We use these to store simplified convex decompositions of collision meshes.
        They are created with https://github.com/gaschler/bounding-mesh

        Based on example here: https://en.wikipedia.org/wiki/VRML
        """
        with open(file, "r") as f:
            buffer = f.read()
        if not buffer.startswith("#VRML V2.0 utf8"):
            raise ValueError("wrl format not as expected")

        shapes = re.findall(r'(?<=Shape {)[\n\t\w {.}\[\],-]*?(?=}\s*Shape|}\s*$)', buffer)
        faces = tuple(re.findall(r'(?<=coordIndex \[)[0-9\s,-]*?(?=,?\s*])', s) for s in shapes)
        vertices = tuple(re.findall(r'(?<=point \[)[e0-9.\s,-]*?(?=,?\s*])', s) for s in shapes)
        if not all(len(f) == len(v) == 1 for f, v in zip(faces, vertices)):
            raise ValueError("Every shape should have one face and vertex array.")

        faces = tuple(np.genfromtxt(io.BytesIO(f.strip().replace("\t", "").replace(",\n", "\n").encode()),
                                    dtype=int, delimiter=",")[:, :3] for f in itertools.chain(*faces))
        vertices = tuple(np.genfromtxt(io.BytesIO(v.strip().replace("\t", "").replace(", \n", "\n").encode()),
                                       dtype=float, delimiter=" ") for v in itertools.chain(*vertices))

        if not all(np.all(f <= v.shape[0]) for f, v in zip(faces, vertices)):
            raise ValueError("Faces should be valid indices in vertex")
        return vertices, faces

    def _make_collision_geometry(self) -> hppfcl.CollisionGeometry:
        """This method generates the hppfcl-based collision geometry for this mesh used for collision detection."""
        if self.abs_filepath.suffix == ".wrl":
            vertices, faces = self.read_wrl(self.abs_filepath)
            geom = hppfcl.BVHModelOBBRSS()
            num_tris = sum(f.shape[0] for f in faces)
            geom.beginModel(num_tris, 3 * num_tris)
            for vs, fs in zip(vertices, faces):
                for f in fs:
                    geom.addTriangle(vs[f[0], :], vs[f[1], :], vs[f[2], :])
            geom.endModel()
            return geom
        else:
            mesh_loader = hppfcl.MeshLoader()
            if self.abs_filepath.suffix == ".ply":
                if importlib.util.find_spec("trimesh"):
                    import trimesh
                    m = trimesh.load(self.abs_filepath)
                    # Map uint8 to 0. .. 1. float value; cut of alpha
                    self._colors = m.visual.vertex_colors[:, :3] / 255
                else:
                    logging.info("Install trimesh / timor full to enable color meshes.")
            return mesh_loader.load(str(self.abs_filepath), self.scale)

    @property
    def abs_filepath(self) -> Path:
        """The absolute filepath of the mesh file."""
        return self.filepath

    @property
    def parameters(self) -> Dict[str, Union[str, float]]:
        """The mesh geometry is the only one that does not return the full parameters: Package dir is not returned!

        The parameters are given as {'file': str, 'scale': [float, array]}, where scale is optional and can be boat,
        a single float or an array of (x, y, z)-scale.
        """
        return {'file': str(self.filepath), 'scale': self.scale.tolist()}  # tolist to allow serialization

    @parameters.setter
    def parameters(self, parameters: Dict[str, any]):
        """Unpacks to filepath and scale."""
        self._filepath = Path(parameters['file'])
        if 'package_dir' in parameters:
            self._filepath = Path(parameters['package_dir']) / self._filepath
        self._scale = np.asarray(parameters.get('scale', 1.0))

    @property
    def urdf_properties(self) -> Tuple[str, Dict[str, Union[str, List[float]]]]:
        """Can only be saved as a dictionary if the mesh was loaded from a file and the file location is still known."""
        return 'mesh', {'filename': str(self.filepath), 'scale': self.scale.tolist()}

    @property
    def viz_object(self) -> Tuple[meshcat.geometry.Geometry, Transformation]:
        """Returns a meshcat mesh"""
        geom = pin.visualize.meshcat_visualizer.loadMesh(self.collision_geometry), self.placement
        if self._colors is not None:
            if geom[0].vertices.shape == self._colors.shape:
                geom[0].color = self._colors
            else:
                logging.warning("Color shape seems to not fit vertices.")
        return geom

    @property
    def volume(self) -> float:
        """
        Returns the volume of the mesh.

        Details can be found in "EFFICIENT FEATURE EXTRACTION FOR 2D/3D OBJECTS IN MESH REPRESENTATION"
        """
        logging.warning("The volume of a mesh is uncertain as there is no guarantee the surface is closed!")
        return self.collision_geometry.computeVolume()

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

    def __deepcopy__(self, memodict={}):
        """Must be implemented for Geometries, as hppfcl does not natively support it."""
        params = self.parameters
        return self.__class__(params, self.placement)

    def __eq__(self, other: Geometry):
        """We don't care about Mesh file paths, but equality of their collision geometries."""
        if isinstance(other, Mesh):
            return self.type == other.type \
                and np.all(self.scale == other.scale) \
                and self.placement == other.placement \
                and self.collision_geometry == other.collision_geometry
        return NotImplemented

    def __hash__(self):
        """Scale can't always be hashed as it can be a numpy array"""
        return hash((self.type, np.array(self.scale).tobytes(), self.placement.to_json_string()))


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
    def as_hppfcl_collision_object(self) -> Tuple[hppfcl.CollisionObject, ...]:
        """Creates hppfcl CollisionObjects at the current placement of self."""
        return tuple(h for g in self.composing_geometries for h in g.as_hppfcl_collision_object)

    @property
    def collision_data(self) -> Tuple[Tuple[np.ndarray, hppfcl.CollisionGeometry], ...]:
        """Composed Geometries are the only ones with more than one element in the collision data"""
        return tuple((g.placement.homogeneous, g.collision_geometry) for g in self.composing_geometries)

    @property
    def composing_geometries(self) -> Tuple[Geometry]:
        """This attribute should be read-only, therefore the leading underscore and the return as tuple."""
        return tuple(self._composing_geometries)

    @property
    def serialized(self) -> List[Dict[str, any]]:
        """Creates a json-compliant dictionary of a Geometry"""
        return [{
            'type': g.type.value,
            'parameters': g.parameters,
            'pose': g.placement.to_json_data()
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

    @property
    def volume(self) -> float:
        """The volume of a composed geometry is the sum of the volumes of the composing geometries."""
        return sum(g.volume for g in self.composing_geometries)

    def __eq__(self, other):
        """Equality is defined as having the same composing geometries."""
        if not isinstance(other, ComposedGeometry):
            return False
        return set(g for g in self.composing_geometries if not isinstance(g, EmptyGeometry)) == \
            set(g for g in other.composing_geometries if not isinstance(g, EmptyGeometry))

    def __hash__(self):
        """Hash is defined as the hash of the composing geometries."""
        return sum(hash(g) for g in self.composing_geometries if not isinstance(g, EmptyGeometry))


class EmptyGeometry(Geometry):
    """This is a dummy geometry to being able to prevent setting a geometry to None."""

    type: GeometryType = GeometryType.EMPTY

    def __init__(self,
                 parameters: Dict[str, Union[float, str]] = None,
                 pose: TransformationLike = Transformation.neutral(),
                 hppfcl_representation: Optional[hppfcl.CollisionGeometry] = None
                 ):
        """For an empty geometry, the parameters are not used."""
        if parameters is not None:
            logging.warning("EmptyGeometry does not use parameters, but they were given.")
        parameters = dict()
        super().__init__(parameters, pose, hppfcl_representation)

    @classmethod
    def from_hppfcl(cls, fcl: hppfcl.CollisionObject) -> Geometry:
        """This method is not implemented as it is not necessary."""
        raise NotImplementedError("Empty geometries are not loaded from hppfcl.")

    @property
    def parameters(self) -> Dict[str, Union[float, str]]:
        """No Geometry, no parameters."""
        return {}

    @parameters.setter
    def parameters(self, parameters: Dict[str, any]):
        """No Geometry, no parameters."""
        pass

    @property
    def as_hppfcl_collision_object(self) -> Tuple[hppfcl.CollisionObject, ...]:
        """An empty geometry does not have a collision object."""
        return ()

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

    @property
    def volume(self) -> float:
        """No Geometry, no volume."""
        return 0.0

    def _make_collision_geometry(self) -> hppfcl.CollisionGeometry:
        """No Geometry, no collision object."""
        pass

    def __eq__(self, other):
        """Empty geometries are equal to other empty geometries."""
        if isinstance(other, EmptyGeometry):
            return True
        return NotImplemented

    def __hash__(self):
        """Empty geometries are equal to other empty geometries."""
        return hash(self.__class__.__name__)
