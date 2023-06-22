#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.01.22
from __future__ import annotations

from typing import Dict, Optional, Union

from hppfcl import hppfcl
import meshcat.geometry
import pinocchio as pin

from timor import Geometry
from timor.utilities.jsonable import JSONable_mixin


class Obstacle(JSONable_mixin):
    """
    A wrapper class to integrate hppfcl obstacles in custom Environments.
    """

    mesh_material = meshcat.geometry.MeshLambertMaterial(color=0xECA19D, reflectivity=.8)
    mesh_material_vertex_color = meshcat.geometry.MeshLambertMaterial(vertexColors=True, reflectivity=.8)

    def __init__(self,
                 ID: str,
                 collision: Geometry.Geometry,
                 visual: Optional[Geometry.Geometry] = None,
                 name: Optional[str] = None
                 ):
        """Generate an obstacle:

        :param ID: Unique identifier
        :param collision: The collision geometry of the object
        :param visual: The visual geometry of the object. Defaults to the collision geometry.
        :param name: Optional display name
        """
        self._id = str(ID)
        self.collision = collision
        self._visual = visual
        self.name: Optional[str] = name

    @classmethod
    def from_json_data(cls, d: Dict[str, Union[str, dict]], *args, **kwargs) -> Obstacle:
        """
        Takes a json geometry specification and returns the according Obstacle.

        :param d: The serialized obstacle information, containing:

            * ID: The obstacle unique ID
            * collision: A Geometry used for collision detection
            * visual: A Geometry for visualization of this Obstacle - defaults to collision if not given
            * name: The obstacle display name
        :return: The obstacle class instance that matches the specification
        """
        ID = d['ID']
        collision = d['collision']
        collision_geometry = Geometry.Geometry.from_json_data(collision)
        if 'visual' in d:
            visual_geometry = Geometry.Geometry.from_json_data(d['visual'])
        else:
            visual_geometry = None

        return cls(ID, collision_geometry, visual_geometry, name=d.get('name', None))

    @classmethod
    def from_hppfcl(cls, ID: str, fcl: hppfcl.CollisionObject,
                    visual_representation: Optional[hppfcl.CollisionObject] = None,
                    name: Optional[str] = None) -> Obstacle:
        """Create a Timor obstacle from a hppfcl obstacle

        A helper function that transforms a generic hppfcl collision object to an Obstacle with the according geometry.
          (Which allows plotting, composing a task of multiple obstacles, ...)

        :param ID: The id that shall be given to the Obstacle.
        :param fcl: The hppfcl collision object
        :param visual_representation: An optional, differing visual representation of the collision object.
        :param name: The obstacle display name
        :return: An instance of the according Obstacle class, based on the kind of collision object handed over.
        """
        collision_geometry = Geometry.Geometry.from_hppfcl(fcl)
        if visual_representation is not None:
            visual_geometry = Geometry.Geometry.from_hppfcl(visual_representation)
        else:
            visual_geometry = None
        return cls(ID, collision_geometry, visual_geometry, name=name)

    def collides(self, other: Obstacle) -> bool:
        """Check if this obstacle collides with another obstacle.

        Be careful, this is probably slow due to the initialisation of a new request/result pair and should only be used
        for debugging. Use the collision checking of the environment / robot instead.

        :param other: The other obstacle
        :return: True if the obstacles collide, False otherwise
        """
        request = hppfcl.CollisionRequest()
        result = hppfcl.CollisionResult()

        for own_collision in self.collision.as_hppfcl_collision_object:
            for other_collision in other.collision.as_hppfcl_collision_object:
                hppfcl.collide(own_collision, other_collision, request, result)
                if result.isCollision():
                    return True
        return False

    def to_json_data(self) -> Dict[str, Union[str, dict]]:
        """
        Interface to write a list of geometry-dictionaries that is serializable to json.

        :return: A dictionary of the format described in the task documentation
        """
        serialized = {
            'ID': self.id,
            'name': self.name,
            'collision': self.collision.serialized
        }
        if self._visual is not None:
            serialized['visual'] = self._visual.serialized
        return serialized

    def visualize(self, viz: pin.visualize.MeshcatVisualizer,
                  material: Optional[meshcat.geometry.Material] = None):
        """
        A custom visualization should be implemented by children.

        :param viz: MeshcatVisualizer to add this obstacle to.
        :param material: Material to use for this obstacle; default: light red / mesh color if provided
        """
        overwrite_mesh_color = False
        if material is None:
            material = self.mesh_material
        else:
            overwrite_mesh_color = True
        if isinstance(self.visual, Geometry.ComposedGeometry):
            for i, geom in enumerate(self.visual.composing_geometries):
                viz_geometry, transform = geom.viz_object
                if not hasattr(viz_geometry, 'color') or viz_geometry.color is None or overwrite_mesh_color:
                    viz.viewer[self.display_name + f'_{i}'].set_object(viz_geometry, material)
                else:
                    viz.viewer[self.display_name + f'_{i}'].set_object(viz_geometry, self.mesh_material_vertex_color)
                viz.viewer[self.display_name + f'_{i}'].set_transform(transform.homogeneous)
        else:
            viz_geometry, transform = self.visual.viz_object
            if not hasattr(viz_geometry, 'color') or viz_geometry.color is None or overwrite_mesh_color:
                viz.viewer[self.display_name].set_object(viz_geometry, material)
            else:
                viz.viewer[self.display_name].set_object(viz_geometry, self.mesh_material_vertex_color)
            viz.viewer[self.display_name].set_transform(transform.homogeneous)

    @property
    def display_name(self) -> str:
        """Used for visualization and debugging. If a diplay name is explicitly given, use it, if not, it's the ID."""
        if self.name is None:
            return self.id
        return self.id

    @property
    def id(self) -> str:
        """Returns the ID of the obstacle"""
        return self._id

    @property
    def visual(self) -> Geometry.Geometry:
        """Defaults to collision"""
        if self._visual is None:
            return self.collision
        return self._visual

    def __copy__(self):
        """Not supported!"""
        raise NotImplementedError("Cannot shallow-copy hppfcl geometries. Use deepcopy.")

    def __eq__(self, other):
        """Compare two obstacles by their ID, name, and geometry"""
        if not isinstance(other, Obstacle):
            return NotImplemented
        return self.id == other.id and self.name == other.name and self.collision == other.collision

    def __hash__(self):
        """The ID is sufficient to hash an obstacle"""
        return hash(self.id)
