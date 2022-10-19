#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.01.22
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from hppfcl import hppfcl
import pinocchio as pin

from timor import Geometry


class Obstacle:
    """
    A wrapper class to integrate hppfcl obstacles in custom Environments.
    """

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
    def from_json_data(cls, d: Dict[str, Union[str, dict]]) -> Obstacle:
        """
        Takes a json geometry specification and returns the according Obstacle.

        :param d: The serialized obstacle information, containing:
            - ID: The obstacle unique ID
            - collision: A Geometry used for collision detection
            - visual: A Geometry for visualization of this Obstacle - defaults to collision if not given
            - package_dir: If a mesh is given, it is given relative to a package directory that must be specified
            - name: The obstacle display name
        :return: The obstacle class instance that matches the specification
        """
        ID = d['ID']
        collision = d['collision']
        package_dir = Path(d['package_dir']) if 'package_dir' in d else None
        collision_geometry = Geometry.Geometry.from_json_data(collision, package_dir=package_dir)
        if 'visual' in d:
            visual_geometry = Geometry.Geometry.from_json_data(d['visual'], package_dir=package_dir)
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

        for own_t, own_collision in self.collision.collision_data:
            for other_t, other_collision in other.collision.collision_data:
                own_t, other_t = map(lambda t: hppfcl.Transform3f(t[:3, :3], t[:3, 3]), [own_t, other_t])
                hppfcl.collide(own_collision, own_t, other_collision, other_t, request, result)
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

    def visualize(self, viz: pin.visualize.MeshcatVisualizer):
        """A custom visualization should be implemented by children."""
        if isinstance(self.visual, Geometry.ComposedGeometry):
            for i, geom in enumerate(self.visual.composing_geometries):
                viz_geometry, transform = geom.viz_object
                viz.viewer[self.display_name + f'_{i}'].set_object(viz_geometry)
                viz.viewer[self.display_name + f'_{i}'].set_transform(transform.homogeneous)
        else:
            viz_geometry, transform = self.visual.viz_object
            viz.viewer[self.display_name].set_object(viz_geometry)
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
