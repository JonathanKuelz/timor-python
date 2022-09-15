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
    def from_crok_description(cls,
                              ID: str,
                              collision: Union[Geometry.Geometry, dict],
                              visual: Optional[Union[Geometry.Geometry, dict]] = None,
                              name: Optional[str] = None,
                              package_dir: Optional[Path] = None
                              ) -> Obstacle:
        """
        Takes a crok geometry specification and returns the according Obstacle.

        :param ID: The obstacle unique ID
        :param collision: A crok Geometry as described in the crok documentation - used for collision detection
        :param visual: A crok Geometry as described in the crok documentation - defaults to collision
        :param package_dir: If a mesh is given, it is given relative to a package directory that must be specified
        :param name: The obstacle display name
        :return: The obstacle class instance that matches the specification
        """
        collision_geometry = Geometry.Geometry.from_crok_description(collision, package_dir=package_dir)
        if visual is not None:
            visual_geometry = Geometry.Geometry.from_crok_description(visual, package_dir=package_dir)
        else:
            visual_geometry = None

        return cls(ID, collision_geometry, visual_geometry, name=name)

    @classmethod
    def from_hppfcl(cls, ID: str, fcl: hppfcl.CollisionObject,
                    visual_representation: Optional[hppfcl.CollisionObject] = None,
                    name: Optional[str] = None) -> Obstacle:
        """Create a Timor obstacle from a hppfcl obstacle

        A helper function that transforms a generic hppfcl collision object to an Obstacle with the according geometry.
          (Which allows plotting, composing a scenario of multiple obstacles, ...)

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

    def to_crok(self) -> Dict[str, Union[str, dict]]:
        """
        Interface to write a crok-ready list of dictionaries that is serializable to json.

        :return: A dictionary of the format described in the crok scenario documentation
        """
        crok_description = {
            'ID': self.id,
            'name': self.name,
            'collision': self.collision.crok
        }
        if self._visual is not None:
            crok_description['visual'] = self._visual.crok
        return crok_description

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
