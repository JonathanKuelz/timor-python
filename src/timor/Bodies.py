#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.02.22
import copy
from enum import Enum
import itertools
from pathlib import Path
import re
from typing import Dict, Iterable, Tuple, Union
import warnings

import numpy as np
import pinocchio as pin

from timor import Robot
from timor.Geometry import ComposedGeometry, EmptyGeometry, Geometry
from timor.utilities.crok_formatting import possibly_nest_as_list
from timor.utilities.dtypes import SingleSet, hppfcl_collision2pin_geometry
import timor.utilities.errors as err
from timor.utilities import logging
from timor.utilities.transformation import TransformationLike, Transformation


class Gender(Enum):
    """Enumerates possible connector genders"""

    f = 0
    female = 0
    m = 1
    male = 1
    h = 2
    hermaphroditic = 2

    def __str__(self) -> str:
        """Ensures defined string casting behavior when storing a connector of a certain gender as json"""
        defaults = {0: 'f', 1: 'm', 2: 'h'}
        return defaults[self.value]

    def connects(self, other: 'Gender') -> bool:
        """
        Indicates whether two genders form a connection. (sorry, code is homoamoric-agnostic)

        Possible connections:
        h <-> h
        f <-> m

        :param other: Another gender
        :return: True if connection works
        """
        if self is Gender.hermaphroditic:
            return self is other
        elif other is Gender.hermaphroditic:
            return False  # Self is not hermaphroditic
        else:  # Male female
            return self is not other


class Connector:
    """
    A connector is the interface defining where a body can be attached to another body.

    In order to connect two bodies, they must have a matching pair of connectors.
    The connector coordinate system is always "pointing away from the body it is attached to".
    """

    def __init__(self,
                 connector_id: str,
                 body2connector: TransformationLike = Transformation.neutral(),
                 parent: 'Body' = None,
                 gender: Union[Gender, str] = Gender.hermaphroditic,
                 connector_type: str = '',
                 size: Union[int, float, np.ndarray] = None  # size format depends on type
                 ):
        """
        Construct a Connector.

        :param connector_id: Unique "own" ID of the connector
        :param body2connector: The transformation from the body cos to connector coordinates
        :param parent: The body this connector is attached to. Will be set automatically when the connector is defined
          before the body and added later on. Every instance can have at most one parent.
        :param gender: Connector gender. Male and Female connect, Hermaphroditic and Hermaphroditic connect.
        :param connector_type: Can be a custom string to define the type of the connector.
        :param size: Can be an integer, float or array describing the size of the connector.
        """
        self._id: str = str(connector_id)
        self.body2connector: Transformation = Transformation(body2connector)
        self.parent: 'Body' = parent
        self.gender: Gender = gender if isinstance(gender, Gender) else Gender[gender]  # Enforce typing
        self._type: str = connector_type
        self.size: Union[int, float, np.ndarray] = size

    def __copy__(self):
        """Custom copy behavior: Remove the reference to the parent body."""
        return self.__class__(self._id, self.body2connector, None, self.gender, self._type, self.size)

    def __str__(self) -> str:
        """The string representation of the connector"""
        return f"Connector: {self.id}"

    @classmethod
    def from_crok_specification(cls, d: Dict) -> 'Connector':
        """
        Maps the crok-connector description to an instance of this class.

        :param d: A dictionary with relevant meta-information
        :return: An instantiated connector
        """
        return cls(
            connector_id=d['ID'],
            body2connector=d.get('pose', None),
            gender=Gender[d.get('gender', 'hermaphroditic')],
            connector_type=d.get('type', ''),
            size=d.get('size', None))

    def to_crok_specification(self) -> Dict[str, any]:
        """
        :return: Returns the body specification in a json-ready dictionary
        """
        if isinstance(self.size, np.ndarray):
            size = self.size.tolist()
        else:
            size = self.size
        return {
            'ID': self._id,
            'pose': self.body2connector.crok_description,
            'gender': str(self.gender),
            'type': self.type,
            'size': size
        }

    @property
    def connector2body(self) -> Transformation:
        """The inverse transformation from connector to body coordinates"""
        return self.body2connector.inv

    @property
    def id(self) -> Tuple[Union[None, str], Union[None, str], str]:
        """Composed ID: (module ID, body ID, connector ID)"""
        try:
            return self.parent.id[0], self.parent.id[1], self._id
        except AttributeError:
            return None, None, self._id

    @property
    def own_id(self) -> str:
        """The last part of the id - uniquely identifying a connector within its module"""
        return self._id

    @property
    def type(self) -> str:
        """Wrapped as _type attribute to not be confused with python-native type(). Also, is kind of static anyway"""
        return self._type

    def connects(self, other: 'Connector') -> bool:
        """
        Checks if this connector can be connected with "other".

        :param other: Another connector
        :return: True if connection is possible. False else.
        """
        size = self.size == other.size
        if not isinstance(size, bool):  # Which is the case when size is an array
            size = all(size)
        return self.type == other.type and size and self.gender.connects(other.gender)


class ConnectorSet(SingleSet[Connector]):
    """A set that raises an error if a duplicate connector is added."""

    def __contains__(self, item: Connector):
        """Custom duplicate check (unique ID)"""
        return item.id in (con.id for con in self)


class Body:
    """
    A crok body.

    Refer to `crok -> robot description <https://crok.cps.in.tum.de/documentation/robot/>`_ for more details.
    """

    def __init__(self,
                 body_id: str,
                 collision: Geometry,
                 visual: Geometry = None,
                 connectors: Iterable[Connector] = None,
                 inertia: pin.Inertia = pin.Inertia(0, np.zeros(3), np.zeros((3, 3))),
                 in_module: 'AtomicModule' = None
                 ):
        """
        Construct a Body.

        :param body_id: The unique ID of the body
        :param collision: The collision geometry of the body
        :param visual: The visual geometry of the body. If None is given, defaults to collision
        :param connectors: The connectors of the body, defining possible interfaces to other bodies
        :param inertia: The inertia of the body, as pin.Inertia
        :param in_module: The module this body is part of. Will be automatically assigned if the body is defined and
          added to a module later on.
        """
        self._id: str = str(body_id)
        self.in_module: 'AtomicModule' = in_module
        self.connectors: ConnectorSet[Connector] = ConnectorSet(connectors) \
            if connectors is not None \
            else ConnectorSet()
        for connector in self.connectors:
            if connector.parent not in (None, self):
                raise ValueError("Cannot set a new parent for a connector!")
            connector.parent = self
        self.inertia: pin.Inertia = inertia

        if collision is None:
            raise ValueError("Collision model must be given! EmptyGeometry is a possible workaround.")
        self.collision: Geometry = collision
        self._visual: Geometry = visual

    def __copy__(self) -> 'Body':
        """Custom copy to allow differing connector parent references"""
        new_connectors = [copy.copy(con) for con in self.connectors]
        return self.__class__(
            body_id=self._id,
            connectors=new_connectors,
            inertia=self.inertia,
            collision=self.collision,
            visual=self.visual,
            in_module=self.in_module
        )

    def __str__(self) -> str:
        """The string representation of a body"""
        return f"Body: {self.id}"

    @classmethod
    def from_crok_specification(cls, d: Dict, package_dir: Path = None) -> 'Body':
        """
        Maps the crok-body description to an instance of this class.

        The dictionary will be modified in-place until empty (everything was parsed).

        :param d: A dictionary with relevant meta-information
        :param package_dir: If a mesh is given, it is given relative to a package directory that must be specified
        :return: An instantiated body
        """
        body_id = d.pop('ID')
        connectors = [Connector.from_crok_specification(con_data)
                      for con_data in possibly_nest_as_list(d.pop('connectors', []))]
        inertia = pin.Inertia(float(d.pop('mass')),
                              np.array(d.pop('r_com'), dtype=float),
                              np.asarray(d.pop('inertia'), dtype=float))
        geometries = [None, None]
        has_wrl = False
        for i, geometry_type in enumerate(('collision', 'visual')):
            try:
                specs = d.pop(geometry_type)
            except KeyError:
                if geometry_type == 'collision':
                    logging.warning(f"No collision model given for body {body_id}")
                    geometries[i] = EmptyGeometry({})
                else:
                    # Visuals can be none
                    geometries[i] = None
                continue

            singular_geometries = list()
            for geo in possibly_nest_as_list(specs):  # There can be multiple geometries
                try:
                    singular_geometries.append(Geometry.from_crok_description(geo, package_dir=package_dir))
                except ValueError as verr:
                    if geo['type'] == 'mesh' and geo['parameters']['file'].endswith('.wrl'):
                        geometries[i] = EmptyGeometry({})
                        warnings.warn("Cannot load wrl files")
                        has_wrl = True
                    else:
                        raise verr

            if len(singular_geometries) > 1:
                geometries[i] = ComposedGeometry(singular_geometries)
            elif len(singular_geometries) == 1:
                geometries[i] = singular_geometries[0]
            else:
                geometries[i] = EmptyGeometry({})

        collision, visual = geometries
        if isinstance(collision, EmptyGeometry) and has_wrl and not isinstance(visual, EmptyGeometry):
            # TODO: Implement wrl geometries
            collision = visual

        if len(d) > 0:
            raise ValueError(f"Unknown keys in body description: {d}")

        return cls(body_id, collision, visual, connectors, inertia)

    def to_crok_specification(self) -> Dict[str, any]:
        """
        :return: Returns the body specification in a json-ready dictionary
        """
        geometry = {}
        for geo in ('_visual', 'collision'):
            if isinstance(self.__dict__[geo], EmptyGeometry) or self.__dict__[geo] is None:
                continue
            geometry[re.sub('_', '', geo)] = possibly_nest_as_list(self.__dict__[geo].crok)
        return {
                   'ID': self._id,
                   'mass': self.mass,
                   'inertia': self.inertia.inertia.tolist(),
                   'r_com': self.inertia.lever.tolist(),
                   'connectors': [con.to_crok_specification() for con in
                                  sorted(self.connectors, key=lambda con: con.id)]
               } | geometry

    @property
    def id(self) -> Tuple[Union[None, str], str]:
        """Composed ID: (module ID, body ID)"""
        try:
            return self.in_module.id, self._id
        except AttributeError:
            return None, self._id

    @property
    def mass(self) -> float:
        """The mass of the body in [kg]"""
        return self.inertia.mass

    @property
    def visual(self) -> Geometry:
        """Defaults to collision if not explicitly specified"""
        return self._visual if self._visual is not None else self.collision

    def as_pin_body(self, placement: TransformationLike = Transformation.neutral()) -> Robot.PinBody:
        """
        Extracts the body properties that describe a body in a pinocchio robot.

        Always places the body in the origin.

        :param placement: Relative placement of the body to its parent joint
        """
        placement = Transformation(placement)
        collision, visual = None, None
        if not isinstance(self.collision, EmptyGeometry):
            collision = [hppfcl_collision2pin_geometry(geo, t, '.'.join(self.id) + f'.c{i}', 0)
                         for i, (t, geo) in enumerate(self.collision.collision_data)]
        visual = [hppfcl_collision2pin_geometry(geo, t, '.'.join(self.id) + f'.v{i}', 0)
                  for i, (t, geo) in enumerate(self.visual.collision_data)]

        return Robot.PinBody(
            inertia=self.inertia,
            placement=pin.SE3(placement.homogeneous),
            name='.'.join(self.id),
            collision_geometries=collision,
            visual_geometries=visual
        )

    def possible_connections_with(self, other: 'Body') -> SingleSet[Tuple[Connector, Connector]]:
        """
        Returns a list of connectors that can connect this with other body.

        This method does not check whether the connector IDs of the other body overlap with the connector IDs
        of this body - this has to be achieved by e.g. putting both bodies in a module or in the same ModulesDB.

        :param other: Another body
        :return: A list of matching connector pairs like (this_body_connector, other_body_connector)
        """
        if other.id == self.id:
            raise err.UniqueValueError("Two connectors with identical ID cannot be connected")
        matches = SingleSet()
        for own, foreign in itertools.product(self.connectors, other.connectors):
            if own.connects(foreign):
                matches.add((own, foreign))

        return matches


class BodySet(SingleSet[Body]):
    """A set that raises an error if a duplicate body is added"""

    def __contains__(self, item: Body):
        """Custom duplicate check (unique ID)"""
        return item.id in (bod.id for bod in self)
