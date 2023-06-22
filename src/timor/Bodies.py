#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.02.22
from __future__ import annotations

import abc
import copy
from enum import Enum
import itertools
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pinocchio as pin

from timor.Geometry import ComposedGeometry, EmptyGeometry, Geometry
from timor.Robot import PinBody
from timor.utilities import logging
from timor.utilities.dtypes import SingleSet, SingleSetWithID, hppfcl_collision2pin_geometry
import timor.utilities.errors as err
from timor.utilities.json_serialization_formatting import possibly_nest_as_list
from timor.utilities.jsonable import JSONable_mixin
from timor.utilities.transformation import Transformation, TransformationLike


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


class Connector(JSONable_mixin):
    """
    A connector is the interface defining where a body can be attached to another body.

    In order to connect two bodies, they must have a matching pair of connectors.
    The connector coordinate system is always "pointing away from the body it is attached to".
    """

    def __init__(self,
                 connector_id: str,
                 body2connector: TransformationLike = Transformation.neutral(),
                 parent: 'BodyBase' = None,
                 gender: Union[Gender, str] = Gender.hermaphroditic,
                 connector_type: str = '',
                 size: Union[int, float, np.ndarray] = tuple()  # size format depends on type
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
        self.parent: 'BodyBase' = parent
        self.gender: Gender = gender if isinstance(gender, Gender) else Gender[gender]  # Enforce typing
        self._type: str = connector_type
        if isinstance(size, Iterable) and not isinstance(size, np.ndarray):
            size = np.array(size)
        self.size: Union[int, float, np.ndarray] = size

    @classmethod
    def from_json_data(cls, d: Dict, *args, **kwargs) -> Connector:
        """
        Maps the serialized connector description to an instance of this class.

        :param d: A dictionary with relevant meta-information
        :return: An instantiated connector
        """
        return cls(
            connector_id=d['ID'],
            body2connector=d.get('pose', None),
            gender=Gender[d.get('gender', 'hermaphroditic')],
            connector_type=d.get('type', ''),
            size=d.get('size', None))

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

    def to_json_data(self) -> Dict[str, any]:
        """
        :return: Returns the connector specification in a json-ready dictionary
        """
        if isinstance(self.size, np.ndarray):
            size = self.size.tolist()
        elif isinstance(self.size, float):
            size = (self.size,)
        else:
            size = self.size
        return {
            'ID': self._id,
            'pose': self.body2connector.to_json_data(),
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

    def __copy__(self):
        """Custom copy behavior: Remove the reference to the parent body."""
        return self.__class__(self._id, self.body2connector, None, self.gender, self._type, self.size)

    def __eq__(self, other):
        """Equality check: Same ID, gender, type, size, and pose"""
        if not isinstance(other, Connector):
            return NotImplemented
        return self.id == other.id and self.gender is other.gender and self.type == other.type \
            and np.all(self.size == other.size) and self.body2connector == other.body2connector

    def __hash__(self):
        """Hashing: We assume IDs are unique, so it poses a valid hash."""
        return hash(self.id)

    def __str__(self) -> str:
        """The string representation of the connector"""
        return f"Connector: {self.id}"


class ConnectorSet(SingleSetWithID):
    """A set that raises an error if a duplicate connector is added."""


class BodyBase(abc.ABC, JSONable_mixin):
    """
    A body describes a rigid robot element with kinematic, dynamic, geometric, and connection properties.

    A body is defined by a geometry and inertia-parameters. Its placement is defined implicitly by the connectors
      attached to it.
    """

    collision: Geometry  # The collision geometry of the body
    connectors: ConnectorSet  # The connectors attached to this body
    inertia: pin.Inertia  # The inertia of the body (containing mass, center of mass, and inertia matrix)
    in_module: 'ModuleBase'  # The parent module of the body
    _id: str  # The unique identifier of the body
    _visual: Optional[Geometry] = None  # The visual geometry of the body - defaults to collision

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

    def as_pin_body(self, placement: TransformationLike = Transformation.neutral()) -> PinBody:
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

        return PinBody(
            inertia=self.inertia,
            placement=pin.SE3(placement.homogeneous),
            name='.'.join(self.id),
            collision_geometries=collision,
            visual_geometries=visual
        )

    def possible_connections_with(self, other: 'BodyBase') -> SingleSet:
        """
        Returns a set of connectors that can connect this with other body.

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

    def to_json_data(self) -> Dict[str, any]:
        """
        :return: Returns the body specification in a json-ready dictionary
        """
        geometry = {}
        for geo in ('_visual', 'collision'):
            if isinstance(self.__dict__.get(geo, None), EmptyGeometry) or self.__dict__.get(geo, None) is None:
                continue
            geometry[re.sub('_', '', geo)] = possibly_nest_as_list(self.__dict__[geo].serialized)
        return {
            'ID': self._id,
            'mass': self.mass,
            'inertia': self.inertia.inertia.tolist(),
            'r_com': self.inertia.lever.tolist(),
            'connectors': [con.to_json_data() for con in
                           sorted(self.connectors, key=lambda con: con.id)],
            **geometry
        }

    @abc.abstractmethod
    def __copy__(self):
        """Returns a copy of the body"""
        pass

    def __eq__(self, other):
        """Equality check. The visual attribute is not checked as it does not provide relevant functionality."""
        if not isinstance(other, BodyBase):
            return NotImplemented
        return self.collision == other.collision and self.connectors == other.connectors and \
            self.inertia == other.inertia and self._id == other._id

    def __hash__(self):
        """Hashing: We assume IDs are unique, so it poses a valid hash."""
        return hash(self.id)

    def __str__(self) -> str:
        """The string representation of a body"""
        return f"Body: {self.id}"


class Body(BodyBase):
    """The default body class, represents a static/immutable rigid body."""

    def __init__(self,
                 body_id: str,
                 collision: Geometry,
                 visual: Geometry = None,
                 connectors: Iterable[Connector] = (),
                 inertia: pin.Inertia = pin.Inertia(0, np.zeros(3), np.zeros((3, 3))),
                 in_module: 'ModuleBase' = None
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
        self.in_module: 'ModuleBase' = in_module
        self.connectors: ConnectorSet = ConnectorSet(connectors)
        for connector in self.connectors:
            if connector.parent not in (None, self):
                raise ValueError("Cannot set a new parent for a connector!")
            connector.parent = self
        self.inertia: pin.Inertia = inertia

        if collision is None:
            raise ValueError("Collision model must be given! EmptyGeometry is a possible workaround.")
        self.collision: Geometry = collision
        self._visual: Geometry = visual

    @classmethod
    def from_json_data(cls, d: Dict, package_dir: Path = None, *args, **kwargs) -> Body:
        """
        Maps the json-serialized body description to an instance of this class.

        The dictionary will be modified in-place until empty (everything was parsed).

        :param d: A dictionary with relevant meta-information
        :param package_dir: If a mesh is given, it is given relative to a package directory that must be specified
        :return: An instantiated body
        """
        body_id = d.pop('ID')
        connectors = [Connector.from_json_data(con_data)
                      for con_data in possibly_nest_as_list(d.pop('connectors', []))]
        inertia = pin.Inertia(float(d.pop('mass')),
                              np.array(d.pop('r_com'), dtype=float),
                              np.asarray(d.pop('inertia'), dtype=float))
        geometries = [None, None]
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
                singular_geometries.append(Geometry.from_json_data(geo, package_dir=package_dir))

            if len(singular_geometries) > 1:
                geometries[i] = ComposedGeometry(singular_geometries)
            elif len(singular_geometries) == 1:
                geometries[i] = singular_geometries[0]
            else:
                geometries[i] = EmptyGeometry({})

        collision, visual = geometries

        if len(d) > 0:
            raise ValueError(f"Unknown keys in body description: {d}")

        return cls(body_id, collision, visual, connectors, inertia)

    @classmethod
    def from_json_string(cls, s: str, package_dir: Path = None) -> Body:
        """
        Maps the json-serialized body description to an instance of this class.

        :param s: A json string with relevant meta-information
        :param package_dir: If a mesh is given, it is given relative to a package directory that must be specified
        :return: An instantiated body
        """
        return cls.from_json_data(json.loads(s), package_dir=package_dir)

    def __copy__(self) -> BodyBase:
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


class BodySet(SingleSetWithID):
    """A set that raises an error if a duplicate body is added"""
