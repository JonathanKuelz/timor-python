from __future__ import annotations

import abc
from collections import Counter
import copy
from dataclasses import dataclass
import datetime
import itertools
import json
import math
from pathlib import Path
import random
import re
from typing import Callable, Collection, Dict, Iterable, List, Optional, Set, Tuple, Union
import uuid
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pinocchio as pin

from timor import Geometry, Robot
from timor.Bodies import Body, BodyBase, BodySet, Connector, ConnectorSet, Gender
from timor.Joints import Joint, JointSet, TimorJointType
from timor.utilities import logging, spatial, write_urdf
from timor.utilities.dtypes import SingleSet, TypedHeader, randomly
import timor.utilities.errors as err
from timor.utilities.json_serialization_formatting import compress_json_vectors, possibly_nest_as_list
from timor.utilities.transformation import Transformation, TransformationLike


@dataclass
class ModuleHeader(TypedHeader):
    """The header every module contains"""

    ID: str
    name: str
    date: datetime.datetime = datetime.datetime(1970, 1, 1)
    author: List[str] = TypedHeader.string_list_factory()
    email: List[str] = TypedHeader.string_list_factory()
    affiliation: List[str] = TypedHeader.string_list_factory()
    cost: float = 0.


class ModuleBase(abc.ABC):
    """Base class for any robot module."""

    def __init__(self,
                 header: Union[Dict, ModuleHeader],
                 bodies: Iterable[BodyBase] = (),
                 joints: Iterable[Joint] = ()
                 ):
        """
        A single module, the basic building block of a modular robot.

        :param header: The header of the module containing distinct meta-information
        :param bodies: The bodies contained in the module - defaults to none.
        :param joints: The joints contained in the module - defaults to none. The parent and child bodies must be
          provided for each joint.
        """
        if isinstance(header, dict):
            header['ID'] = str(header['ID'])  # Sensitive due to json parsing
            header = ModuleHeader(**header)
        self.header: ModuleHeader = header
        self._bodies: BodySet = BodySet(bodies)
        for body in self._bodies:
            body.in_module = self
        for joint in joints:
            if not ((joint.parent_body in self.bodies) and (joint.child_body in self.bodies)):
                raise ValueError("At least one of the joints parent/child bodies are missing in the module!")
            joint.in_module = self
        self._joints: JointSet = JointSet(joints)
        self._check_unique_connectors()

    def __copy__(self):
        """Disable builtin copy, there should be no use case for two instances of a module with the same ID."""
        raise NotImplementedError("Use custom module copy function and provide an ID suffix")

    def __str__(self):
        """String representation for the module."""
        return f"Module {self.name}, ID: {self.id}"

    @property
    def connectors_by_own_id(self) -> Dict[str, Connector]:
        """
        Returns a mapping of connector._id values to the instance.

        Keep in mind, these are not the composed id's of the connectors, so they are not unambiguous outside a module!
        """
        return {con._id: con for con in itertools.chain.from_iterable(b.connectors for b in self.bodies)}

    @property
    def module_graph(self) -> nx.DiGraph:
        """
        Returns a graph with nodes that can either be joints, bodies or connectors.

        Each element in the module graph belongs to one body, joint or connector. The edges define child-parent
        relationships between the elements. Every edge has a transform property, describing the relative transformation
        from one element's reference frame to another element's reference frame. The edges are directed and
        anti-parallel, s.t. for every edge between two nodes (u, v), there exists an edge between (v, u) for which
        the transform attribute is the inverse of the first edge's transform.
        """
        G_m = nx.DiGraph()
        for body in self.bodies:
            G_m.add_node(body)
        for jnt in self.joints:
            G_m.add_node(jnt)
            # If a node is defined twice, networkx just interprets it as the same node
            G_m.add_edge(jnt.parent_body, jnt, transform=jnt.parent2joint)
            G_m.add_edge(jnt, jnt.child_body, transform=jnt.joint2child)

        for connector_id, connector in self.available_connectors.items():
            G_m.add_edge(connector.parent, connector, transform=connector.body2connector)

        for u, v, t in G_m.edges.data('transform'):
            # Generate the edges in opposing directing
            G_m.add_edge(v, u, transform=t.inv)

        return G_m

    @property
    def available_connectors(self) -> Dict[Tuple[str, str, str], Connector]:
        """Returns all connectors of all bodies contained in this module as a mapping ID -> instance"""
        return {c.id: c for c in itertools.chain.from_iterable(b.connectors for b in self.bodies)}

    @property
    def bodies(self) -> BodySet:
        """Returns all bodies contained in this module."""
        return self._bodies

    @property
    def id(self) -> str:
        """The unique ID of this module."""
        return self.header.ID

    @property
    def joints(self) -> JointSet:
        """Returns all joints contained in this module"""
        return self._joints

    @property
    def mass(self) -> float:
        """Returns the total mass of the module."""
        return sum(b.mass for b in self.bodies)

    @property
    def num_bodies(self):
        """The number of bodies in this module."""
        return len(self._bodies)

    @property
    def num_connectors(self):
        """The number of connectors available in the module."""
        # Slightly faster than instantiating available_connectors and counting then
        return sum(1 for _ in itertools.chain.from_iterable(b.connectors for b in self.bodies))

    @property
    def num_joints(self):
        """The number of joints in this module"""
        return len(self._joints)

    @property
    def name(self) -> str:
        """Returns the unique name of this module."""
        return self.header.name

    def _check_unique_connectors(self):
        """Sanity check that all connectors are unique"""
        if len(self.available_connectors) == 0:
            return
        counter = Counter(c._id for c in itertools.chain.from_iterable(b.connectors for b in self.bodies))
        if max(counter.values()) > 1:
            raise err.UniqueValueError("A module cannot contain multiple connectors with the same id.")

    def copy(self, suffix: str) -> ModuleBase:
        """
        Copies a module, but changes the id of the copy.

        :param suffix: Suffix to add to the copies' id. Necessary, to distinguish original and copy
        :return: A functional copy of this module with altered ID
        """
        new_header = ModuleHeader(
            ID=self.id + suffix,
            name=self.name + suffix,
            author=self.header.author, email=self.header.email, affiliation=self.header.affiliation
        )
        new_bodies = BodySet(copy.copy(body) for body in self.bodies)
        bodies_by_id = {body.id: body for body in new_bodies}
        new_joints = JointSet()
        for jnt in self.joints:
            new = copy.copy(jnt)
            new.parent_body = bodies_by_id[new.parent_body.id]
            new.child_body = bodies_by_id[new.child_body.id]
            new_joints.add(new)
        return self.__class__(new_header, new_bodies, new_joints)

    def debug_visualization(self,  # pragma: no cover
                            viz: pin.visualize.MeshcatVisualizer = None,
                            show_com: bool = True,
                            base_placement: np.array = Transformation.neutral()) -> pin.visualize.MeshcatVisualizer:
        """
        Do not use this method in proper environment visualizations

        :param viz: MeshcatVisualizer to show debug information in
        :param show_com: show center of mass frame
        :param base_placement: where to show this module in the global frame
        :return: visualizer object (keep if you want to add more to this)
        """
        # Usually single female is connector closer to base
        base_connector = [c for c in self.available_connectors.values() if c.gender is Gender.female]
        if len(base_connector) == 1:
            base_connector_id = (0, base_connector[0].id[2])
            tmp_ass = ModuleAssembly(ModulesDB([self]), [self.id], (), base_connector_id)
        else:  # Otherwise, any connector can be fixed to environment
            base_connector_id = (0, list(self.available_connectors)[0][2])
            tmp_ass = ModuleAssembly(ModulesDB([self]), [self.id], (), base_connector_id)

        return tmp_ass.to_pin_robot(add_com_frames=show_com, base_placement=base_placement).visualize(viz, 'full')

    def can_connect(self, other: ModuleBase) -> bool:
        """Returns true if at least one of the connectors of this module matches at least one connector of other"""
        return any(this_con.connects(other_con) for this_con, other_con
                   in itertools.product(self.available_connectors.values(), other.available_connectors.values()))

    def to_json_data(self) -> Dict:
        """
        Write a json in that fully describes this module.

        :return: Returns the module specification in a json-ready dictionary
        """
        header = self.header._asdict()
        header['date'] = header['date'].strftime('%Y-%m-%d')

        return {
            'header': header,
            'bodies': [body.to_json_data() for body in sorted(self.bodies, key=lambda b: b.id)],
            'joints': [joint.to_json_data() for joint in sorted(self.joints, key=lambda j: j.id)]
        }

    def to_json_string(self) -> str:
        """
        Write a json in that fully describes this module.

        :return: Returns the module specification in a json-ready dictionary
        """
        return json.dumps(self.to_json_data(), indent=2)


class AtomicModule(ModuleBase):
    """
    Atomic Modules are the simplest module representation. They are static (immutable) and can be (de)serialized.
    """

    def __init__(self, header: Union[Dict, ModuleHeader], bodies: Iterable[Body] = (), joints: Iterable[Joint] = ()):
        """Overwrites the default constructor for more precise type hints"""
        super().__init__(header=header, bodies=bodies, joints=joints)

    def __copy__(self):
        """Deactivates the builtin copy method."""
        super().__copy__()

    @classmethod
    def from_concert_specification(cls, d: Dict, package_dir: Path) -> 'ModuleBase':  # pragma: no cover
        """
        Maps a concert-module description to an instance of this class.

        This library was developed in the context of the EU sponsored CONCERT project.
        For more information visit https://concertproject.eu/
        The concert module definition closely follows the Timor one. However, in the concert project, we only deal with
        "serial", "chain-like" robots and monodirectional modules. For the sake of simplicity (makes implementation for
        partners easier), proximal (here: female) connectors are pointing INSIDE the module. To work in the more general
        case this toolbox is dealing with, they have to be "turned to the outside".
        Furthermore, CONCERT defines child geometries relative to joint frames, while we would define them relative to
        their body frames (for CONCERT, the "body" frame always aligns with the connector frame). Therefore, geometry
        placements after joints need to be offset.
        :param package_dir: Package directory relative to which mesh file paths are defined
        :param d: A dictionary with relevant meta-information
        :return: An instantiated module
        """
        module = cls.from_json_data(d, package_dir)
        for connector in module.available_connectors.values():
            if connector.gender is Gender.female:
                connector.body2connector = connector.body2connector @ spatial.rotX(np.pi)

        # Concert defines child geometries relative to the joint frame
        for joint in module.joints:
            child = joint.child_body
            offset = joint.child2joint
            child.visual.placement = child.visual.placement @ offset
            child.collision.placement = child.collision.placement @ offset
        return module

    @classmethod
    def from_json_data(cls, d: Dict, package_dir: Path) -> AtomicModule:
        """
        Maps the json module description to an instance of this class.

        The dictionary will be modified in-place until empty (everything was parsed).

        :param package_dir: Package directory relative to which mesh file paths are defined
        :param d: A dictionary with relevant meta-information
        :return: An instantiated module
        """
        header = d.pop('header')
        header = ModuleHeader(**header)

        if 'bodies' in d:
            bodies = [Body.from_json_data(body_data, package_dir)
                      for body_data in possibly_nest_as_list(d.pop('bodies'))]
            body_name_to_instance = {b._id: b for b in bodies}
            if 'joints' in d:
                joints = [Joint.from_json_data(joint_data, body_name_to_instance)
                          for joint_data in possibly_nest_as_list(d.pop('joints'))]
            else:
                joints = ()
        else:
            if 'joints' in d:
                raise ValueError("Joints specified without any bodies!")
            bodies = ()
            joints = ()

        if len(d) > 0:
            raise ValueError("Unrecognized keys in module specification: " + str(d.keys()))

        return cls(header, bodies, joints)

    @classmethod
    def from_json_string(cls, s: str, package_dir: Path) -> AtomicModule:
        """
        Maps the json module string to an instance of this class.

        :param package_dir: Package directory relative to which mesh file paths are defined
        :param s: A json string with relevant meta-information
        :return: An instantiated module
        """
        return cls.from_json_data(json.loads(s), package_dir)


class ModulesDB(SingleSet):
    """
    A Database of Modules. The inheritance from SingleSet ensures no duplicates are within one DB.

    Also, this means this class offers many useful methods inherited from Set, such as add() and update().

    This class provides safety regarding uniqueness of module, joint, body and connector IDs, but this comes at a price.
    Adding and removing elements can be expensive with growing modules. This class is not intended to be changed much,
    after being instantiated!
    """

    connection_type = Tuple[ModuleBase, Connector, ModuleBase, Connector]  # More specific than in Assembly!

    def __contains__(self, item: ModuleBase) -> bool:
        """
        If a new module should be added to a Module DB, the following properties must be preserved:

        (All of the below holds for modules and their sub-modules and their sub-sub-modules, and...
        to keep it short, all of those are just described as "modules")

          - All module IDs in the DB are unique
          - All module names in the DB are unique
          - All JointIDs in the DB are unique
          - All BodyIDs in the DB are unique
          - All ConnectorIDs in the DB are unique

        :param item: A module
        :return: Boolean indicator whether the module OR ANY OF THE INHERENT IDs are already in the DB
        """
        if item.id in self.all_module_ids:
            return True
        if item.name in self.all_module_names:
            return True
        if item.bodies.intersection(self.all_bodies):
            return True
        if item.joints.intersection(self.all_joints):
            return True
        return False

    def add(self, element: AtomicModule) -> None:
        """
        As the connector ID is composed of (module ID, body ID, connector ID), we need an additional check,

        we don't want two allow two connectors with same names in a module, even if attached to different bodies.
        """
        # if any(con_id in self.all_connector_own_ids for con_id in element._connectors_by_own_id):
        #     raise err.UniqueValueError("There cannot be two connectors with identical IDs in a module")
        super().add(element)

    @classmethod
    def from_file(cls, filepath: Union[Path, str], package_dir: Union[Path, str]) -> ModulesDB:
        """
        Loads a modules Database from a json file.

        :param filepath: Path to the json file of the modulesDB definition.
        :param package_dir: Package directory relative to which mesh file paths are defined
        :return: A ModulesDB
        """
        filepath, package_dir = map(Path, (filepath, package_dir))
        with filepath.open('r') as json_file:
            content = json_file.read()
        return cls.from_json_string(content, package_dir)

    @classmethod
    def from_json_string(cls, json_string: str, package_dir: Path) -> ModulesDB:
        """
        Loads a modules Database from a json string.

        :param json_string: Json string of the modulesDB definition.
        :param package_dir: Package directory relative to which mesh file paths are defined
        :return: A ModulesDB
        """
        db = cls()

        for module_data in json.loads(json_string):
            db.add(AtomicModule.from_json_data(module_data, package_dir))
        return db

    def to_json_file(self, save_at: Union[Path, str]):
        """
        Writes the ModulesDB to a json file.

        :param save_at: File location to write the DB to. Should be the "name" of the DB.
        """
        save_at = Path(save_at)
        content = compress_json_vectors(self.to_json_string())
        with Path(save_at).open('w') as savefile:
            savefile.write(content)

    def to_json_string(self) -> str:
        """
        Writes the ModulesDB to a json string.

        :return: The json string
        """
        return json.dumps([mod.to_json_data() for mod in sorted(self, key=lambda mod: mod.id)], indent=2)

    @property
    def all_module_names(self) -> Set[str]:
        """All names of modules in the DB"""
        return set(mod.name for mod in self)

    @property
    def all_module_ids(self) -> Set[str]:
        """All IDs of modules in the DB"""
        return set(mod.id for mod in self)

    @property
    def all_bodies(self) -> BodySet:
        """All bodies in all modules in the DB"""
        return BodySet(itertools.chain.from_iterable(mod.bodies for mod in self))

    @property
    def all_joints(self) -> JointSet:
        """All joints in all modules in the DB"""
        return JointSet(itertools.chain.from_iterable(mod.joints for mod in self))

    @property
    def all_connector_own_ids(self) -> Set[str]:
        """
        All local connector ids in the DB

        (not full connector IDs, which would consist of [module, body, connectorID])
        """
        raise NotImplementedError("Deprecated. No guarantee for unique own ids in Modules DB")
        # return set(con._id for con in itertools.chain.from_iterable(body.connectors for body in self.all_bodies))

    @property
    def all_connectors(self) -> ConnectorSet:
        """All connectors of all bodies of all modules in the DB"""
        return ConnectorSet(itertools.chain.from_iterable(body.connectors for body in self.all_bodies))

    @property
    def all_connectors_by_id(self) -> Dict[Tuple[str, str, str], Connector]:
        """All complete connecor IDs in the DB (counter piece to all_connector_own_ids, which is basically a subset)"""
        return {c.id: c for c in self.all_connectors}

    @property
    def by_id(self) -> Dict[str, ModuleBase]:
        """Returns this DB as a dictionary, mapping the Module ID to the module"""
        return {mod.id: mod for mod in self}

    @property
    def by_name(self) -> Dict[str, ModuleBase]:
        """Returns this DB as a dictionary, mapping the Module Name to the module"""
        return {mod.name: mod for mod in self}

    @property
    def bases(self) -> ModulesDB[ModuleBase]:
        """Returns all modules containing at least one base connector"""
        return self.__class__(mod for mod in self if any(c.type == 'base' for c in mod.available_connectors.values()))

    @property
    def end_effectors(self) -> ModulesDB[ModuleBase]:
        """Returns all modules containing at least one eef connector"""
        return self.__class__(mod for mod in self if any(c.type == 'eef' for c in mod.available_connectors.values()))

    @property
    def possible_connections(self) -> Set[connection_type]:
        """
        Returns a set of all possible connections in this db.

        As connections are symmetric, only one copy of {A --> B, B --> A} will be returned.
        """
        connections: Set[ModulesDB.connection_type] = set()
        for mod_a, mod_b in itertools.combinations_with_replacement(self, 2):
            for con_a, con_b in itertools.product(mod_a.available_connectors.values(),
                                                  mod_b.available_connectors.values()):
                if con_a.connects(con_b):
                    if mod_a is mod_b and ((mod_a, con_a, mod_b, con_b) in connections):
                        # Avoiding duplicates
                        continue
                    connections.add((mod_a, con_a, mod_b, con_b))
        return connections

    @property
    def connectivity_graph(self) -> nx.DiGraph:
        """
        Returns a graph of all possible module connections in the db.

        :return: An undirected graph
        """
        G = nx.DiGraph()
        for mod_a, con_a, mod_b, con_b in self.possible_connections:
            G.add_edge(mod_a, mod_b, connectors=(con_a, con_b))
            G.add_edge(mod_b, mod_a, connectors=(con_b, con_a))
        return G

    def debug_visualization(self,  # pragma: no cover
                            viz: pin.visualize.MeshcatVisualizer = None,
                            stride: float = 1) -> pin.visualize.MeshcatVisualizer:
        """
        Show debug visualization of all contained modules (show them in a square array)

        :param viz: MeshcatVisualizer to use
        :param stride: Distance between the plotted modules in meters (default: 1m)
        :return: MeshcatVisualizer used
        """
        if viz is None:
            viz = pin.visualize.MeshcatVisualizer()
            viz.initViewer()

        cols = math.floor(math.sqrt(len(self.all_module_ids)))

        for i, module_idx in enumerate(sorted(list(self.all_module_ids))):
            self.by_id[module_idx].debug_visualization(
                viz=viz, base_placement=spatial.homogeneous((stride * math.floor(i / cols), stride * i % cols, 0)))

        return viz


class ModuleAssembly:
    """
    A combination of modules with defined connections between each another.

    High level abstraction of a robot.
    """

    connection_type = Tuple[int, str, int, str]

    def __init__(self,
                 database: ModulesDB,
                 assembly_modules: Collection[str] = (),
                 connections: Collection[connection_type] = (),
                 base_connector: Tuple[int, str] = None
                 ):
        """
        A combination of modules, drawn from a common module database and arranged by defined connections.

        :param database: A ModulesDB instance holding all relevant modules for the assembly
        :param assembly_modules: This list defines which modules are within the assembly and also defines their index
           position when referencing single modules. This list holds module IDs, but opposed to a ModuleSet, IDs can
           be present multiple times. (Read: If multiple copies of the conceptually same module should be used in an
           assembly) The order itself is arbitrary (but will be used for further referencing), but in case of tree- or
           chain-like assemblies, it is recommended to provide the ID's in the order of the kinematic chain.
        :param connections: Defines how the modules are connected to each other. Each connection consists of a tuple
          that specifies (module_a, c_a, module_b, c_b), where c_a and c_b are the connectors used for the connection
          between modules module_a and module_b. Connectors can either be provided as instance reference or via ID.
          Modules should be provided as indices of the module_instances list.
        :param base_connector: For every assembly, there must be a 'base' connector that indicates where the
          base of the robot coordinate system is. If it is not explicitly given, there must be exactly one connector of
          type 'base' in the assembly. The base connector argument is expected to be of type [module_idx, connector._id]
        """
        self.db: ModulesDB = database
        self.module_instances: List[ModuleBase] = []
        self.connections: Set[Tuple[ModuleBase, Connector, ModuleBase, Connector]] = set()

        self._module_copies: Dict[str, str] = dict()
        if isinstance(assembly_modules, str):
            # A string is a collection - but we won't allow that, it only leads to confusion
            raise ValueError("Argument assembly_modules must be a collection of strings, not a string!")
        for module_id in assembly_modules:
            self._add_module(module_id)

        # Unpack the connections and establish object references if int/str references were given
        for mod_a, con_a, mod_b, con_b in connections:
            if isinstance(mod_a, int):
                mod_a = self.module_instances[mod_a]
            else:
                raise ValueError("Non-integer module specifiers are deprecated")
            if isinstance(mod_b, int):
                mod_b = self.module_instances[mod_b]
            else:
                raise ValueError("Non-integer module specifiers are deprecated")
            if isinstance(con_a, str):
                con_a = mod_a.connectors_by_own_id[con_a]
            else:
                raise ValueError("Non-id connector specifiers are deprecated")
            if isinstance(con_b, str):
                con_b = mod_b.connectors_by_own_id[con_b]
            else:
                raise ValueError("Non-id connector specifiers are deprecated")
            self.connections.add((mod_a, con_a, mod_b, con_b))

        try:
            self._assert_is_valid()
        except AssertionError:
            raise err.InvalidAssemblyError("Robot Assembly seems to be invalid. Check the input arguments.")

        self._base_connector = None
        if base_connector is None:
            con2base = [con for con in self.free_connectors.values() if con.type == 'base']
            if len(con2base) == 1:
                self._base_connector = con2base[0]
            elif len(self.module_instances) > 0:
                raise ValueError(
                    "There are {} candidates for a base connector in the assembly.".format(len(con2base)))
        else:
            idx, own_id = base_connector
            if idx not in range(len(self.module_instances)):
                raise ValueError("Invalid module specifier for the base connector parent module.")
            try:
                base_connector = self.module_instances[idx].connectors_by_own_id[own_id]
            except KeyError:
                raise ValueError(f"The connector ID you provided ({own_id}) is not available "
                                 f"in module {self.internal_module_ids[idx]}.")
            if base_connector.id not in self.free_connectors:
                raise err.InvalidAssemblyError("The base connector seems to be occupied")
            self._base_connector = base_connector

    @classmethod
    def from_serial_modules(cls, db: ModulesDB, module_chain: Iterable[str]) -> 'ModuleAssembly':
        """
        This function works on the assumption that the robot modules are arranged in a chain.

        :param db: The database used to build the robot
        :param module_chain: The series of modules in the robot, referring to the module ids in the db.
        """
        by_id = db.by_id
        assembly = cls(db)
        for i, module_id in enumerate(module_chain):
            if i == 0:
                assembly._add_module(module_id, set_base=True)
            else:
                module_instance = by_id[module_id]
                previous = i - 1
                possible_connections = [(module_instance, con_id[-1], previous, to_con_id)
                                        for con_id, mod_con in module_instance.available_connectors.items()
                                        for to_con_id, to_con in assembly.free_connectors.items()
                                        if mod_con.connects(to_con)]
                if len(possible_connections) != 1:
                    raise err.InvalidAssemblyError(f"There is no unique way to add {module_instance} to the robot.")
                connection = possible_connections[0]
                if connection[3][0] != assembly.internal_module_ids[-1]:
                    raise ValueError("The provided modules do not build a chain, so the ordering is unambiguous.")
                assembly.add(*connection)

        return assembly

    def _assert_is_valid(self):
        """Sanity checks to ensure the assembly has a unique and well-defined meaning"""
        no_module_error = "You defined a connection containing module: {} , which is not in module_instances."
        no_connector_error = "You defined a connection with connector <{}> and module <{}>," \
                             " which does not contain such a connector."
        # Modules can appear multiple times in the connections, as they can have multiple connectors, so in this case,
        # set is the right container and to be preferred over the uniqueness-requiring ModuleSet.
        seen_modules = set()
        seen_connectors: ConnectorSet = ConnectorSet()  # Tuples of connectorID-parentID
        for module in self.module_instances:
            if module.id not in self._module_copies:
                assert module in self.db, f"You provided a module that is not in the database: {module}"

        for (mod_a, con_a, mod_b, con_b) in self.connections:
            # First check that modules and connectors are unique and customize errors if not
            try:
                seen_modules.update((mod_a.id, mod_b.id))
            except err.UniqueValueError:
                # This also will be raised when mod_a == mod_b -> No self-loops allowed
                raise err.UniqueValueError(f"There are multiple identical instances of either module {mod_a}"
                                           f" or module {mod_b} in the assembly.")
            try:
                for con in (con_a, con_b):
                    seen_connectors.add(con)
            except err.UniqueValueError:
                raise err.UniqueValueError(f"Connector {con} is used multiple times in the assembly.")

            # Check that the objects in the connections are actually contained in the assembly
            assert mod_a.id in self.internal_module_ids, no_module_error.format(mod_a)
            assert mod_b.id in self.internal_module_ids, no_module_error.format(mod_b)
            assert con_a in mod_a.available_connectors.values(), no_connector_error.format(con_a, mod_a)
            assert con_b in mod_b.available_connectors.values(), no_connector_error.format(con_b, mod_b)
            # Check the connection actually works
            assert con_a.connects(con_b), f"You defined an invalida connection between {con_a} and {con_b}."

        # Lastly, check there are no "loose" modules
        if len(self.module_instances) > 1 and not len(seen_modules) == len(self.module_instances):
            raise ValueError(f"Not every module in the assembly seems to be connect!\n"
                             f"Connected: {len(seen_modules)}\n"
                             f"Modules in the assembly: {len(self.module_instances)}")

    def _add_module(self, module_id: str, set_base: bool = False) -> int:
        """
        Takes modules from a database and ensures duplicate module_ids do not lead to identical module instances.

        The added module is not connected - use self.add() method to properly add a new module from outside this class

        :param module_id: The module id - the right instance will be drawn from the internal db
        :param set_base: Needs to be set if the first module is added externally to an empty assembly.
        :return: Returns the module index (as in the internal module order)
        """
        if isinstance(module_id, ModuleBase):  # Common error
            raise ValueError("You provided a Module, but an ID was expected.")
        db_by_id = self.db.by_id
        previous_identical = module_id in self.internal_module_ids

        if set_base:
            module = db_by_id[module_id]
            if len(self.internal_module_ids) != 0:
                raise NotImplementedError("Cannot reset the base after having set one already.")
            base_connectors = [c for c in module.available_connectors.values() if c.type == 'base']
            if len([c for c in module.available_connectors.values() if c.type == 'base']) != 1:
                raise err.InvalidAssemblyError(
                    "A base module needs to have exactly one base connector if added this way.")
            self._base_connector = base_connectors[0]

        if not previous_identical:
            self.module_instances.append(db_by_id[module_id])
        else:
            i = 1
            while True:
                # Find the first free module id + suffix id and add the module as such
                suffix = f'-{i}'
                if (module_id + suffix) not in self.internal_module_ids:
                    new_module = db_by_id[module_id].copy(f'-{i}')
                    self._module_copies[new_module.id] = module_id
                    self.module_instances.append(new_module)
                    break
                i += 1

        return len(self.module_instances) - 1

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """
        Returns the adjacency matrix for the assembly graph.

        Assumes every module is a node and links are tuples of connectors (as defined in self.connections). This method
          assumes undirected edges and no self-loops.

        :return: An NxN matrix, where N=#modules and A(i, j)=#connections between i and j
        """
        A = np.zeros((self.nModules, self.nModules), dtype=int)
        for mod_a, mod_b in itertools.combinations(range(self.nModules), 2):
            A[mod_a, mod_b] = len(self.connections_between(mod_a, mod_b))
        return A + A.T

    @property
    def base_connector(self) -> Connector:
        """The first body in the kinematic chain of the robot"""
        return self._base_connector

    @property
    def free_connectors(self) -> Dict[Tuple[str, str, str], Connector]:
        """Returns a dict of all connectors in the Assembly that are not connected currently"""
        all_connectors = dict()
        for module in self.module_instances:
            all_connectors.update(module.available_connectors)
        return {con_id: con for con_id, con in all_connectors.items() if not (
                any(con in connection for connection in self.connections)
                or (con is self.base_connector))
                }

    @property
    def internal_module_ids(self) -> Tuple[str]:
        """
        Returns the module IDs for the Assemblies Modules the way they are stored internally

        :note: this might differ significantly from the IDs in the db!
        """
        return tuple(mod.id for mod in self.module_instances)

    @property
    def graph_of_modules(self) -> nx.Graph:
        """The graph of modules represents all modules in an assembly and how they are connected to each other.

        Returns a networkx graph object with a node for every module in the assembly and connections for every
          connector-connector pair defined by one of the assemblies connections.
          Not to be confused with the module_graph (a graph depicting the structure of the elements within a single
          module), nor the assembly_graph (a more verbose variant of the graph_of_modules, where instead of module
          nodes, every module is represented by its module_graph)
        """
        G = nx.Graph()
        G.add_nodes_from(mod.id for mod in self.module_instances)
        for mod_a, con_a, mod_b, con_b in self.connections:
            G.add_edge(mod_a.id, mod_b.id, label=[con_a._id, con_b._id])
        return G

    @property
    def mass(self) -> float:
        """The total mass of the assembly"""
        return sum(mod.mass for mod in self.module_instances)

    @property
    def nJoints(self) -> int:
        """The number of joints in the assembly"""
        return sum(len(mod.joints) for mod in self.module_instances)

    @property
    def nModules(self) -> int:
        """The number of modules in the assembly"""
        return len(self.module_instances)

    @property
    def assembly_graph(self) -> nx.DiGraph:
        """
        Returns a networkx graph representing the internal modules in the assembly on a body, connector and joint level.

        Nodes represent either of (body, joint, connector) while the directed edges define child-parent relations
        between them with their edge properties containing information about the relative transformation from one of the
        objects to another.
        """
        # TODO: Include the universe in this graph and include checks for connectivity with only 1 module as well
        if len(self.module_instances) == 1:
            return self.module_instances[0].module_graph

        G_a = nx.DiGraph()
        rotX = Transformation(spatial.rotX(np.pi))  # Necessary to align connector coordinate systems

        for mod_a, con_a, mod_b, con_b in self.connections:
            # First, add the module graphs
            for u, v, transform in mod_a.module_graph.edges.data('transform'):
                G_a.add_edge(u, v, transform=transform)
            for u, v, transform in mod_b.module_graph.edges.data('transform'):
                G_a.add_edge(u, v, transform=transform)

            # Then, connect the module graphs accordingly
            G_a.add_edge(con_a, con_b, transform=rotX)
            G_a.add_edge(con_b, con_a, transform=rotX)

        return G_a

    def add(self,
            new: ModuleBase,
            new_connector: str,
            to: int,
            to_connector: Union[str, Tuple[str, str, str]]) -> int:
        """
        Add a new module to the assembly. Works with ``add(*connection)``.

        :param new: The new module to add
        :param new_connector: The connector's own id to use in the new connection
        :param to: The module to add the module to
        :param to_connector: The connectors own id in module "to" that is used
            for the connection. Alternatively, the full connector id.
        :return: new_module_index
        """
        if isinstance(to_connector, tuple):
            assert to_connector[0] == self.internal_module_ids[to], "Connector ID and module index do not match"
            to_connector = to_connector[-1]  # take the "own id"
        to = self.module_instances[to]
        to_connector = to.connectors_by_own_id[to_connector]

        new_index = self._add_module(new.id)
        new = self.module_instances[-1]
        new_connector = new.connectors_by_own_id[new_connector]
        if not new_connector.connects(to_connector):
            raise err.InvalidAssemblyError("Invalid connection provided. Connectors do not match.")
        self.connections.add((new, new_connector, to, to_connector))

        return new_index

    def add_random_from_db(self, db_filter: Callable[[ModuleBase], bool] = None) -> int:
        """
        Adds a random module from the database to this assembly

        :param db_filter: If provided, only modules that fulfill the filter function will be taken into consideration
        :return: See add()
        """
        if db_filter is None:
            db = self.db
        else:
            db = self.db.filter(db_filter)

        if len(self.module_instances) == 0:
            bases = [con for con in self.db.all_connectors if con.type == 'base']
            if len(bases) == 0:
                raise LookupError("There are no possible base connectors in the database.")
            base = random.choice(bases)
            self._add_module(base.parent.in_module.id)
            self._base_connector = base  # This is the first module, so the instance stays the same
            return 0

        for con_new, con_present in itertools.product(randomly(db.all_connectors),
                                                      randomly(self.free_connectors.values())):
            if con_present.connects(con_new):
                return self.add(con_new.parent.in_module, con_new.own_id,
                                self.module_instances.index(con_present.parent.in_module), con_present.own_id)

        raise LookupError("There is no possible connection that could be added")

    def connections_between(self, mod_a: int, mod_b: int) -> Set[connection_type]:
        """
        Returns all connections that exist between module mod_a and module mod_b.

        :note: as connections are bidirectional, connections_between(a, b) == connections_between(b, a).

        :param mod_a: Either a module reference or the index of the module
        :param mod_b: Either a module reference or the index of the module
        :return: A set (so, unordered) of connections between the two modules
        """
        mod_a = self.module_instances[mod_a]
        mod_b = self.module_instances[mod_b]
        return {connection for connection in self.connections if (mod_a in connection and mod_b in connection)}

    def plot_graph(self):
        """Draws the assembly graph."""
        G = self.assembly_graph
        plt.subplots(figsize=(12, 12))
        pos = nx.spring_layout(G, k=1.1 / (len(G) ** .5), scale=1.2)
        nx.draw(G, pos=pos, with_labels=True)
        plt.tight_layout()
        plt.show()

    def to_pin_robot(self,
                     base_placement: TransformationLike = Transformation.neutral(),
                     add_com_frames: bool = False,
                     collisions_between_neighboring_bodies: bool = False) -> Robot.PinRobot:
        """
        Creates a pinocchio robot from a module tree

        :param base_placement: placement for the (single) base connector as 4x4 hom. transformation
        :param add_com_frames: Add frames for the center of mass of each Body
        :param collisions_between_neighboring_bodies: If True, any two bodies in the resulting robot can collide as long
          as they are not rigidly connected. If false, bodies connected only by a joint are not checked for collisions.
          The latter one allows defining overlapping geometries, but in that case collision-safety must be provided by
          joint limits.
        :return: pinocchio robot for this assembly
        """
        if len(self.module_instances) == 0:
            raise ValueError("Cannot create a robot model from an empty assembly!")
        G = self.assembly_graph
        edges = self.assembly_graph.edges
        robot = Robot.PinRobot(wrapper=pin.RobotWrapper(pin.Model()))
        body_collision_geometries = dict()

        parent_joints = {'universe': 0, self.base_connector.id: 0}
        no_update = {'update_kinematics': False, 'update_home_collisions': False}
        new_frame = robot.model.addFrame(pin.Frame('.'.join(self.base_connector.id),
                                                   0,
                                                   robot.model.getFrameId('universe'),
                                                   pin.SE3(spatial.rotX(np.pi)),
                                                   pin.FrameType.OP_FRAME)
                                         )

        # work with np, not with transformations - pin needs np anyways
        transforms: Dict[Tuple[str, ...], np.ndarray] = {self.base_connector.id: spatial.rotX(np.pi)}
        frames = {self.base_connector.id: new_frame}
        for node, successors in nx.bfs_successors(G, self.base_connector):
            for successor in successors:
                transform = transforms[node.id] @ edges[node, successor]['transform'].homogeneous
                transforms[successor.id] = transform
                if isinstance(successor, BodyBase):
                    if not isinstance(successor, Body):
                        successor = successor.freeze()
                    new_frame, new_geometries = robot._add_body(successor.as_pin_body(transform),
                                                                parent_joints[node.id],
                                                                frames[node.id])
                    frames[successor.id] = new_frame
                    parent_joints[successor.id] = parent_joints[node.id]
                    body_collision_geometries[successor.id] = new_geometries
                    if add_com_frames:  # Add center of mass as a child frame of this body
                        new_frame = robot \
                            .model \
                            .addFrame(pin.Frame('.'.join(successor.id + ("CoM",)),
                                                parent_joints[successor.id],
                                                frames[successor.id],
                                                pin.SE3((
                                                                Transformation(transforms[successor.id]) @
                                                                Transformation.from_translation(successor.inertia.lever)
                                                        ).homogeneous),
                                                pin.FrameType.OP_FRAME))
                        frames[successor.id + ("CoM",)] = new_frame
                        parent_joints[successor.id + ("CoM",)] = parent_joints[successor.id]
                elif isinstance(successor, Joint) and successor.type is not TimorJointType.fixed:
                    options = successor.pin_joint_kwargs
                    options.update(no_update)  # Robot data will be updated once only - at the very end
                    options['joint_placement'] = pin.SE3(transform)  # Relative placement to parent joint
                    new_joint_id = robot._add_joint(
                        joint=successor.pin_joint_type(),
                        parent=parent_joints[node.id],
                        bodies=(), fixed_joints=(),
                        **options
                    )
                    parent_joints[successor.id] = new_joint_id
                    transforms[successor.id] = Transformation.neutral().homogeneous
                    frames[successor.id] = robot.model.getFrameId(options['joint_name'])
                elif isinstance(successor, Connector) or \
                        (isinstance(successor, Joint) and successor.type is TimorJointType.fixed):
                    new_frame = robot.model.addFrame(
                        pin.Frame('.'.join(successor.id), parent_joints[node.id], frames[node.id], pin.SE3(transform),
                                  pin.FrameType.FIXED_JOINT))
                    frames[successor.id] = new_frame
                    parent_joints[successor.id] = parent_joints[node.id]
                else:
                    raise NotImplementedError("Graph can only contain bodies, joints and connectors")

        # Now update robot kinematic and collision data
        robot._tcp = robot.model.nframes - 1
        robot._update_kinematic()
        robot.set_base_placement(Transformation(base_placement))

        if not collisions_between_neighboring_bodies:
            # Remove collisions between bodies connected by a joint
            neighbors_to_clean = set()
            for node in G.nodes:
                for neighbor in G.neighbors(node):
                    if isinstance(neighbor, Joint):
                        for potential_remove in G.neighbors(neighbor):
                            if isinstance(potential_remove, Body) and not (potential_remove is node):
                                neighbors_to_clean.add((node.id, potential_remove.id))
                                logging.debug(f"Removing body pair ({node.id}, {potential_remove.id}) from collisions")
            cps_to_remove = list()
            for b1, b2 in neighbors_to_clean:
                cps_to_remove.extend([[first, second] for first, second in itertools.product(
                    body_collision_geometries[b1], body_collision_geometries[b2])])

            original = robot.collision_pairs
            active = np.zeros((robot.collision.ngeoms, robot.collision.ngeoms), bool)  # defaults all to false
            for cg1, cg2 in original:
                if [cg1, cg2] not in cps_to_remove:
                    active[cg1, cg2] = True
            robot.collision.setCollisionPairs(active)
            robot.collision_data = robot.collision.createData()

        return robot

    def to_urdf(self, name: str = None, write_to: Optional[Path] = None, replace_wrl: bool = False) -> str:
        """
        Creates a URDF file from the assembly.

        :param name: name to be given to robot inside URDF
        :param write_to: output URDF file
        :param replace_wrl: Whether to replace wrl geometries with visual geometry (if available).
        """
        G = self.assembly_graph
        edges = self.assembly_graph.edges
        if name is None:
            name = 'Auto-Generated: ' + uuid.uuid4().hex  # Random Name

        def get_urdf_joint_type(joint_instance: Joint) -> str:
            pin2urdf = {pin.JointModelRZ: 'revolute', pin.JointModelPZ: 'prismatic'}
            if joint_instance.pin_joint_type not in pin2urdf:
                raise NotImplementedError()
            return pin2urdf[joint_instance.pin_joint_type]

        urdf = ET.Element('robot', {'name': name})
        links = []
        joints = []

        transforms = {self.base_connector.id: spatial.rotX(np.pi)}
        for node, successors in nx.bfs_successors(G, self.base_connector):
            for successor in successors:
                transform = transforms[node.id] @ edges[node, successor]['transform'].homogeneous
                transforms[successor.id] = transform
                if isinstance(successor, BodyBase):
                    if not isinstance(successor, Body):
                        successor = successor.freeze()
                    body_name = '.'.join(successor.id)
                    link = ET.Element('link', {'name': body_name})
                    link.append(write_urdf.from_pin_inertia(successor.inertia, transform))

                    visual_geom = None
                    if not isinstance(successor.visual, Geometry.EmptyGeometry):
                        visual_geom = write_urdf.from_geometry(successor.visual, 'visual',
                                                               name=body_name + '_visual', link_frame=transform)
                        link.append(visual_geom)

                    if not isinstance(successor.collision, Geometry.EmptyGeometry):
                        coll_geom = write_urdf.from_geometry(successor.collision, 'collision',
                                                             name=body_name + '_collision', link_frame=transform)
                        if replace_wrl:
                            # Find all wrl references and replace with geometry found in visual description
                            if visual_geom is None:
                                raise ValueError("Cannot replace wrl if there is no visual geometry to replace with.")
                            for m in coll_geom.findall('geometry/mesh'):
                                if re.search(r"\.wrl$", m.get("filename")) is not None:
                                    coll_geom = ET.Element("collision", {"name": coll_geom.get("name")})
                                    for child in visual_geom:
                                        coll_geom.append(child)
                                    logging.info(f"Replaced geometry with wrl file by visual geom: {m.get('filename')}")
                        link.append(coll_geom)

                    links.append(link)
                elif isinstance(successor, Joint) and successor.type is not TimorJointType.fixed:
                    joint = ET.Element('joint', {'name': '.'.join(successor.id),
                                                 'type': get_urdf_joint_type(successor)})
                    ET.SubElement(joint, 'origin', {'xyz': ' '.join(map(str, transform[:3, 3])),
                                                    'rpy': ' '.join(map(str, spatial.mat2euler(transform[:3, :3])))})
                    ET.SubElement(joint, 'parent', {'link': '.'.join(successor.parent_body.id)})
                    ET.SubElement(joint, 'child', {'link': '.'.join(successor.child_body.id)})
                    ET.SubElement(joint, 'axis', {'xyz': '0 0 1'})
                    ET.SubElement(joint, 'dynamics', {'damping': str(successor.friction_viscous),
                                                      'friction': str(successor.friction_coulomb)})
                    ET.SubElement(joint, 'limit', {'lower': str(successor.limits.squeeze()[0]),
                                                   'upper': str(successor.limits.squeeze()[1]),
                                                   'effort': str(successor.torque_limit),
                                                   'velocity': str(successor.velocity_limit)})
                    joints.append(joint)
                    transforms[successor.id] = Transformation.neutral().homogeneous
                    # Gear Ratio is missing but seemingly not supported by URDF
                elif isinstance(successor, Connector):
                    if isinstance(node, Connector):  # Only add the second connector of each connection
                        joint = ET.Element('joint', {'name': '.'.join(successor.id),
                                                     'type': 'fixed'})
                        ET.SubElement(joint, 'origin', {'xyz': ' '.join(map(str, transform[:3, 3])),
                                                        'rpy': ' '.join(map(str, spatial.mat2euler(transform[:3, :3])))
                                                        })
                        ET.SubElement(joint, 'parent', {'link': '.'.join(node.parent.id)})
                        ET.SubElement(joint, 'child', {'link': '.'.join(successor.parent.id)})
                        joints.append(joint)
                        transforms[successor.id] = Transformation.neutral().homogeneous
                elif isinstance(successor, Joint) and successor.type is TimorJointType.fixed:
                    assert False, "Did not think about that yet"
                else:
                    raise NotImplementedError("Graph can only contain bodies, joints and connectors")

        for subtree in links + joints:
            urdf.append(subtree)

        try:
            ET.indent(urdf, space='\t', level=0)
        except AttributeError:
            # indent was introduced in Python 3.9 -- for previous versions, we just don't pretty-print
            pass
        urdf_string = ET.tostring(urdf, 'unicode')

        if write_to is not None:
            with write_to.open('w') as writefile:
                writefile.write(urdf_string)

        return urdf_string
