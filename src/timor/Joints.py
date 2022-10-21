#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.02.22
from enum import Enum
import inspect
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import pinocchio as pin

from timor.Bodies import BodyBase, ConnectorSet
from timor.utilities.dtypes import SingleSet, fuzzy_dict_key_matching
import timor.utilities.errors as err
from timor.utilities.transformation import Transformation, TransformationLike


class TimorJointType(Enum):
    """Enumerates joint types available in Timor"""

    revolute = 0
    continuous = 0
    prismatic = 1
    # Aliases
    revolute_active = 0
    prismatic_active = 1
    # Passive Joints
    revolute_passive = 2
    prismatic_passive = 2
    fixed = 3

    def __str__(self) -> str:
        """Ensures defined string casting behavior when storing a joint of a certain type as json"""
        defaults = {0: 'revolute', 1: 'prismatic', 2: 'passive', 3: 'fixed'}
        return defaults[self.value]


class Joint:
    """
    A joint is a bodyless axis.

    Masses, inertias etc. are contained in the parent and child bodies (except for the simplified motor inertia).
    What is called a "joint" in everyday language (the hardware part you attach to a robot) would, in this definition,
    be a chain of parent_body -> joint -> child body.
    """

    def __init__(self,
                 joint_id: str,
                 joint_type: Union[str, TimorJointType],
                 parent_body: BodyBase,
                 child_body: BodyBase,
                 in_module: 'ModuleBase' = None,
                 q_limits: Union[np.ndarray, List[float], Tuple[float, float]] = (-np.inf, np.inf),
                 torque_limit: float = np.inf,
                 velocity_limit: float = np.inf,
                 acceleration_limit: float = np.inf,
                 parent2joint: TransformationLike = Transformation.neutral(),
                 joint2child: TransformationLike = Transformation.neutral(),
                 gear_ratio: float = 1.,
                 motor_inertia: float = 0.,
                 friction_coulomb: float = 0.,
                 friction_viscous: float = 0.
                 ):
        """
        Holds all important joint meta-information. Not intended for calculations (kinematics / dynamics).

        :param joint_id: Unique joint identifier. Has to be unique within each module set or robot it is in,
        :param parent_body: The body the joint is attached to
        :param child_body: The child body that is attached to the joint
        :param in_module: The parent module this Joint belongs to
        :param q_limits: A tuple/array of (lower_limit, upper_limit) for the joint parameter q
        :param velocity_limit: Maximum angular velocity in rad/s
        :param acceleration_limit: Maximum angular acceleration in rad/s**2
        :param torque_limit: Maximum joint torque in Nm
        :param parent2joint: The homogeneous transformation from parent body coordinate system to
          joint coordinate system
        :param joint2child: The homogeneous transformation from joint coordinate system to child body coordinate system
        :param gear_ratio: The gear ratio of a implicit gear in the joint, defined as w_in / w_out
        :param motor_inertia: The inertia of a motor sitting in this joint
        :param friction_coulomb: Coulomb friction parameter
        :param friction_viscous: Viscous friction parameter
        """
        self._id: str = str(joint_id)
        self._type: TimorJointType = TimorJointType[joint_type] if isinstance(joint_type, str) else joint_type
        if parent_body.id == child_body.id:
            raise err.UniqueValueError("Child and Parent Body IDs cannot be the same.")
        self.parent_body: BodyBase = parent_body
        self.child_body: BodyBase = child_body
        self.in_module: 'ModuleBase' = in_module
        try:
            con = self.all_connectors  # noqa: F841
        except err.UniqueValueError:
            raise err.UniqueValueError("The parent and child bodies you provided contain overlapping connector IDs")
        if joint_type == 'continuous':  # Ros input
            assert all(abs(lmt) == np.inf for lmt in q_limits), "Continuous Joints cant have limits"
        self.limits: np.ndarray = np.asarray(q_limits, dtype=float).reshape([2, -1])  # Evaluates to nan in case of None
        self.velocity_limit: float = float(velocity_limit)
        self.acceleration_limit: float = float(acceleration_limit)
        self.torque_limit: float = float(torque_limit)
        self.gear_ratio: float = float(gear_ratio)
        self.motor_inertia: float = float(motor_inertia)
        self.friction_coulomb: float = float(friction_coulomb)
        self.friction_viscous: float = float(friction_viscous)

        self.parent2joint: Transformation = Transformation(parent2joint)
        self.joint2child: Transformation = Transformation(joint2child)

    def __str__(self):
        """Joint representation of the joint"""
        return f"Joint: {self.id}"

    @classmethod
    def from_json_data(cls, d: Dict, body_id_to_instance: Dict = None) -> 'Joint':
        """
        Maps the serialized json description to an instance of this class.

        :param d: A dictionary with relevant meta-information
        :param body_id_to_instance: A mapping from body IDs to the python instance of this body
        :return: An instantiated joint
        """
        parent = d['parent']
        child = d['child']
        if body_id_to_instance is not None:
            parent = body_id_to_instance[str(parent)]
            child = body_id_to_instance[str(child)]

        if d.get('passive', False) in (True, 'True', 'true'):
            raise NotImplementedError("Passive joints not yet implemented")

        # Unpack items from the input dictionary - even if they have a slightly modified structure from the expected
        aliases = {'velocity': 'velocity_limit',
                   'peakTorque': 'torque_limit',
                   'poseParent': 'parent2joint',
                   'poseChild': 'joint2child',
                   "gearRatio": 'gear_ratio',
                   "motorInertia": 'motor_inertia',
                   "frictionCoulomb": 'friction_coulomb',
                   "frictionViscous": 'friction_viscous'}  # Valid aliases we accept
        # Get a list of all optional (=they have a default value) parameters that we could find
        optional_arguments = tuple(name for name, param in inspect.signature(cls.__init__).parameters.items() if
                                   param.default is not inspect._empty)
        optional_kwargs = dict()  # Fill this dictionary with all possible arguments
        optional_kwargs.update(d)
        if 'limits' in d:
            optional_kwargs.update(d['limits'])
        optional_kwargs = fuzzy_dict_key_matching(optional_kwargs, aliases, optional_arguments)

        return cls(
            joint_id=d['ID'],
            joint_type=d['type'],
            parent_body=parent,
            child_body=child,
            q_limits=[
                fuzzy_dict_key_matching(d['limits'], {'lower': 'positionLower'})['positionLower'],
                fuzzy_dict_key_matching(d['limits'], {'upper': 'positionUpper'})['positionUpper']
                         ],
            **optional_kwargs
        )

    def to_json_data(self) -> Dict[str, any]:
        """
        :return: Returns the join specification in a json-ready dictionary
        """
        if self.type in (TimorJointType.revolute_passive, TimorJointType.prismatic_passive):
            raise ValueError("Passive Joint serialization not yet implemented")
        return {
            'ID': self._id,
            'parent': self.parent_body._id,
            'child': self.child_body._id,
            'type': str(self.type),
            'passive': False,
            'poseParent': self.parent2joint.serialized,
            'poseChild': self.joint2child.serialized,
            'limits': {
                'positionLower': self.limits[0, :].tolist()[0],
                'positionUpper': self.limits[1, :].tolist()[0],
                'peakTorque': self.torque_limit,
                'velocity': self.velocity_limit,
                'acceleration': self.acceleration_limit
            },
            'gearRatio': self.gear_ratio,
            'motorInertia': self.motor_inertia,
            'frictionCoulomb': self.friction_coulomb,
            'frictionViscous': self.friction_viscous
        }

    @property
    def all_connectors(self) -> ConnectorSet:
        """All connectors in the bodies attached to this joint"""
        return ConnectorSet(self.parent_body.connectors.union(self.child_body.connectors))

    @property
    def id(self) -> Tuple[Union[None, str], str]:
        """Compose ID: (module ID, joint ID)"""
        try:
            return self.in_module.id, self._id
        except AttributeError:
            return None, self._id

    @property
    def type(self) -> TimorJointType:
        """Returns the type of this joint"""
        return self._type

    @property
    def joint2parent(self) -> Transformation:
        """The homogeneous transformation from joint to parent body coordinate system"""
        return self.parent2joint.inv

    @property
    def child2joint(self) -> Transformation:
        """The homogeneous transformation from child body to joint coordinate system"""
        return self.joint2child.inv

    @property
    def pin_joint_kwargs(self) -> Dict:
        """Provides the key word arguments needed when adding a new joint to a pin robot"""
        return {
            'joint_placement': pin.SE3.Identity(),
            'joint_name': '.'.join(self.id),
            'max_effort': self.torque_limit,
            'max_velocity': self.velocity_limit,
            'min_config': self.limits[0],
            'max_config': self.limits[1],
            'gearRatio': self.gear_ratio,
            'motorInertia': self.motor_inertia,
            'friction': self.friction_coulomb,
            'damping': self.friction_viscous
        }

    @property
    def pin_joint_type(self) -> Type[pin.JointModel]:
        """Returns a reference to the pinocchio joint class that matches this joints type"""
        if self.type is TimorJointType.revolute:
            return pin.JointModelRZ
        elif self.type is TimorJointType.prismatic:
            return pin.JointModelPZ
        elif self.type is TimorJointType.fixed:
            raise TypeError("Pinocchio does not have fixed joints. They are represented as frames")
        elif (self.type is TimorJointType.revolute_passive) or (self.type is TimorJointType.prismatic_passive):
            raise TypeError("Pinocchio does not have passive joints. They are represented as frames")


class JointSet(SingleSet):
    """A set that raises an error if a duplicate joint is added"""

    def __contains__(self, item: Joint) -> bool:
        """Custom check for duplicate joints (same ID)"""
        return item.id in (jnt.id for jnt in self)
