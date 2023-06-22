from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pinocchio

from timor.task import Solution, Tolerance
from timor.utilities import logging
from timor.utilities.jsonable import JSONable_mixin
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.transformation import Transformation


class ConstraintBase(ABC, JSONable_mixin):
    """Base class defining the interface all Constraints must implement."""

    global_only: bool = False  # If this is True, the constraint cannot be used as a local goal constraint.

    @classmethod
    def from_json_data(cls, d: Dict[str, Any], *args, **kwargs) -> ConstraintBase:
        """
        Converts a description (string, parsed from json) to a constraint

        :param d: A description, mapping a constraint name to one or more constraint specifications
        """
        try:
            constraint_type = d.pop('type')
        except KeyError:
            if len(d) == 0:
                return AlwaysTrueConstraint()
            raise KeyError("Any constraint description must contain type information.")

        type2class = {
            'allGoalsFulfilled': AllGoalsFulfilled,
            'allGoalsFulfilledInOrder': GoalOrderConstraint,
            'alwaysTrue': AlwaysTrueConstraint,
            'collisionFree': CollisionFree,
            'goalsBySameRobot': GoalsBySameRobot,
            'joint': JointLimits,
            'selfCollisionFree': SelfCollisionFree,
            'validAssembly': AlwaysTrueConstraint,
            'basePlacement': BasePlacement,
            'endEffector': EndEffector
        }

        # Shortcut to _from_json_data if used on subclasses to return Error or expected type if possible
        if cls is not ConstraintBase:
            if cls is type2class[constraint_type]:
                return cls._from_json_data(d)
            raise ValueError(f"Want to create {type(cls)} with description of type {constraint_type}.")
        try:
            class_ref = type2class[constraint_type]
        except KeyError as e:
            raise NotImplementedError(f"Unknown constraint of type {constraint_type}!") from e

        return class_ref._from_json_data(d)

    @classmethod
    def _from_json_data(cls, description: Dict[str, any]) -> ConstraintBase:
        """
        Default constraints can just be constructed from using json fields as constructor inputs.

        Can be overwritten by children if more complex unpacking needed.
        """
        return cls(**description)

    @abstractmethod
    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Should be implemented to check a constraint on a subset of a solution until time t."""
        pass

    @abstractmethod
    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """Main access point to check constraints"""
        pass

    @abstractmethod
    def to_json_data(self) -> Dict[str, any]:
        """Should be implemented to convert the constraint to a json-serializable dictionary"""

    @property
    @abstractmethod
    def _equality_parameters(self) -> Tuple[any, ...]:
        """Returns the parameters that should be used to check equality of constraints"""

    def visualize(self, viz: pinocchio.visualize.MeshcatVisualizer, scale: float = 1.) -> None:
        """Visualize the constraint in an existing meshcat window.

        If possible a constraint should have a visualization. The default is no visualization and should be overwritten
        if a visualization is possible.

        :param viz: A meshcat visualizer as defined in pinocchio
        :param scale: The visualization method assumes the robot size is ~1-2m maximum reach. Use the scale factor
            to adapt for significantly different sizes of robots and/or obstacles.
        """
        pass

    def __eq__(self, other):
        """Checks if two constraints are equal"""
        if type(self) is not type(other):
            return NotImplemented
        return self._equality_parameters == other._equality_parameters


class AlwaysTrueConstraint(ConstraintBase):
    """
    This constraint is always true and can be used for constraints ensured by the software library before evaluation.
    """

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """As the name says, this constraint is always valid"""
        return True

    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """As the name says, this constraint is always valid"""
        return True

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary"""
        logging.info("Always true constraint is skipped during serialization")
        return {}

    @property
    def _equality_parameters(self) -> Tuple[any, ...]:
        """Two always true constraints are always equal"""
        return tuple()


class GoalsBySameRobot(ConstraintBase):
    """Checks that all goals specified were achieved by the same robot."""

    global_only = True

    def __init__(self, goals: Iterable[str]):
        """
        :param goals: The goals that need to be achieved by the same robot
        """
        self.goals = set(goals)
        raise NotImplementedError("Goals by same robot not yet specified nor implemented")

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Evaluates the solution up to time t"""
        # goal_ids, times = map(tuple, zip(*sorted(solution.t_goals.items(), key=lambda x: x[1])))
        # goals = [solution.task.goals_by_id[gid] for gid in goal_ids]
        # check_for = tuple(goal for i, goal in enumerate(goals) if goal.id in self.goals and times[t] <= t)
        # TODO: Check that all goals in check_for were done by the same robot
        warn("Multi-Robot-Tasks not yet implemented", UserWarning)
        return True

    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """Checks the whole solution for all goals specified by this constraint"""
        # goals = [goal for goal in solution.task.goals if goal.id in self.goals]
        # TODO: Check that all goals with ID in self.goals were done by the same robot
        warn("Multi-Robot-Tasks not yet implemented", UserWarning)
        return True

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary"""
        return {
            'type': 'goalsBySameRobot',
            'goals': list(self.goals)
        }

    @property
    def _equality_parameters(self) -> Tuple[str, ...]:
        """Same goals, same constraint"""
        return tuple(sorted(self.goals, key=lambda g: g.id))


class GoalOrderConstraint(ConstraintBase):
    """Checks that the goals in the task are fulfilled in the right order."""

    global_only = True

    def __init__(self, order: Iterable[str]):
        """
        :param order: An iterable of goal IDs that defines the order in which they should be achieved
        """
        self.order: Tuple[str, ...] = tuple(map(str, order))

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Evaluates the solution up to time t"""
        goal_ids, times = map(tuple, zip(*sorted(solution.t_goals.items(), key=lambda x: x[1])))
        check_for = tuple(gid for i, gid in enumerate(goal_ids) if gid in self.order and times[t] <= t)
        return check_for == self.order[:len(check_for)]  # Only check until time t

    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """Every goal specified by this constraint must be fulfilled in the right order"""
        goal_ids, times = map(tuple, zip(*sorted(solution.t_goals.items(), key=lambda x: x[1])))
        return tuple(gid for gid in goal_ids if gid in self.order) == self.order

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary"""
        return {
            'type': 'allGoalsFulfilledInOrder',
            'order': self.order
        }

    @property
    def _equality_parameters(self) -> Tuple[str, ...]:
        """Same order, same constraint"""
        return tuple(self.order)


class AllGoalsFulfilled(ConstraintBase):
    """Checks whether all goals are fulfilled"""

    global_only = True

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Test if all goals can still be fulfilled up to time t."""
        return True

    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """Check if all goals are fulfilled by this solution."""
        return all(g.achieved(solution) for g in solution.task.goals)

    def to_json_data(self) -> Dict[str, any]:
        """Return json-able dict of this constraint."""
        return {'type': 'allGoalsFulfilled'}

    @property
    def _equality_parameters(self) -> Tuple[any, ...]:
        """All goals, same constraint"""
        return tuple()


class JointLimits(ConstraintBase):
    """Holds constraints on the robots (hardware) limits like position, velocity, torques, etc."""

    robot_limit_types = ('q', 'dq', 'ddq', 'tau')
    _robot_attributs = ('joint_limits', 'joint_velocity_limits', 'joint_acceleration_limits', 'joint_torque_limits')

    def __init__(self, parts: Iterable[str]):
        """Builds the joint limit constraint

        :param parts: Can be any of (q, dq, ddq, tau) and define what is constrained
        """
        if isinstance(parts, str) and ',' in parts:
            parts = parts.split(',')
        parts = list(map(str.lower, parts))
        if not all(kind in self.robot_limit_types for kind in parts):
            raise ValueError(f"Invalid limit kind: {parts}")
        assert len(parts) > 0, "At least one kind of limit needs to be specified or this constraint has no effect"
        self.q = 'q' in parts
        self.dq = 'dq' in parts
        self.ddq = 'ddq' in parts
        self.tau = 'tau' in parts

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Evaluates the solution up to time t"""
        valid = True
        for kind, robot_att in zip(self.robot_limit_types, self._robot_attributs):
            if getattr(self, kind):  # Check what kinds of limits are tested
                limits = getattr(solution.robot, robot_att)  # Get the limits for the robot
                in_solution = getattr(solution, kind)[solution.get_time_id(t)]  # Get the real values up until t
                if len(limits.shape) == 2:  # Lower and upper limits
                    valid = valid and np.all(limits[0, :] <= in_solution) and np.all(limits[1, :] >= in_solution)
                else:
                    valid = valid and np.all(np.abs(in_solution) <= limits)
        return valid

    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """Checks the whole solution for all applied joint constraints"""
        valid = True
        for kind, robot_att in zip(self.robot_limit_types, self._robot_attributs):
            if getattr(self, kind):  # Check what kinds of limits are tested
                limits = getattr(solution.robot, robot_att)  # Get the limits for the robot
                in_solution = getattr(solution, kind)  # Get the real values (in solution also called q, dq, ...)
                if len(limits.shape) == 2:  # Lower and upper limits
                    valid = valid and np.all(limits[0, :] <= in_solution) and np.all(limits[1, :] >= in_solution)
                else:
                    valid = valid and np.all(np.abs(in_solution) <= limits)
        return valid

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary"""
        return {
            'type': 'joint',
            'parts': [kind for kind in self.robot_limit_types if getattr(self, kind)]
        }

    @property
    def _equality_parameters(self) -> Tuple[str, ...]:
        """Same parts, same constraint"""
        return tuple(kind for kind in self.robot_limit_types if getattr(self, kind))


class BasePlacement(ConstraintBase):
    """Constraint on the placement of the base coordinate system."""

    global_only = True  # If this should be used for mobile bases at any point, set global_only to False

    def __init__(self, base_pose: ToleratedPose):
        """
        :param base_pose: The desired placement of the base coordinate system with arbitrary tolerances.
        """
        self.base_pose = base_pose

    @classmethod
    def _from_json_data(cls, description: Dict[str, any]) -> ConstraintBase:
        """Create Base placement constraint from json data."""
        base_pose = ToleratedPose.from_json_data(description['pose'])
        return cls(base_pose)

    @property
    def tolerance(self):
        """Utility wrapper"""
        return self.base_pose.tolerance

    def sample_random_valid_position(self) -> Transformation:
        """
        Helps to find a valid base placement, but only works if the tolerances are not too complex.

        :return: A valid base placement as 4x4 np array
        """
        if isinstance(self.tolerance, Tolerance.Cartesian):
            sample = self.base_pose.nominal @ self.tolerance.valid_random_deviation
        elif isinstance(self.tolerance, Tolerance.Composed):
            logging.warning(
                "Sampling random base placement with composed tolerances is incomplete and ignores orientation")
            cart = [tol for tol in self.tolerance.tolerances if isinstance(tol, Tolerance.Cartesian)]
            if len(cart) != 1:
                raise ValueError(
                    "Not possible to sample a valid base placement when multiple or no cartesian tolerances are given.")
            else:
                sample = self.base_pose.nominal @ cart[0].valid_random_deviation
        elif isinstance(self.tolerance, Tolerance.Rotation):
            sample = self.base_pose.nominal @ self.tolerance.valid_random_deviation
        else:
            raise NotImplementedError(f"Cannot sample valid deviation for Tolerance of type {type(self.tolerance)}.")

        if not self.base_pose.valid(sample):
            raise ValueError("Something went wrong when sampling a valid base position.")

        return sample

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """We assume the base placement does not change, so this method makes no sense."""
        raise NotImplementedError("Assuming fixed base for now - use fulfilled method.")

    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """Checks whether the center of the base is within the defined tolerance."""
        base_placement = solution.robot.placement
        return self.base_pose.valid(base_placement)

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary"""
        return {
            "type": "basePlacement",
            "pose": self.base_pose.to_json_data()
        }

    def visualize(self, viz: pinocchio.visualize.MeshcatVisualizer, scale: float = 1.) -> None:
        """Visualize the constraint"""
        self.base_pose.visualize(viz, scale=scale, name="base_constraint")

    @property
    def _equality_parameters(self) -> Tuple[ToleratedPose]:
        """Equal if the placements are equal"""
        return (self.base_pose,)


class CoterminalJointAngles(ConstraintBase):
    """Constrains a robot can reach a certain pose, being ignorant about full circle rotation deviations

    This constraint checks whether given joint position, given by joint angles, can be reached in general.
    It does not check for the exact joint angles but for their coterminal counterpart in the range [-pi, pi)
    """

    def __init__(self, limits: np.ndarray):
        """
        :param limits: A 2xdof array of joint limits: [[upper_limits], [lower_limimts]]
        """
        if not limits.shape[0] == 2:
            raise ValueError("Joint limits must be given as array in (2, #dof) shape")
        if not (limits[0, :] <= limits[1, :]).all():
            raise ValueError("Not all upper limits are larger than the according lower joint limits.")
        self.limits = limits
        super().__init__()

    @staticmethod
    def map(q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Maps the input q to the range [-pi, pi)."""
        return np.mod(q + np.pi, 2 * np.pi) - np.pi

    def _validate(self, q: np.ndarray) -> bool:
        """
        Returns whether q is within the joint limits

        :param q: Joint angles in the interval [-pi, pi)
        """
        q = self.map(q)
        return all(self.limits[0, :] <= q) and all(self.limits[1, :] >= q)

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Checks whether the given joint angles are within the defined limits at time t."""
        q = solution.q[solution.get_time_id(t)]
        return self._validate(q)

    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """Checks whether the given joint angles are within the defined limits in the solution."""
        return all(self._validate(q) for q in solution.q)

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary"""
        raise NotImplementedError("Coterminal Joint Angles not yet available.")

    @property
    def _equality_parameters(self) -> Tuple[float, ...]:
        """Flatten the limits to a 1d array"""
        return tuple(self.limits.flatten())


class JointAngles(CoterminalJointAngles):
    """This is a stronger constraint than coterminal joint angles, verifying exact angles against joint limits."""

    def __init__(self, limits: np.ndarray):
        """
        limits: A 2xdof array of joint limits: [[upper_limits], [lower_limimts]]
        """
        super().__init__(limits)

    @staticmethod
    def map(q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Overwrites the mapping to the interval [-pi, pi) s.t. the original angles are checked."""
        return q

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary"""
        raise NotImplementedError("Use JointAnglesRobot - independent joint angles constraint not yet defined")


class CollisionFree(ConstraintBase):
    """A constraint that checks whether the robot is in collision with the environment or itself."""

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Checks whether the robot is in collision with the environment or itself at time t."""
        q = solution.q[solution.get_time_id(t)]
        solution.robot.update_configuration(q)
        return not solution.robot.has_collisions(solution.task)

    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """Evaluates the whole solution for collision-freeness."""
        return all(self.is_valid_until(solution, t) for t in solution.time_steps)

    def to_json_data(self) -> Dict:
        """Dumps this constraint to a dictionary"""
        return {'type': 'collisionFree'}

    @property
    def _equality_parameters(self) -> Tuple[any, ...]:
        """Collision-free is equal to itself"""
        return ()


class SelfCollisionFree(CollisionFree):
    """A constraint that checks whether the robot is in collision with itself."""

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Checks whether the robot is in collision with itself at time t."""
        q = solution.q[solution.get_time_id(t)]
        return not solution.robot.has_self_collision(q)

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary"""
        return {'type': 'selfCollisionFree'}


class EndEffector(ConstraintBase):
    """A constraint that checks whether the end-effector (eef) keeps a certain pose and/or velocity."""

    default_velocity_limits = np.asarray((-math.inf, math.inf))

    def __init__(self, *,  # Enforce keyword as any part can be missing
                 pose: Optional[ToleratedPose] = None,
                 velocity_lim: Optional[np.ndarray] = None,
                 rotation_velocity_lim: Optional[np.ndarray] = None):
        r"""
        Initialize eef constraint.

        :param pose: tolerated pose to keep within (either initialized or serialized in dict)
        :param velocity_lim: Minimal / Maximal translation velocity :math:`v` of the end-effector in
          :math:`\frac{m}{s}`; defaults to any allowed
        :param rotation_velocity_lim: Minimal / Maximal rotation velocity :math:`\omega` of the end-effector in
          :math:`\frac{rad}{s}`; defaults to any allowed
        """
        self.pose: Optional[ToleratedPose] = pose
        self.velocity_lim: Optional[np.ndarray] = self.default_velocity_limits.copy() \
            if velocity_lim is None else velocity_lim
        self.rotation_velocity_lim: Optional[np.ndarray] = self.default_velocity_limits.copy() \
            if rotation_velocity_lim is None else rotation_velocity_lim
        if self.velocity_lim[0] > self.velocity_lim[1] or self.rotation_velocity_lim[0] > self.rotation_velocity_lim[1]:
            logging.warning("velocity_lim_min > velocity_lim_max or "
                            "rotation_velocity_lim_min > rotation_velocity_lim_max; "
                            "eef_constraint will always be invalid.")

    @classmethod
    def _from_json_data(cls, description: Dict[str, any]) -> ConstraintBase:
        pose = ToleratedPose.from_json_data(description.pop('pose')) if "pose" in description else None
        velocity_lim = np.asarray((description.pop('v_min', -math.inf), description.pop('v_max', math.inf)))
        rotation_velocity_lim = np.asarray((description.pop('o_min', -math.inf), description.pop('o_max', math.inf)))
        return cls(pose=pose, velocity_lim=velocity_lim, rotation_velocity_lim=rotation_velocity_lim)

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Checks whether the robot obeys eef velocity and pose constraints at time t."""
        q = solution.q[solution.get_time_id(t)]
        dq = solution.dq[solution.get_time_id(t)]

        # Check eef velocities
        solution.robot.update_configuration(q, dq)
        v_robot = solution.robot.tcp_velocity
        v_robot, o_robot = np.linalg.norm(v_robot[:3]), np.linalg.norm(v_robot[3:])
        if v_robot < self.velocity_lim[0] or self.velocity_lim[1] < v_robot or \
                o_robot < self.rotation_velocity_lim[0] or self.rotation_velocity_lim[1] < o_robot:
            return False

        # Check eef pose
        if self.pose is None:
            return True
        return self.pose.valid(solution.robot.fk())

    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """Evaluates the whole solution for obeying the eef constraint."""
        return all(self.is_valid_until(solution, t) for t in solution.time_steps)

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary."""
        data = {'type': 'endEffector'}
        for k, v in (('v_min', self.velocity_lim[0]), ('v_max', self.velocity_lim[1]),
                     ('o_min', self.rotation_velocity_lim[0]), ('o_max', self.rotation_velocity_lim[1])):
            if abs(v) < math.inf:
                data[k] = v
        if self.pose is not None:
            data['pose'] = self.pose.to_json_data()
        return data

    @property
    def _equality_parameters(self) -> Tuple[any, ...]:
        """End-effector constraints are equal if they have the same pose and limits"""
        return (self.pose,) + tuple(self.velocity_lim.flat) + tuple(self.rotation_velocity_lim.flat)
