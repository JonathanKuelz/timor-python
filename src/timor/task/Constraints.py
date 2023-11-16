from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pinocchio

from timor import RobotBase
from timor.task import Solution, Task, Tolerance
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
                return cls._from_json_data(d, *args, **kwargs)
            raise ValueError(f"Want to create {type(cls)} with description of type {constraint_type}.")
        try:
            class_ref = type2class[constraint_type]
        except KeyError as e:
            raise AttributeError(f"Unknown constraint of type {constraint_type}!") from e

        return class_ref._from_json_data(d, *args, **kwargs)

    @classmethod
    def _from_json_data(cls, description: Dict[str, any], *args, **kwargs) -> ConstraintBase:
        """
        Default constraints can just be constructed from using json fields as constructor inputs.

        Can be overwritten by children if more complex unpacking needed.
        """
        return cls(**description)

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Checks that a constraint is valid on a subset of a solution until time t."""
        return all(self.is_valid_at(solution, t_i) for t_i in solution.time_steps[:solution.get_time_id(t) + 1])

    def fulfilled(self, solution: Solution.SolutionBase) -> bool:
        """Returns True iff a constraint is held over the whole solution"""
        return self.is_valid_until(solution, solution.time_steps[-1])

    @abstractmethod
    def is_valid_at(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Checks that a constraint is not hurt at a specific time t"""

    @abstractmethod
    def to_json_data(self) -> Dict[str, any]:
        """Should be implemented to convert the constraint to a json-serializable dictionary"""

    @property
    @abstractmethod
    def _equality_parameters(self) -> Tuple[any, ...]:
        """Returns the parameters that should be used to check equality of constraints"""

    def visualize(self, viz: pinocchio.visualize.MeshcatVisualizer, scale: float = 1., show_name: bool = False) -> None:
        """Visualize the constraint in an existing meshcat window.

        If possible a constraint should have a visualization. The default is no visualization and should be overwritten
        if a visualization is possible.

        :param viz: A meshcat visualizer as defined in pinocchio
        :param scale: The visualization method assumes the robot size is ~1-2m maximum reach. Use the scale factor
            to adapt for significantly different sizes of robots and/or obstacles.
        :param show_name: If True, the name of the constraint is shown in the visualization
        """
        pass

    def __eq__(self, other):
        """Checks if two constraints are equal"""
        if type(self) is not type(other):
            return NotImplemented  # pragma: no cover
        return self._equality_parameters == other._equality_parameters


class RobotConstraint(ABC):
    """
    This is a mixin for constraints that can be verified given a task, a robot, and its joint positions and velocities.

    They are not necessarily to be used only in tasks, but can be used to check whether a robot is in a valid state for
    downstream tasks like the computation of the inverse kinematics.
    """

    def check_single_state(self,
                           task: Optional[Task.Task],
                           robot: RobotBase,
                           q: Sequence[float],
                           dq: Sequence[float],
                           ddq: Optional[Sequence[float]] = None) -> bool:
        """
        Checks whether the constraint is fulfilled for a single state of the robot.

        :param task: The task that is being solved -- if not provided, an empty task will be generated.
        :param robot: The robot that is used to solve the task.
        :param q: The joint positions of the robot.
        :param dq: The joint velocities of the robot.
        :param ddq: The joint accelerations of the robot.
        """

        def reshape(arr: Sequence[float]) -> np.ndarray:
            """Helper to reshape arrays to (1, dof)"""
            ret = np.reshape(arr, (1, -1))
            if not ret.shape[1] == robot.dof:
                raise ValueError(f"Invalid shape of input argument q and/or dq: {np.asarray(arr).shape}")
            return ret

        q, dq = map(reshape, (q, dq))
        ddq = reshape(ddq) if ddq is not None else None
        task = task if task is not None else Task.Task.empty()
        return self._check_single_state(task, robot, q, dq, ddq)

    @abstractmethod
    def _check_single_state(self, task: Optional[Task.Task], robot: RobotBase, q: np.ndarray, dq: np.ndarray,
                            ddq: Optional[np.ndarray] = None) -> bool:
        """
        This method needs to be implemented by all constraints that inherit from this mixin.
        """


class AlwaysTrueConstraint(ConstraintBase, RobotConstraint):
    """
    This constraint is always true and can be used for constraints ensured by the software library before evaluation.
    """

    def _check_single_state(self, task: Optional[Task.Task], robot: RobotBase, q: np.ndarray, dq: np.ndarray,
                            ddq: Optional[np.ndarray]) -> bool:
        """As the name says, this constraint is always valid"""
        return True

    def is_valid_at(self, solution: Solution.SolutionBase, t: float) -> bool:
        """As the name says, this constraint is always valid"""
        return True

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

    def is_valid_at(self, solution: Solution.SolutionBase, t: float) -> bool:
        """This constraint inherently needs access to more than one point in time"""
        raise AttributeError("Goals by same robot constraint cannot be checked at one specific time")

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Evaluates the solution up to time t"""
        # goal_ids, times = map(tuple, zip(*sorted(solution.t_goals.items(), key=lambda x: x[1])))
        # goals = [solution.task.goals_by_id[gid] for gid in goal_ids]
        # check_for = tuple(goal for i, goal in enumerate(goals) if goal.id in self.goals and times[t] <= t)
        # TODO: Check that all goals in check_for were done by the same robot
        raise NotImplementedError()

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

    def is_valid_at(self, solution: Solution.SolutionBase, t: float) -> bool:
        """This constraint inherently needs access to more than one point in time"""
        raise AttributeError("Goal order constraint cannot be checked at one specific time")

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

    def is_valid_at(self, solution: Solution.SolutionBase, t: float) -> bool:
        """This constraint inherently needs access to more than one point in time"""
        if t >= solution.time_steps[-1]:
            return self.fulfilled(solution)
        raise AttributeError("All goals fulfilled constraint cannot be checked at one specific time")

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Test if all goals can still be fulfilled up to time t."""
        if t >= solution.time_steps[-1]:
            return self.fulfilled(solution)
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


class JointLimits(ConstraintBase, RobotConstraint):
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

    def _check_single_state(self, task: Task.Task, robot: RobotBase, q: np.ndarray, dq: np.ndarray,
                            ddq: Optional[np.ndarray] = None) -> bool:
        """
        Checks whether the limits of the robot are held.

        As a single state doesn't have a solution reference, this check assumes no external influences.
        :param task: The task is not being used in this check.
        :param robot: The robot for which the limits are checked.
        :param q: The joint positions of the robot.
        :param dq: The joint velocities of the robot.
        :param ddq: The joint accelerations of the robot.
        """
        get_att = {'q': lambda: q, 'dq': lambda: dq, 'ddq': lambda: ddq if ddq is not None else np.zeros_like(q)}
        get_att['tau'] = lambda: robot.id(q, dq, get_att['ddq']())
        valid = True
        for kind, robot_att in zip(self.robot_limit_types, self._robot_attributs):
            if getattr(self, kind):  # Check what kinds of limits are tested
                limits = getattr(robot, robot_att)  # Get the limits for the robot
                state = get_att[kind]()
                if len(limits.shape) == 2:  # Lower and upper limits
                    valid = valid and np.all(limits[0, :] <= state) and np.all(limits[1, :] >= state)
                else:
                    valid = valid and np.all(np.abs(state) <= limits)
        return valid

    def is_valid_at(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Evaluates the solution at time t"""
        valid = True
        for kind, robot_att in zip(self.robot_limit_types, self._robot_attributs):
            if getattr(self, kind):  # Check what kinds of limits are tested
                limits = getattr(solution.robot, robot_att)  # Get the limits for the robot
                in_solution = getattr(solution, kind)[solution.get_time_id(t)]  # Get the real values at time t
                if len(limits.shape) == 2:  # Lower and upper limits
                    valid = valid and np.all(limits[0, :] <= in_solution) and np.all(limits[1, :] >= in_solution)
                else:
                    valid = valid and np.all(np.abs(in_solution) <= limits)
        return valid

    def is_valid_until(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Evaluates the solution up to time t"""
        valid = True
        for kind, robot_att in zip(self.robot_limit_types, self._robot_attributs):
            if getattr(self, kind):  # Check what kinds of limits are tested
                limits = getattr(solution.robot, robot_att)  # Get the limits for the robot
                in_solution = getattr(solution, kind)[:solution.get_time_id(t) + 1]  # Get the real values up until t
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
    def _from_json_data(cls, description: Dict[str, any], *args, **kwargs) -> ConstraintBase:
        """Create Base placement constraint from json data."""
        frame_name = 'constraint.base'
        if 'frameName' in kwargs:
            frame_name = f"{kwargs.pop('frameName', None)}.{frame_name}"
        base_pose = ToleratedPose.from_json_data(description['pose'], **kwargs, frameName=frame_name)
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

    def is_valid_at(self, solution: Solution.SolutionBase, t: float) -> bool:
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

    def visualize(self, viz: pinocchio.visualize.MeshcatVisualizer, scale: float = 1., show_name: bool = False) -> None:
        """Visualize the constraint"""
        self.base_pose.visualize(viz, scale=scale, name="base_constraint", text="Base constraint" * show_name,
                                 background_color="red", text_color="black")

    @property
    def _equality_parameters(self) -> Tuple[ToleratedPose]:
        """Equal if the placements are equal"""
        return (self.base_pose,)


class CoterminalJointAngles(ConstraintBase, RobotConstraint):
    """Constrains a robot can reach a certain pose, being ignorant about full circle rotation deviations

    This constraint checks whether given joint position, given by joint angles, can be reached in general.
    It does not check for the exact joint angles but for their coterminal counterpart in the range [-pi, pi)
    """

    def __init__(self, limits: np.ndarray):
        """
        :param limits: A 2xdof array of joint limits: [[lower_limits], [upper_limimts]]
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

    def _check_single_state(self, task: Optional[Task.Task], robot: RobotBase, q: np.ndarray,
                            dq: np.ndarray, ddq: Optional[np.ndarray] = None) -> bool:
        """Checks if q is within the joint limits, ignoring all other arguments."""
        return self._validate(q)

    def _validate(self, q: np.ndarray) -> bool:
        """
        Returns whether q is within the joint limits

        :param q: Joint angles in the interval [-pi, pi)
        """
        q = self.map(q)
        return all(self.limits[0, :] <= q) and all(self.limits[1, :] >= q)

    def is_valid_at(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Checks whether the given joint angles are within the defined limits at time t."""
        q = solution.q[solution.get_time_id(t)]
        return self._validate(q)

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
        limits: A 2xdof array of joint limits: [[lower_limits], [upper_limimts]]
        """
        super().__init__(limits)

    @staticmethod
    def map(q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Overwrites the mapping to the interval [-pi, pi) s.t. the original angles are checked."""
        return q

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary"""
        raise NotImplementedError("Use JointAnglesRobot - independent joint angles constraint not yet defined")


class CollisionFree(ConstraintBase, RobotConstraint):
    """A constraint that checks whether the robot is in collision with the environment or itself."""

    def __init__(self, safety_margin: float = 0):
        """
        Initialize collision-free constraint

        :param safety_margin: A safety margin to use for collision checking. The default is 0, which means that the
            collision checking is done with the exact geometries. For any value larger than 0, the collision checking
            is done with the inflated geometries.
        """
        super().__init__()
        if safety_margin < 0:
            raise ValueError("Safety margin must be non-negative.")
        self.safety_margin: float = safety_margin

    def _check_single_state(self, task: Optional[Task.Task], robot: RobotBase, q: np.ndarray, dq: np.ndarray,
                            ddq: Optional[np.ndarray] = None) -> bool:
        """Checks the constraint for the given task in configuration q."""
        robot.update_configuration(q)
        return not robot.has_collisions(task, safety_margin=self.safety_margin)

    def is_valid_at(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Checks whether the robot is in collision with the environment or itself at time t."""
        return self._check_single_state(solution.task, solution.robot, solution.q[solution.get_time_id(t)],
                                        solution.dq[solution.get_time_id(t)])

    def to_json_data(self) -> Dict:
        """Dumps this constraint to a dictionary"""
        return {'type': 'collisionFree'}

    @property
    def _equality_parameters(self) -> Tuple[any, ...]:
        """Collision-free is equal to itself"""
        return ()


class SelfCollisionFree(CollisionFree, RobotConstraint):
    """A constraint that checks whether the robot is in collision with itself."""

    def __init__(self, safety_margin: float = 0):
        """
        Initialize self-collision-free constraint

        :param safety_margin: For self collisions, this parameter is currently ignored.
        """
        super().__init__(safety_margin=safety_margin)
        if safety_margin != 0:
            logging.warning(f"The safety margin of {safety_margin} is currently ignored for self collision checking.")

    def _check_single_state(self, task: Optional[Task.Task], robot: RobotBase, q: np.ndarray, dq: np.ndarray,
                            ddq: Optional[np.ndarray] = None) -> bool:
        """Checks the constraint for the given task in configuration q."""
        return not robot.has_self_collision(q)

    def to_json_data(self) -> Dict[str, any]:
        """Dumps this constraint to a dictionary"""
        return {'type': 'selfCollisionFree'}


class EndEffector(ConstraintBase, RobotConstraint):
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
    def _from_json_data(cls, description: Dict[str, any], *args, **kwargs) -> ConstraintBase:
        frame_name = 'constraint.eef'
        if 'frameName' in kwargs:
            frame_name = f"{kwargs.pop('frameName', None)}.{frame_name}"
        pose = ToleratedPose.from_json_data(description.pop('pose'), frameName=frame_name, **kwargs) \
            if "pose" in description else None
        velocity_lim = np.asarray((description.pop('v_min', -math.inf), description.pop('v_max', math.inf)))
        rotation_velocity_lim = np.asarray((description.pop('o_min', -math.inf), description.pop('o_max', math.inf)))
        return cls(pose=pose, velocity_lim=velocity_lim, rotation_velocity_lim=rotation_velocity_lim)

    def _check_single_state(self, task: Task, robot: RobotBase, q: np.ndarray, dq: np.ndarray,
                            ddq: Optional[np.ndarray] = None) -> bool:
        """Can be used to check this constraint against a single, existing robot state. The task is ignored."""
        robot.update_configuration(q, dq)
        v_robot = robot.tcp_velocity
        v_robot, o_robot = np.linalg.norm(v_robot[:3]), np.linalg.norm(v_robot[3:])
        if v_robot < self.velocity_lim[0] or self.velocity_lim[1] < v_robot or \
                o_robot < self.rotation_velocity_lim[0] or self.rotation_velocity_lim[1] < o_robot:
            return False
        # Check eef pose
        if self.pose is None:
            return True
        return self.pose.valid(robot.fk())

    def is_valid_at(self, solution: Solution.SolutionBase, t: float) -> bool:
        """Checks whether the robot obeys eef velocity and pose constraints at time t."""
        q = solution.q[solution.get_time_id(t)]
        dq = solution.dq[solution.get_time_id(t)]
        return self.check_single_state(task=None, robot=solution.robot, q=q, dq=dq)

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


DEFAULT_CONSTRAINTS = (JointLimits(('q', 'dq', 'tau')),
                       CollisionFree(),
                       AllGoalsFulfilled())
