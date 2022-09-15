from abc import ABC, abstractmethod
import inspect
from typing import Dict, List, Tuple, Union, Optional

import meshcat.geometry
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

from timor import scenario
from timor.scenario import Constraints, Tolerance
from timor.utilities.dtypes import fuzzy_dict_key_matching
import timor.utilities.logging as logging
from timor.utilities.tolerated_pose import ToleratedPose


class GoalBase(ABC):
    """The base implementation for a crok scenario goal."""

    green_material = meshcat.geometry.MeshBasicMaterial(color=0x90ee90)  # for visualization of goals

    def __init__(self,
                 goal_id: Union[int, str],
                 constraints: List['Constraints.ConstraintBase'] = None,
                 ):
        """Initiate goals with the according parameters

        :param goal_id: The id of the goal. This is used to identify the goal in the solution.
        :param constraints: A list of constraints that need to be satisfied for the goal to be achieved - but only here.
          (Which separates them from scenario goals which must hold for a whole scenario)
        """
        self._id = str(goal_id)
        self.constraints = constraints if constraints is not None else list()
        self._assert_constraints_local()

    @staticmethod
    def from_crok_description(description: Dict[str, any]):
        """Maps a crok json description to a goal instance"""
        type2class = {
            'At': At,
            'at': At,
            'Reach': Reach,
            'reach': Reach,
            'ReturnTo': ReturnTo,
            'returnTo': ReturnTo,
            'pause': Pause
        }
        aliases = {'ID': 'goal_id',
                   'tolerance': 'position_tolerance',  # Assuming tolerance defaults to position
                   'time': 'return_to_t'
                   }
        class_ref = type2class[description['type']]

        # Match keys from the input case-insensitive to all possible input arguments of the class to be instantiated
        keywords = fuzzy_dict_key_matching(description, aliases,
                                           desired_only=tuple(inspect.signature(class_ref.__init__).parameters.keys()))

        if 'goal_pose' in keywords:
            keywords['goal_pose'] = ToleratedPose.from_crok(keywords['goal_pose'])

        return class_ref(**keywords)

    @property
    def id(self) -> str:
        """The unique goal identifier"""
        return self._id

    @abstractmethod
    def _achieved(self, solution: 'scenario.Solution.SolutionBase') -> bool:
        """Needs to be implemented by children according to the goal"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, any]):
        """Deserializes a goal from a crok-like dictionary"""

    @abstractmethod
    def to_dict(self) -> Dict[str, any]:
        """
        Returns a crok-compatible dictionary representation of the goal.
        """
        pass

    @abstractmethod
    def visualize(self, viz: MeshcatVisualizer, scale: float = .33) -> None:
        """
        Every goal should have a visualization. The default color is green.

        :param viz: A meshcat visualizer as defined in pinocchio
        :param scale: The visualization method assumes the robot size is ~1-2m maximum reach. Use the scale factor
          to adapt for significantly different sizes of robots and/or obstacles.
        """
        pass

    def _valid(self, solution: 'scenario.Solution.SolutionBase') -> bool:
        """Checks all goal-level constraints"""
        t = solution.t_goals[self.id]
        return all(c.is_valid_at(solution, t) for c in self.constraints)

    def achieved(self, solution: 'scenario.Solution.SolutionBase') -> bool:
        """Calls the custom _achieved method and checks that constraints are held"""
        return self._achieved(solution) and self._valid(solution)

    def _assert_constraints_local(self):
        """Checks that the constraints of this goal have a "local meaning".

        :raises: ValueError if one of the constraints would be better off as a global constraint.
        """
        for constraint in self.constraints:
            if constraint.global_only:
                raise ValueError(f"Constraint {constraint} should be a global constraint.")

    def _constraints_serialized(self) -> List[Dict[str, any]]:
        constraints = []
        joint_constraint = dict(type='Joint', parts=list())  # Multiple joint constraints can be merged in a single one
        for constraint in self.constraints:
            info = constraint.to_dict()
            if info.get('type', None) == 'Joint':
                joint_constraint['parts'].extend(info['parts'])
            else:
                constraints.append(info)
        if len(joint_constraint['parts']) > 0:
            constraints.append(joint_constraint)

        return constraints

    def _get_t_idx_goal(self, solution: 'scenario.Solution.SolutionBase') -> Tuple[float, int]:
        """Finds the time t and the index within the time array idx at which the goal is achieved."""
        try:
            t_goal = solution.t_goals[self.id]
        except KeyError:
            raise KeyError(f"Solution {solution} does not seem to include goal {self.id}.")
        try:
            id_goal = solution.get_time_id(t_goal)
        except ValueError:
            raise ValueError(f"Goal {self.id} seems to be achieved at t={t_goal}, "
                             f"but could not find it in the {solution.__class__.__name__}'s time array.")

        return t_goal, id_goal


class At(GoalBase):
    """A goal that is achieved when the robot is at a certain position for at least one time step t.

    The robot does not need to be still standing, though.
    """

    def __init__(self,
                 goal_id: Union[int, str],
                 goal_pose: ToleratedPose,
                 constraints: List['Constraints.ConstraintBase'] = None,
                 ):
        """Pass a desired pose

        :param goal_id: The id of the goal. This is used to identify the goal in the solution.
        :param goal_pose: The goal pose to be reached.
        :param constraints: A list of constraints that need to be satisfied for the goal to be achieved.
        """
        super().__init__(goal_id, constraints)
        self.goal_pose: ToleratedPose = goal_pose

    def visualize(self, viz: MeshcatVisualizer, scale: float = .33, show_name: bool = False) -> None:
        """Shows an error pointing towards the desired position"""
        self.goal_pose.visualize(viz, scale=scale, name=f"goal_{self.id}", text=f"Goal {self.id}" * show_name)

    @classmethod
    def from_dict(cls, d: Dict[str, any]):
        """Loads the At goal to a dictionary description."""
        return cls(
            goal_id=d['ID'],
            goal_pose=ToleratedPose.from_crok(d['goal_pose']),
            constraints=Constraints.ConstraintBase.from_crok_description(d['constraints'])
        )

    @property
    def nominal_goal(self):
        """The nominal goal pose"""
        return self.goal_pose.nominal

    def to_dict(self) -> Dict[str, any]:
        """Dumps the At goal to a dictionary description."""
        return {
            'ID': self.id,
            'type': 'At',
            'constraints': self._constraints_serialized(),
            'goal_pose': self.goal_pose.crok_description,
        }

    def _achieved(self, solution: 'scenario.Solution.SolutionBase') -> bool:
        """
        Returns true, if the solution satisfies this goal.

        :param solution: Any solution instance that must contain a time at which this goal was solved
        :return: True if the end effector placement at time t is within the defined tolerances for the desired pose
        """
        t_goal, id_goal = self._get_t_idx_goal(solution)
        is_pose = solution.tcp_pose_at(t_goal)
        return self.goal_pose.valid(is_pose)


class Reach(At):
    """A goal that is achieved when the robot is at a certain position for at least one time step at zero velocity"""

    def __init__(self,
                 goal_id: Union[int, str],
                 goal_pose: ToleratedPose,
                 velocity_tolerance: Tolerance.ToleranceBase = Tolerance.Abs6dPoseTolerance.default(),
                 constraints: List['Constraints.ConstraintBase'] = None,
                 ):
        """Reach a desired pose while standing still.

        :param goal_id: The id of the goal. This is used to identify the goal in the solution.
        :param goal_pose: The goal pose to be reached.
        :param velocity_tolerance: Defines the velocity tolerance for v_desired.
        :param constraints: A list of constraints that need to be satisfied for the goal to be achieved.
        """
        super().__init__(goal_id, goal_pose, constraints)
        self.velocity_tolerance = velocity_tolerance

    @classmethod
    def from_dict(cls, d: Dict[str, any]):
        """Loads the Reach goal to a dictionary description."""
        return cls(
            goal_id=d['ID'],
            goal_pose=ToleratedPose.from_crok(d['goal_pose']),
            # TODO: Velocity tolerance is not documented - should a crok description be used?
            velocity_tolerance=Tolerance.Abs6dPoseTolerance.default(),
            constraints=Constraints.ConstraintBase.from_crok_description(d['constraints'])
        )

    def to_dict(self) -> Dict[str, any]:
        """Dumps the Reach goal to a dictionary description."""
        return {
            'ID': self.id,
            'type': 'Reach',
            'constraints': self._constraints_serialized(),
            'goal_pose': self.goal_pose.crok_description,
        }

    def _achieved(self, solution: 'scenario.Solution.SolutionBase') -> bool:
        """
        Returns true, if position and velocity are within the defined tolerances
        """
        at_achieved = super()._achieved(solution)
        t_goal, id_goal = self._get_t_idx_goal(solution)
        is_velocity = solution.tcp_velocity_at(t_goal)
        return at_achieved and self.velocity_tolerance.valid(np.zeros(is_velocity.shape), is_velocity)


class ReturnTo(GoalBase):
    """This goal is achieved when the robot returns to the state it was in at a previous goal"""

    def __init__(self,
                 goal_id: Union[int, str],
                 return_to_t: float,
                 velocity_tolerance: Tolerance.ToleranceBase = Tolerance.Abs6dPoseTolerance.default(),
                 acceleration_tolerance: Tolerance.ToleranceBase = Tolerance.Abs6dPoseTolerance.default(),
                 constraints: List['Constraints.ConstraintBase'] = None,
                 ):
        """Return to a previous goal

        :param goal_id: The id of the goal. This is used to identify the goal in the solution.
        :param return_to_t: This time step defines the position, velocity and acceelertion to which the robot shall
          later return to.
        :param velocity_tolerance: Defines the velocity tolerance relative to the velocity at return_to_t.
        :param acceleration_tolerance: Defines the acceleration tolerance relative to the acceleration at return_to_t.
        """
        # TODO: Currently, crok does not support providing custom tolerances
        super().__init__(goal_id, constraints)
        self.t: float = return_to_t
        self.velocity_tolerance = velocity_tolerance
        self.acceleration_tolerance = acceleration_tolerance
        self.desired_pose: Optional[ToleratedPose] = None

    @classmethod
    def from_dict(cls, d: Dict[str, any]):
        """Loads the Return goal to a dictionary description."""
        return cls(
            goal_id=d['ID'],
            return_to_t=d['time'],
            velocity_tolerance=Tolerance.Abs6dPoseTolerance.default(),
            acceleration_tolerance=Tolerance.Abs6dPoseTolerance.default(),
            constraints=Constraints.ConstraintBase.from_crok_description(d['constraints'])
        )

    def to_dict(self) -> Dict[str, any]:
        """Dumps the Return goal to a dictionary description."""
        return {
            'ID': self.id,
            'type': 'ReturnTo',
            'constraints': self._constraints_serialized(),
            'time': self.t,
        }

    def visualize(self, viz: MeshcatVisualizer, scale: float = .33) -> None:
        """Shows an error pointing towards the desired position"""
        logging.warning("ReturnTo cannot be visualized (yet)")

    def _achieved(self, solution: 'scenario.Solution.SolutionBase') -> bool:
        """
        Returns true, if position, velocity and acceleration are (almost) equal to those at t=return_to_t
        """
        t_goal, id_goal = self._get_t_idx_goal(solution)
        desired_position = solution.tcp_pose_at(self.t)
        self.desired_position = ToleratedPose(nominal=desired_position, tolerance=Tolerance.DEFAULT_SPATIAL)
        desired_velocity = solution.tcp_velocity_at(self.t)
        desired_acceleration = solution.tcp_velocity_at(self.t)
        is_position = solution.tcp_pose_at(t_goal)
        is_velocity = solution.tcp_velocity_at(t_goal)
        is_acceleration = solution.tcp_velocity_at(t_goal)

        return (self.desired_position.valid(is_position)
                and self.velocity_tolerance.valid(desired_velocity, is_velocity)
                and self.acceleration_tolerance.valid(desired_acceleration, is_acceleration))


class Pause(GoalBase):
    """
    This goal is achieved if the robot does not move for a preconfigured duration.
    """

    epsilon: float = 1e-9  # Precision error: Pause is allowed to be epsilon shorter than self.duration

    def __init__(self,
                 goal_id: Union[int, str],
                 duration: float,
                 constraints: List['scenario.Constraints.ConstraintBase'] = None,
                 ):
        """Pause for a given amount of time.

        :param goal_id: The id of the goal. This is used to identify the goal in the solution.
        :param duration: The duration of the pause in seconds.
        """
        super().__init__(goal_id=goal_id, constraints=constraints)
        self.duration: float = duration

    def visualize(self, viz: MeshcatVisualizer, scale: float = .33) -> None:
        """Pause goals cannot be visualized, but the method is kept for consistency"""
        logging.info("Not visualizing the pause goal - there is no meaningful visualization.")

    @classmethod
    def from_dict(cls, d: Dict[str, any]):
        """Loads the Pause goal from a dictionary description."""
        return cls(
            goal_id=d['ID'],
            duration=d['duration'],
            constraints=Constraints.ConstraintBase.from_crok_description(d['constraints'])
        )

    def to_dict(self) -> Dict[str, any]:
        """Dumps the Pause goal to a dictionary description."""
        return {
            'ID': self.id,
            'type': 'pause',
            'duration': self.duration,
            'constraints': self._constraints_serialized()
        }

    def _achieved(self, solution: 'scenario.Solution.SolutionBase') -> bool:
        """
        Returns true, if the robot does not move for a preconfigured duration, starting at t_goal.
        """
        t_goal, id_goal = self._get_t_idx_goal(solution)
        t_goal_starts = t_goal - self.duration + self.epsilon
        # Start pause goal at earliest possible time that is in the solution trajectory
        try:
            t_goal_starts = solution.time_steps[solution.time_steps <= t_goal_starts][-1]
        except IndexError:
            return False  # The pause goal would start before the trajectory even begins
        t_idx_start = solution.get_time_id(t_goal_starts)
        while solution.time_steps[t_idx_start] > t_goal_starts + 1e-6:
            # The trajectory is not defined at the exact time when this goal time span ends, so we need to check the
            #  next time step.
            if t_idx_start == 0:
                logging.warning("Pause goal is not achievable, because the trajectory is not defined at t={}.".format(
                    t_goal_starts))
                return False
            t_idx_start -= 1
        q_start = solution.q[t_idx_start]
        return all((q_start == solution.q[t]).all() and
                   not solution.dq[t].any() and
                   not solution.ddq[t].any()
                   for t in range(t_idx_start+1, id_goal+1))
