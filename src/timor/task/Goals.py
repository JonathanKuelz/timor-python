from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from typing import Dict, List, Optional, Tuple

import meshcat.geometry
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

from timor import task
from timor.task import Constraints, Solution, Tolerance
from timor.utilities.dtypes import fuzzy_dict_key_matching
from timor.utilities.errors import TimeNotFoundError
from timor.utilities.jsonable import JSONable_mixin
import timor.utilities.logging as logging
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.trajectory import Trajectory


class GoalBase(ABC, JSONable_mixin):
    """The base implementation for a task goal."""

    green_material = meshcat.geometry.MeshBasicMaterial(color=0x90ee90)  # for visualization of goals

    def __init__(self,
                 ID: str,
                 constraints: List[Constraints.ConstraintBase] = None,
                 ):
        """
        Initiate goals with the according parameters

        :param ID: The id of the goal. This is used to identify the goal in the solution.
        :param constraints: A list of constraints that need to be satisfied for the goal to be achieved - but only here.
            (Which separates them from task goals which must hold for a whole task)
        """
        self._id = str(ID)
        self.constraints = constraints if constraints is not None else list()
        self._assert_constraints_local()

    @staticmethod
    def goal_from_json_data(description: Dict[str, any]):
        """Maps a json description to a goal instance"""
        type2class = {
            'At': At,
            'at': At,
            'Reach': Reach,
            'reach': Reach,
            'ReturnTo': ReturnTo,
            'returnTo': ReturnTo,
            'pause': Pause,
            'follow': Follow
        }
        class_ref = type2class[description['type']]

        # Match keys from the input case-insensitive to all possible input arguments of the class to be instantiated
        keywords = fuzzy_dict_key_matching(description, {},
                                           desired_only=tuple(inspect.signature(class_ref.__init__).parameters.keys()))

        return class_ref.from_json_data(keywords)

    @property
    def id(self) -> str:
        """The unique goal identifier"""
        return self._id

    @abstractmethod
    def _achieved(self, solution: task.Solution.SolutionBase, t_goal: float, idx_goal: int) -> bool:
        """Needs to be implemented by children according to the goal"""
        pass

    @classmethod
    @abstractmethod
    def from_json_data(cls, d: Dict[str, any], *args, **kwargs):
        """Deserializes a goal from a dictionary"""

    @abstractmethod
    def to_json_data(self) -> Dict[str, any]:
        """
        Returns a json-compatible dictionary representation of the goal.
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

    def _valid(self, solution: task.Solution.SolutionBase, t_goal: float, idx_goal: int) -> bool:
        """Checks all goal-level constraints"""
        return all(c.is_valid_until(solution, t_goal) for c in self.constraints)

    def achieved(self, solution: Solution.SolutionBase) -> bool:
        """Calls the custom _achieved method and checks that constraints are held"""
        try:
            t_goal, idx_goal = self._get_t_idx_goal(solution)
        except KeyError:
            logging.debug(f"Goal {self.id} not in solution trajectory and therefore not achieved.")
            return False
        return self._achieved(solution, t_goal, idx_goal) and self._valid(solution, t_goal, idx_goal)

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
            info = constraint.to_json_data()
            if info.get('type', None) == 'Joint':
                joint_constraint['parts'].extend(info['parts'])
            else:
                constraints.append(info)
        if len(joint_constraint['parts']) > 0:
            constraints.append(joint_constraint)

        return constraints

    def _get_t_idx_goal(self, solution: Solution.SolutionBase) -> Tuple[float, int]:
        """Finds the time t and the index within the time array idx at which the goal is achieved."""
        try:
            t_goal = solution.t_goals[self.id]
        except KeyError:
            raise KeyError(f"Solution {solution} does not seem to include goal {self.id}.")
        try:
            id_goal = solution.get_time_id(t_goal)
        except TimeNotFoundError:
            raise TimeNotFoundError(f"Goal {self.id} seems to be achieved at t={t_goal}, "
                                    f"but could not find it in the {solution.__class__.__name__}'s time array.")

        return t_goal, id_goal

    def __eq__(self, other):
        """Enforce the implementation"""
        raise NotImplementedError(f"Equality of {self.__class__.__name__} objects is not implemented.")

    def __hash__(self):
        """Hashes the goal by its id"""
        return hash(self.id)


class GoalWithDuration(GoalBase, ABC):
    """
    Any goal that needs to test more than a single time-step to be achieved.

    This also includes checking the goal constraints at any point of the goal's duration.
    """

    epsilon: float = 1e-9  # Precision error: Goal is allowed to be epsilon shorter than self.duration

    def __init__(self,
                 ID: str,
                 duration: float,
                 constraints: List[Constraints.ConstraintBase] = None
                 ):
        """Construct a goal with duration."""
        super().__init__(ID, constraints)
        self._duration = duration  # Private as not every subtype has explicit duration

    def _get_time_range_goal(self, solution: Solution.SolutionBase) -> Tuple[Tuple[float, ...], range]:
        """
        Get all time-steps in solution that belong to this goal's duration as list of times and indices.

        :param solution: Solution to evaluate this goal with duration on.
        :returns:
          - A tuple of times that are included in the duration of the goal
          - A range of indices into the solution's time array that fall into the duration of this goal
        """
        t_goal, id_goal = self._get_t_idx_goal(solution)
        t_goal_starts = t_goal - self._duration + self.epsilon  # Expected start time
        try:  # Find index of time step before expected start time
            t_goal_starts = solution.time_steps[solution.time_steps <= t_goal_starts][-1]
        except IndexError:
            raise ValueError("Duration goal expects that whole duration covered by trajectory")
        id_start = solution.get_time_id(t_goal_starts)
        ts = solution.time_steps[id_start:id_goal + 1]
        assert np.all(np.diff(ts) <= solution.task.header.timeStepSize + self.epsilon), \
            "Solution should be sampled every timeStepSize"
        return ts, range(id_start, id_goal + 1)  # id_goal should be included in goal range

    def _valid(self, solution: task.Solution.SolutionBase, t_goal: float, idx_goal: int) -> bool:
        """A goal with duration needs to ensure that its constraints hold at all time-steps within this duration."""
        return all(c.is_valid_until(solution, t) for c in self.constraints
                   for t in self._get_time_range_goal(solution)[0])


class At(GoalBase):
    """A goal that is achieved when the robot is at a certain position for at least one time step t.

    The robot does not need to be still standing, though.
    """

    def __init__(self,
                 ID: str,
                 goalPose: ToleratedPose,
                 constraints: List[Constraints.ConstraintBase] = None,
                 ):
        """
        Pass a desired pose

        :param ID: The id of the goal. This is used to identify the goal in the solution.
        :param goalPose: The goal pose to be reached.
        :param constraints: A list of constraints that need to be satisfied for the goal to be achieved.
        """
        super().__init__(ID, constraints)
        if not isinstance(goalPose, ToleratedPose):
            raise ValueError(f"Goal {ID} needs a ToleratedPose, got a variably of type {type(goalPose)}.")
        self.goal_pose: ToleratedPose = goalPose

    def visualize(self, viz: MeshcatVisualizer, scale: float = .33, show_name: bool = False) -> None:
        """Shows an error pointing towards the desired position"""
        self.goal_pose.visualize(viz, scale=scale, name=f"goal_{self.id}", text=f"Goal {self.id}" * show_name)

    @classmethod
    def from_json_data(cls, d: Dict[str, any], *args, **kwargs):
        """Loads the At goal to a dictionary description."""
        return cls(
            ID=d['ID'],
            goalPose=ToleratedPose.from_json_data(d['goalPose']),
            constraints=[Constraints.ConstraintBase.from_json_data(c) for c in d.get('constraints', [])]
        )

    @property
    def nominal_goal(self):
        """The nominal goal pose"""
        return self.goal_pose.nominal

    def to_json_data(self) -> Dict[str, any]:
        """Dumps the At goal to a dictionary description."""
        return {
            'ID': self.id,
            'type': 'at',
            'constraints': self._constraints_serialized(),
            'goalPose': self.goal_pose.to_json_data(),
        }

    def _achieved(self, solution: task.Solution.SolutionBase, t_goal: float, idx_goal: int) -> bool:
        """
        Returns true, if the solution satisfies this goal.

        :param solution: Any solution instance that must contain a time at which this goal was solved
        :return: True if the end effector placement at time t is within the defined tolerances for the desired pose
        """
        is_pose = solution.tcp_pose_at(t_goal)
        return self.goal_pose.valid(is_pose)

    def __eq__(self, other):
        """At goals are same if they have the same constraints and desired pose"""
        if not isinstance(other, At):
            return False
        return self.constraints == other.constraints and self.goal_pose == other.goal_pose


class Reach(At):
    """A goal that is achieved when the robot is at a certain position for at least one time step at zero velocity"""

    def __init__(self,
                 ID: str,
                 goalPose: ToleratedPose,
                 velocity_tolerance: Tolerance.ToleranceBase = Tolerance.Abs6dPoseTolerance.default(),
                 constraints: List[Constraints.ConstraintBase] = None,
                 ):
        """
        Reach a desired pose while standing still.

        :param ID: The id of the goal. This is used to identify the goal in the solution.
        :param goalPose: The goal pose to be reached.
        :param velocity_tolerance: Defines the velocity tolerance for v_desired.
        :param constraints: A list of constraints that need to be satisfied for the goal to be achieved.
        """
        super().__init__(ID, goalPose, constraints)
        self.velocity_tolerance = velocity_tolerance

    @classmethod
    def from_json_data(cls, d: Dict[str, any], *args, **kwargs):
        """Loads the Reach goal to a dictionary description."""
        return cls(
            ID=d['ID'],
            goalPose=ToleratedPose.from_json_data(d['goalPose']),
            velocity_tolerance=Tolerance.Abs6dPoseTolerance.default(),
            constraints=[Constraints.ConstraintBase.from_json_data(c) for c in d.get('constraints', [])]
        )

    def to_json_data(self) -> Dict[str, any]:
        """Dumps the Reach goal to a dictionary description."""
        return {
            'ID': self.id,
            'type': 'reach',
            'constraints': self._constraints_serialized(),
            'goalPose': self.goal_pose.to_json_data(),
        }

    def _achieved(self, solution: task.Solution.SolutionBase, t_goal: float, idx_goal: int) -> bool:
        """
        Returns true, if position and velocity are within the defined tolerances
        """
        at_achieved = super()._achieved(solution, t_goal, idx_goal)
        if not at_achieved:
            return False
        is_velocity = solution.tcp_velocity_at(t_goal)
        return at_achieved and self.velocity_tolerance.valid(np.zeros(is_velocity.shape), is_velocity)

    def __eq__(self, other):
        """Reach goals are same if they have the same constraints, velocity, and desired pose"""
        if not isinstance(other, Reach):
            return NotImplemented
        return super().__eq__(other) and self.velocity_tolerance == other.velocity_tolerance


class ReturnTo(GoalBase):
    """This goal is achieved when the robot returns to the state it was in at a previous goal"""

    def __init__(self,
                 ID: str,
                 returnToGoal: Optional[str] = None,
                 epsilon: float = 1e-4,
                 constraints: List[Constraints.ConstraintBase] = None
                 ):
        """
        Return to a previous goal

        :param ID: The id of the goal. This is used to identify the goal in the solution.
        :param returnToGoal: ID of other goal which should be matched again; can be left out to denote the start of the
            solution trajectory.
        :param epsilon: Allowed L2 distance between joint states at this goal and the the other goal.
        """
        super().__init__(ID, constraints)
        self.return_to_goal = returnToGoal
        self.distance_tolerance = Tolerance.VectorDistance(epsilon, order=2)
        self.desired_pose: Optional[ToleratedPose] = None

    @classmethod
    def from_json_data(cls, d: Dict[str, any], *args, **kwargs):
        """Loads the Return goal to a dictionary description."""
        return cls(
            ID=d['ID'],
            returnToGoal=d.get('returnToGoal', None),
            constraints=[Constraints.ConstraintBase.from_json_data(c) for c in d.get('constraints', [])]
        )

    def to_json_data(self) -> Dict[str, any]:
        """Dumps the Return goal to a dictionary description."""
        return_to_goal = {'returnToGoal': self.return_to_goal} if self.return_to_goal is not None else {}
        return {
            'ID': self.id,
            'type': 'returnTo',
            'constraints': self._constraints_serialized(),
            **return_to_goal
        }

    def visualize(self, viz: MeshcatVisualizer, scale: float = .33) -> None:
        """Shows an error pointing towards the desired position"""
        logging.info("ReturnTo Goals cannot be visualized (yet)")

    def _achieved(self, solution: task.Solution.SolutionBase, t_goal: float, idx_goal: int) -> bool:
        """
        Returns true, if position, velocity and acceleration are (almost) equal to those at t=return_to_t
        """
        goal_idx = solution.get_time_id(t_goal)
        other_idx = solution.get_time_id(solution.t_goals[self.return_to_goal]) if self.return_to_goal is not None \
            else solution.get_time_id(0)
        desired_position = solution.q[other_idx]
        desired_velocity = solution.dq[other_idx]
        desired_acceleration = solution.ddq[other_idx]
        is_position = solution.q[goal_idx]
        is_velocity = solution.dq[goal_idx]
        is_acceleration = solution.ddq[goal_idx]

        return (self.distance_tolerance.valid(desired_position, is_position)
                and self.distance_tolerance.valid(desired_velocity, is_velocity)
                and self.distance_tolerance.valid(desired_acceleration, is_acceleration))

    def __eq__(self, other):
        """Return goals are same if they have the same constraints, tolerance, and return to option"""
        if not isinstance(other, ReturnTo):
            return NotImplemented
        return self.constraints == other.constraints \
            and self.distance_tolerance == other.distance_tolerance \
            and self.return_to_goal == other.return_to_goal


class Pause(GoalWithDuration):
    """
    This goal is achieved if the robot does not move for a preconfigured duration.
    """

    epsilon: float = 1e-9  # Precision error: Pause is allowed to be epsilon shorter than self._duration

    def visualize(self, viz: MeshcatVisualizer, scale: float = .33) -> None:
        """Pause goals cannot be visualized, but the method is kept for consistency"""
        logging.debug("Not visualizing a pause goal - there is no meaningful visualization.")

    @property
    def duration(self) -> float:
        """Getter for explicit duration of Pause Goal."""
        return self._duration

    @classmethod
    def from_json_data(cls, d: Dict[str, any], *args, **kwargs):
        """Loads the Pause goal from a dictionary description."""
        return cls(
            ID=d['ID'],
            duration=d['duration'],
            constraints=[Constraints.ConstraintBase.from_json_data(c) for c in d.get('constraints', [])]
        )

    def to_json_data(self) -> Dict[str, any]:
        """Dumps the Pause goal to a dictionary description."""
        return {
            'ID': self.id,
            'type': 'pause',
            'duration': self.duration,
            'constraints': self._constraints_serialized()
        }

    def _achieved(self, solution: task.Solution.SolutionBase, t_goal: float, idx_goal: int) -> bool:
        """
        Returns true, if the robot does not move for a preconfigured duration, starting at t_goal.
        """
        try:
            sol_times, sol_idx = self._get_time_range_goal(solution)
        except ValueError as e:
            logging.debug(f"Duration not covered by trajectory; {e}")
            return False
        return all((solution.q[sol_idx[0]] == solution.q[idx]).all()
                   and not solution.dq[idx].any()
                   and not solution.ddq[idx].any()
                   for idx in sol_idx)

    def __eq__(self, other):
        """Two pause goals are equal if there's (almost) the same pause and constraints."""
        if not isinstance(other, Pause):
            return NotImplemented
        return self.constraints == other.constraints \
            and abs(self.duration - other.duration) < self.epsilon


class Follow(GoalWithDuration):
    """
    This goal is achieved if the robot follows a prescribed end-effector trajectory.
    """

    def __init__(self,
                 ID: str,
                 trajectory: Trajectory,
                 constraints: List[Constraints.ConstraintBase] = None,):
        """
        Follow a given trajectory.
        """
        if not trajectory.has_poses:
            raise ValueError("Follow goal needs trajectory of workspace poses.")
        if trajectory.is_timed:
            duration = trajectory[-1].t - trajectory[0].t
        else:
            duration = float("inf")
        super().__init__(ID=ID, duration=duration, constraints=constraints)
        self._trajectory: Trajectory = trajectory

    @property
    def trajectory(self) -> Trajectory:
        """The trajectory to follow along"""
        return self._trajectory

    @classmethod
    def from_json_data(cls, d: Dict[str, any], *args, **kwargs):
        """Loads the Follow goal from a dictionary description."""
        return cls(
            ID=d['ID'],
            trajectory=Trajectory.from_json_data(d['trajectory']),
            constraints=[Constraints.ConstraintBase.from_json_data(c) for c in d.get('constraints', [])]
        )

    def to_json_data(self) -> Dict[str, any]:
        """Dumps the Follow goal to a dictionary description."""
        return {
            'ID': self.id,
            'type': 'follow',
            'trajectory': self.trajectory.to_json_data(),
            'constraints': self._constraints_serialized()
        }

    def visualize(self, viz: MeshcatVisualizer, *,
                  scale: float = .33,
                  num_markers: Optional[int] = 2) -> None:
        """
        Shows the follow goal with num_markers triads to show desired orientation.

        :param num_markers: How many triads to display along the desired trajectory
          (2 = show at start and end of trajectory)
        """
        self._trajectory.visualize(viz, name_prefix=str(self.id), scale=scale, num_markers=num_markers)

    def _achieved(self, solution: Solution.TrajectorySolution, t_goal: float, idx_goal: int) -> bool:
        """
        Implements check if goal achieved at claimed time

        :note: Side-effect if desired trajectory is not timed: Will set `self.duration` such that the trajectory is
          followed in the time-interval [t_goal-duration, t_goal]. This allows constraint checking in _valid to know
          all time-steps within this goal.
        """
        if self.trajectory.is_timed:
            # With timed trajectory we can just go over all time steps and make sure the desired pose is reached
            t_start_follow = t_goal - self.trajectory[-1].t
            for step in self.trajectory:
                try:
                    if not step.pose[0].valid(solution.tcp_pose_at(t_start_follow + step.t[0])):
                        logging.info(
                            f"Follow not fulfilled at time {step.t[0]} "
                            f"after start of trajectory at time {t_start_follow}")
                        return False
                except TimeNotFoundError:  # e.g. happens if solution is shorter than the trajectories duration
                    return False
            return True

        else:
            # Without fixed trajectory times we need to go backwards from t_goal till either the solution starts or all
            # trajectory poses have been fulfilled.
            idx_goal_start = idx_goal  # at which solution index does this trajectory start
            for p in self.trajectory[::-1].pose:
                while not p.valid(solution.tcp_pose_at(solution.time_steps[idx_goal_start])):
                    idx_goal_start -= 1
                    if idx_goal_start < 0:
                        logging.info(f"Followed trajectory till start; follow was not fulfilled for pose {p}.")
                        self._duration = float('inf')
                        return False
            self._duration = solution.time_steps[idx_goal] - solution.time_steps[idx_goal_start]
            return True

    def __eq__(self, other):
        """Two follow goals are equal if they have the same duration, trajectory and constraints."""
        if not isinstance(other, Follow):
            return NotImplemented
        return self.constraints == other.constraints \
            and self._duration == other._duration \
            and self.trajectory == other.trajectory
