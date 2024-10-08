from __future__ import annotations

import abc
from abc import ABC, abstractmethod
import contextlib
from dataclasses import dataclass, field
from enum import Enum
import itertools
from typing import Collection, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Type, Union

import networkx as nx
import numpy as np

from timor import AtomicModule, ModuleAssembly, ModulesDB
from timor.Robot import RobotBase
from timor.task import Constraints, Goals, Task, Tolerance
from timor.utilities import dtypes, logging
from timor.utilities.dtypes import LimitedSizeMap
from timor.utilities.trajectory import Trajectory

GOALS_WITH_POSE = Union[Goals.At, Goals.Follow, Goals.Reach]
GOALS_WITH_ONE_POSE = Union[Goals.At, Goals.Reach]


class FilterNotApplicableException(Exception):
    """
    Raise when a filter is called on a solution that does not contain any goal/constraint that could fail the filter.
    """


class InvalidFilterCombinationError(Exception):
    """
    Can be raised when a sequence of filters does not work together.

    Mostly, because requirements of succeeding filters are not given by preceding ones.
    """


class OverwritesIntermediateResult(ValueError):
    """
    Raise this exception whenever someone tries to override an intermediate result, which must be handled with care!

    This exception can be caught to deal with this problem, but it should never pass silently.
    """


class EternalResult(dtypes.EternalDict):
    """An immutable dictionary with a custom error"""

    # Do not change this manually - it partially breaks the safety mechanism of this class!
    __allow_overwriting: bool = False

    def _immutable(self, *args, **kws):
        """This method is called whenever an operation would change the dictionary."""
        raise OverwritesIntermediateResult()

    def __setitem__(self, key, value):
        """The only way possible to change this dictionary: Add new values"""
        if key in self and not self.__allow_overwriting:
            self._immutable()
        dict.__setitem__(self, key, value)


class ResultType(Enum):
    """
    Definition of possible fields in the IntermediatResults dataclass.

    This enum goes hand-in-hand with the IntermediateResults dataclass. It defines the communication interfaces
    between various RobotFilters: Each one has some dependencies and fills one or more fields of an intermediate result.
    The ResultType is the proper way to define, 'which' fields a filter works with.
    Also, it provides some handy aliases.
    """

    robot = 0
    goal_ik_success = 1
    q_goal_pose = 2
    tau_static = 3
    collision_free_goal = 4
    trajectory = 5


@dataclass
class IntermediateFilterResults:
    """
    Keeps track of intermediate results for a assembly.

    E.g. if the inverse kinematics for certain goals were already calculated, but a complete trajectory was not.
    """

    # Maps a goal ID to final joint positions q (e.g. after solving the inverse kinematics)
    q_goal_pose: EternalResult[str, np.ndarray] = field(default_factory=lambda: EternalResult())
    # Maps a goal ID to static torques calculated at the goalPose
    tau_static: EternalResult[str, float] = field(default_factory=lambda: EternalResult())
    # Maps a tuple of (first_goal, second_goal)-ID to a trajectory combining them
    #   or maps a Follow-goal id on a trajectory
    trajectory: EternalResult[Union[Tuple[str, str], str], Trajectory] = \
        field(default_factory=lambda: EternalResult())
    # RobotBase object created
    robot: RobotBase = field(default=None, repr=False)
    # Maps a goal ID to success in solving inverse kinematics for that goal
    goal_ik_success: EternalResult[str, bool] = field(default_factory=lambda: EternalResult())
    # Maps a goal ID to avoiding obstacle collisions for a previously computed q_goal_pose
    collision_free_goal: EternalResult[str, bool] = field(default_factory=lambda: EternalResult())

    _filters_validated: List[AssemblyFilter] = field(default_factory=lambda: list(), repr=False)
    __allow_overwriting: bool = field(default=None, repr=False)

    def allow_changes(self):
        """
        To be used in a context manager. Temporarily sets __allow_overwriting to True.

        Example use:

        .. code-block:: python

            result = my_method_that_returns_result()
            with result.allow_changes():
                result.some_field = another_value
        """
        return _allow_alter_results(self)

    @property
    def filters_validated(self) -> List[AssemblyFilter]:
        """Make this a private attribute, should never be set from the outside, but only be appended to."""
        return self._filters_validated

    def __setattr__(self, key, value):
        """
        Custom __setattr__ behavior to prevent already checked intermediate results:

        Whenever an attribute is set, and it was not None before, at least one filter stated this attribute would
        pose a valid (partial) solution to a goal. Re-setting this attribute cannot guarantee previous checks are still
        valid. This case must be handled with care, which is why a custom exception is raised that can later be caught.

        :raises: OverwritesIntermediateResult
        """
        if key not in self.__dict__:
            pass  # That's fine, it was not set before
        elif key == '_IntermediateFilterResults__allow_overwriting':
            # If the intermediate result allows overwriting, its properties should do so as well
            eternal_properties = tuple(key for key, value in self.__dict__.items() if isinstance(value, EternalResult))
            for ep in eternal_properties:
                # This should be the only place to override this attribute. Access is complex on purpose - you can only
                #  change this if you really know what you're doing.
                self.__dict__[ep]._EternalResult__allow_overwriting = value
        elif self.__getattribute__(key) is not None:
            if self.__allow_overwriting:
                logging.info(f'Overwriting {key} in intermediate results with {value}.')
            else:
                raise OverwritesIntermediateResult(f'{key} has already been set.')
        super().__setattr__(key, value)


@contextlib.contextmanager
def _allow_alter_results(intermediate_result: IntermediateFilterResults):
    """
    Utility function to allow altering fields of an IntermediateResult using a context manager.

    :param intermediate_result: An IntermediateFilterResults object that should be editable within a context
    """
    try:
        intermediate_result._IntermediateFilterResults__allow_overwriting = True
        yield intermediate_result
    finally:
        intermediate_result._IntermediateFilterResults__allow_overwriting = False


class AssemblyFilter(ABC):
    """
    An interface to define assembly filters.

    An idea derived from the paper "Effortless creation of safe robots from Modules through self-programming and
    self-verification" by Althoff, Giusti, Liu, Pereira [2019]

    Assemblies (=partial solutions) to tasks can be piped through a filter that performs specific checks to quickly
    evaluate obviously infeasible assemblies before applying an expensive full-check and solution generation.

    They are not intended to solve a problem completely but should be combined with a more powerful but most likely
    slower optimizer for all module assemblies not discarded by the hierarchical filter.

    Filters are intended to be called by planners - and are designed to be applied hierarchically, passing intermediate
    results from one to another.
    """

    # TODO: Check in Planner that a chain of filters provides coming requirements
    requires: Tuple[ResultType] = ()  # Which previous intermediate results must've been calculated to use this filter?
    provides: Tuple[ResultType] = ()  # Which intermediate results will be provided by this filter
    # A Collection of goal types the filter can work on: Isinstance is not enough - types must be the same!
    goal_types_filtered: Tuple[Type[Goals.GoalBase]] = tuple()
    skip_not_applicable: bool  # Should a non-applicable solution be passed quietly or not?

    def __init__(self, skip_not_applicable: bool = True):
        """
        Instantiate statistic variables as private variables (should not be reset from outside)

        :param skip_not_applicable: If false, the filter raises an Exception when called on a solution it cannot filter.
        """
        self.skip_not_applicable = skip_not_applicable
        self._passed: int = 0
        self._failed: int = 0

    def check(self, assembly: ModuleAssembly, task: Task.Task, results: IntermediateFilterResults,
              stop_early: bool = True) -> bool:
        """
        The default use of a AssemblyFilter.

        Instantiate your filter, then you can do:

        .. code-block:: python

            if filter_instance.check(assembly):
                go_on
            else:
                discard_solution.

        All intermediate results obtained when checking will be stored

        :param assembly: A assembly to be checked
        :param task: The task that should be solved by the assembly
        :param results: An intermediate results object with previous calculations. New calcualtions by this filter will
          be added to this object.
        :param stop_early: Whether to stop check call on first error, e.g. failing goal, or try to apply it to all goals
          individually such that result returns best effort eventhough filter failed.
        :returns: True if the assembly is possible valid. False iff the assembly is certainly not valid.
        :raises: FilterNotApplicableException
        """
        if any(self._applies(goal) for goal in task.goals):
            passes = self._check(assembly, task, results, stop_early)
        elif len(task.goals) == 0:
            passes = True
        else:  # The filter is not applicable to the solution
            if self.skip_not_applicable:
                passes = True
            else:
                raise FilterNotApplicableException(f"{self.__class__.__name__} not applicable on "
                                                   f"task {task.header.ID}.")
        if passes:
            self._passed += 1
        else:
            self._failed += 1
        return passes

    def goals_applicable(self, task: Task) -> Generator[Goals.GoalBase, None, None]:
        """
        Filters task for goals this filter does work on

        :param task: Any task with goals
        :return: A generator over goals in task that this filter can work with
        """
        return (goal for goal in task.goals if self._applies(goal))

    def reset(self):
        """Resets the failed/passed counts"""
        self._failed = 0
        self._passed = 0

    @property
    def failed(self) -> int:
        """A counter on how many assemblies were dropped by this filter."""
        return self._failed

    @property
    def passed(self) -> int:
        """A counter on how many assemblies passed (including were skipped by) this filter."""
        return self._passed

    @property
    def report(self) -> str:
        """Reports filter statistics."""
        total = self.failed + self.passed
        percentage = 100 * self.failed / total if total != 0 else 0
        return f"{self.__class__.__name__} discarded {percentage:.2f}% of {total} checked solutions."

    def _applies(self, goal: Goals.GoalBase) -> bool:
        """Can this goal be filtered? True -> Filter applicable. False -> Goal will always pass this filter"""
        return self._applies_to_type(type(goal))

    def _applies_to_type(self, goal_type: Type[Goals.GoalBase]) -> bool:
        """Utility wrapper: Can this type be filtered by this filter?"""
        try:
            goal_type in self.goal_types_filtered
        except TypeError:
            # This is the case when goal_types_filtered is of UnionType (which cannot be checked prior to python 3.10)
            return goal_type in self.goal_types_filtered.__args__

    @abstractmethod
    def _check(self, assembly: ModuleAssembly, task: Task.Task, results: IntermediateFilterResults,
               stop_early: bool = True) -> bool:
        """
        The logic of the filter. Returns False if the assembly fails, true else. Used only be self.check
        """


class AssemblyModuleLengthFilter(AssemblyFilter):
    """
    Filter to discard assemblies for which the total size of the modules is not sufficient to reach all goals.

    This filter works with over-approximations of the base placement tolerance and ignores the goal pose tolerances,
    this means it is neither complete, nor is it guaranteed to filter invalid assemblies only if goal tolerances are
    significantly large.

    Attention! This filter caches intermediate computations for modules, identifying them by ID. This means that you
    can't use it for different module databases at the same time.
    """

    goal_types_filtered = GOALS_WITH_POSE
    provides = ()
    requires = ()

    def __init__(self, db: Optional[ModulesDB] = None, skip_not_applicable: bool = True):
        """
        Pre-compute module lengths and initialize the task cache

        :param db: A module database to set up this filter -- this is needed for pre-computing module lengths.
        """
        super().__init__(skip_not_applicable)
        self.module_lengths: Dict[str, float] = dict()
        self.task_required_len_cache: LimitedSizeMap[str, float] = LimitedSizeMap(maxsize=10)
        if db is not None:
            for m in db:
                self.module_lengths[m.id] = self._get_module_length(m)

    def reset(self):
        """Resets the task cache"""
        super().reset()
        self.task_required_len_cache.reset()

    def _check(self, assembly: ModuleAssembly, task: Task.Task, results: IntermediateFilterResults,
               stop_early: bool = True) -> bool:
        """Sums up the module lengths and compares them to a required distance"""
        required_len = self._get_task_required_length(task)
        module_len = sum(self._get_module_length(m) for m in assembly.module_instances)
        return module_len >= required_len

    def _get_module_length(self, module: AtomicModule):
        """Computes an over-approximation of connector-to-connector distance for a module"""
        if module.id in self.module_lengths:
            return self.module_lengths[module.id]
        distance = -1
        for c1, c2 in itertools.product(module.connectors_by_own_id.values(), repeat=2):
            path = nx.shortest_path(module.module_graph, c1, c2)
            d = 0
            for n1, n2 in zip(path[:-1], path[1:]):
                d += module.module_graph.edges[n1, n2]['transform'].norm.translation_euclidean
            distance = max(distance, d)
        self.module_lengths[module.id] = distance
        return distance

    def _get_task_required_length(self, task: Task.Task) -> float:
        """Computes the maximum distance between any goal position and the base placement in the task."""
        if task.id in self.task_required_len_cache:
            return self.task_required_len_cache[task.id]
        distance = -1
        base_nominal = task.base_constraint.base_pose.nominal
        base_tolerance = task.base_constraint.base_pose.tolerance
        if isinstance(task.base_constraint.base_pose.tolerance, Tolerance.Composed):
            base_tolerance = Tolerance.Composed(
                [t for t in base_tolerance.tolerances if isinstance(t, Tolerance.Cartesian)])
        if isinstance(base_tolerance, Tolerance.CartesianSpheric):
            tol = base_tolerance.stacked[0, 1]  # That's the upper limit for the radius
        elif isinstance(base_tolerance, Tolerance.CartesianXYZ):
            # We over-approximate the tolerance by a sphere around the nominal pose
            tol = np.sum(np.abs(base_tolerance.stacked))
        else:
            raise ValueError(f'Cannot compute required robot length for a {type(base_tolerance)} base tolerance.')
        for goal in task.goals:
            if not self._applies(goal):
                continue
            if isinstance(goal, Goals.Follow):
                d = max(base_nominal.absolute.distance(n.nominal.absolute).translation_euclidean
                        for n in goal.trajectory.pose)
            elif isinstance(goal, (Goals.At, Goals.Reach)):
                d = base_nominal.absolute.distance(goal.goal_pose.nominal.absolute).translation_euclidean
            else:
                raise NotImplementedError(f'Cannot compute required robot length for a goal of type {type(goal)}.')
            distance = max(distance, d)
        distance = distance - tol
        self.task_required_len_cache[task.id] = distance
        return distance


class RobotCreationFilter(AssemblyFilter):
    """Filter to explicitly create a robot object for a given assembly."""

    provides = (ResultType.robot,)
    requires = ()

    def _check(self, assembly: ModuleAssembly, task: Task.Task, results: IntermediateFilterResults,
               stop_early: bool = True) -> bool:
        """Turns assembly into a robot and stores it in the intermediate results."""
        results.robot = assembly.robot
        return True

    def _applies(self, goal: Goals.GoalBase) -> bool:
        """Robot can be created for any goal"""
        return True


class GoalByGoalFilter(AssemblyFilter, abc.ABC):
    """The superclass for all filters that check a condition in a task goal by goal."""

    def __init__(self, skip_not_applicable: bool = True, use_intermediate_results: bool = True):
        """A goal by goal filter checks a condition for each goal in a task.

        :param use_intermediate_results: If false, intermediate results are not read from during a check. They will be
          written to, though.
        """
        super().__init__(skip_not_applicable)
        self._goals_success: int = 0
        self._goals_failed: int = 0
        self.use_intermediate_results: bool = use_intermediate_results

    def _check(self, assembly: ModuleAssembly, task: Task.Task, results: IntermediateFilterResults,
               stop_early: bool = True) -> bool:
        """Perform a validity check on all goals in sequence and interrupts if one fails.

        Takes over the looping and separation of goals that are already solved and just need to be checked
        and those that need some more calculation.
        """
        valid = True
        for goal in self.goals_applicable(task):
            if self.use_intermediate_results \
                    and len(self.provides) > 0 \
                    and all(goal.id in results.__getattribute__(provided.name) for provided in self.provides):
                # In this case, it is sufficient to check already existing intermediate results
                valid_goal = self._check_given(assembly, goal, results, task)
            else:
                with results.allow_changes():  # Potentially, some of the results are overwritten
                    valid_goal = self._check_goal(assembly, goal, results, task)
            valid &= valid_goal
            if valid_goal:
                self._goals_success += 1
            else:
                self._goals_failed += 1
            if not valid and stop_early:
                return False
        return valid

    @abstractmethod
    def _check_given(self, assembly: ModuleAssembly, goal: Goals.GoalBase,
                     results: IntermediateFilterResults, task: Task):
        """Checks a predefined solution.

        Called when all required calculations for this filter already have been done: In this case, check if they
        are valid only.
        """

    @abstractmethod
    def _check_goal(self, assembly: ModuleAssembly, goal: Goals.GoalBase,
                    results: IntermediateFilterResults, task: Task):
        """The same logic as _check, but performed on a single goal that this filter can work with."""

    @property
    def goals_success(self) -> int:
        """
        A counter on how many goals passed this filter.

        Note that in case one goal did not pass this filter, no further goals of the same tasks are even evaluated.
        """
        return self._goals_success

    @property
    def goals_failed(self) -> int:
        """
        A counter on how many goals failed this filter.

        Note that in case one goal did not pass this filter, no further goals of the same tasks are even evaluated.
        """
        return self._goals_failed

    def reset(self):
        """Resets all counters of this filter"""
        super().reset()
        self._goals_success = 0
        self._goals_failed = 0


class InverseKinematicsSolvable(GoalByGoalFilter):
    """
    Checks whether all goal poses can be theoretically reached by the assembly without colliding with itself.

    Does not check for collisions with the environment.
    Does only work with goals that have one pose at most. Filtering assemblies based on target trajectories needs to be
    done within another class.

    :note: Can be used to also verify freedom of collision by setting ik_kwargs to the task to check collisions with
    """

    requires: Tuple[ResultType] = (ResultType.robot,)
    provides: Tuple[ResultType] = (ResultType.q_goal_pose, ResultType.goal_ik_success)
    goal_types_filtered: Collection[Type[Goals.GoalBase]] = GOALS_WITH_ONE_POSE

    def __init__(self, skip_not_applicable: bool = True, **ik_kwargs):
        """Initialize the filter on inverse kinematics

        :param ik_kwargs: Keyword arguments to be passed to the :func:`inverse kinematics solver <timor.Robot.Robot.ik>`
        """
        self.kwargs: Dict[str, any] = ik_kwargs
        if 'task' in self.kwargs:
            self.provides += (ResultType.collision_free_goal,)
        for c in itertools.chain(ik_kwargs.get('outer_constraints', ()),
                                 ik_kwargs.get('task', Task.Task.empty()).constraints):
            if isinstance(c, Constraints.JointLimits):
                if c.tau:
                    self.provides += (ResultType.tau_static,)
                    break
        super().__init__(skip_not_applicable)

    @property
    def ensures_static_torques(self) -> bool:
        """Does this filter ensure that static torques are respected?"""
        return ResultType.tau_static in self.provides

    @property
    def ensures_collision_free_goal(self) -> bool:
        """Does this filter ensure collision-freedom with respect to a task's scene?"""
        return 'task' in self.kwargs

    def _check_given(self, assembly: ModuleAssembly, goal: Goals.GoalBase, results: IntermediateFilterResults,
                     task: Task):
        """Just checks that the goal pose is reached without self collisions"""
        if not self._applies_to_type(type(goal)):
            return True  # Not applicable, so filter is not invalid
        is_pose = results.robot.fk(results.q_goal_pose[goal.id])
        passes = goal.goal_pose.valid(is_pose) and not results.robot.has_self_collision()
        if self.ensures_collision_free_goal:
            passes &= not results.robot.has_collisions(self.kwargs['task'])
        if self.ensures_static_torques:
            tau = results.robot.static_torque(results.q_goal_pose[goal.id])
            passes &= (np.abs(tau) <= results.robot.joint_torque_limits).all()
        return passes

    def _check_goal(self, assembly: ModuleAssembly, goal: GOALS_WITH_ONE_POSE, results: IntermediateFilterResults,
                    task: Task) -> bool:
        """Performs the check whether the goals are valid.

        Assemblies pass if and only if, for all goal poses, a self-collision free inverse kinematics solution can be
        found. Writes an inverse kinematics solution for the goal pose to the goal, if it finds one.
        """
        q, valid = results.robot.ik(goal.goal_pose, **self.kwargs.copy())
        results.q_goal_pose[goal.id] = q
        results.goal_ik_success[goal.id] = valid
        if self.ensures_static_torques:
            results.tau_static[goal.id] = valid or (np.abs(results.robot.static_torque(results.q_goal_pose[goal.id]))
                                                    <= results.robot.joint_torque_limits).all()
        if self.ensures_collision_free_goal:
            results.robot.update_configuration(q)
            if valid:
                results.collision_free_goal[goal.id] = True
        return valid

    def __repr__(self):
        """Debug string to identify the filter incl IK kwargs"""
        return f"{self.__class__.__name__}({self.kwargs})"


class StaticTorquesValid(GoalByGoalFilter):
    """
    Checks whether the static torques for assembly joints in a certain configuration are within the torque limits.

    Currently, operates without static forces ==> gravity only.
    """

    requires: Tuple[ResultType] = (ResultType.q_goal_pose, ResultType.robot)
    provides: Tuple[ResultType] = (ResultType.tau_static,)
    goal_types_filtered: Collection[Type[Goals.GoalBase]] = GOALS_WITH_ONE_POSE

    def _check_given(self, assembly: ModuleAssembly, goal: Goals.GoalBase, results: IntermediateFilterResults,
                     task: Task):
        tau = results.tau_static[goal.id]
        return (np.abs(tau) <= results.robot.joint_torque_limits).all()

    def _check_goal(self, assembly: ModuleAssembly, goal: Goals.GoalBase, results: IntermediateFilterResults,
                    task: Task):
        """Calculates static torques in a goal position and compares them to the robot assembly limits.

        Writes the static torques to the intermediate results.
        """
        q = results.q_goal_pose[goal.id]
        results.robot.update_configuration(q)
        tau = results.robot.static_torque(q)
        results.tau_static[goal.id] = tau
        return (np.abs(tau) <= results.robot.joint_torque_limits).all()


class JointLimitsMet(AssemblyFilter):
    """
    Checks whether a assembly's joint (position, velocity) and the torque limits are held for a trajectory.
    """

    requires: Tuple[ResultType] = (ResultType.trajectory, ResultType.robot)
    provides: Tuple[ResultType] = ()
    goal_types_filtered: Collection[Type[Goals.GoalBase]] = tuple()  # TODO: Implement trajectory goals

    def _check(self, assembly: ModuleAssembly, task: Task.Task, results: IntermediateFilterResults,
               stop_early: bool = True) -> bool:
        """Performs the check whether the joint limits are met.

        Returns true if and only if joint limits, velocity limits and torque limits are held for all trajectories in
        the intermediate results.
        """
        # TODO: Access necessary trajectories from goals and then check if they are in the results
        valid = True
        for trajectory in results.trajectory.values():
            q, dq, ddq = trajectory.q, trajectory.dq, trajectory.ddq
            q_valid = results.robot.q_in_joint_limits(q)
            dq_valid = np.all(np.abs(dq) <= results.robot.joint_velocity_limits)
            ddq_valid = np.all(np.abs(ddq) <= results.robot.joint_acceleration_limits)
            valid = valid and q_valid and dq_valid and ddq_valid
            if not valid and stop_early:
                return False

            tau = np.vstack([results.robot.id(p, v, a) for p, v, a in zip(q, dq, ddq)])
            valid = valid and (np.abs(tau) < results.robot.joint_torque_limits).all()

        return valid


def assert_filters_compatible(filters: Iterable[Type[AssemblyFilter]]) -> None:
    """
    Checks that a sequence of filters can be used in this ordering.

    :param filters: A sequence of filter types
    :raises: InvalidFilerCombinationError
    """
    requirements_given = list()
    for filter_class in filters:
        for requirement in filter_class.requires:
            if requirement not in requirements_given:
                raise InvalidFilterCombinationError()
        requirements_given.extend(filter_class.provides)


def default_filters(task: Task.Task) -> Tuple[AssemblyFilter, ...]:
    """
    Default filters for assembly planning.

    These filters are designed to handle various aspects of robot manipulation and can be used as a starting point
    for configuring the system and optimizing a robot assembly and adapting the robot to the desired task.

    The filters in this function ensure the following:
    - RobotCreationFilter: Ensures the robot instance is created and can be used for further filters.
    - InverseKinematicsSolvable: Ensures the goals can be reached with a valid inverse kinematics solution
    without collisions and within static torque limits.

    :param task: The task to be evaluated.
    """
    assembly_length = AssemblyModuleLengthFilter()
    create_robot = RobotCreationFilter()
    ik_simple = InverseKinematicsSolvable(ignore_self_collisions=True, max_iter=150)
    ik_complex = InverseKinematicsSolvable(task=task, max_iter=1500)
    return assembly_length, create_robot, ik_simple, ik_complex


def run_filters(filters: Sequence[AssemblyFilter],
                assembly: ModuleAssembly,
                task: Task.Task) -> Tuple[IntermediateFilterResults, bool]:
    """
    Runs a sequence of filters on an assembly.

    :param filters: A sequence of filter types
    :param assembly: The assembly to be checked
    :param task: The task to be evaluated
    :return: True, if the assembly passes all filters + all results generated by the filters
    """
    assert_filters_compatible((type(f) for f in filters))
    results = IntermediateFilterResults()
    for fltr in filters:
        with results.allow_changes():
            if not fltr.check(assembly, task, results):
                return results, False
    return results, True
