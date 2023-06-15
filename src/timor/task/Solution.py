from __future__ import annotations

import abc
from collections import namedtuple
from copy import copy
from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import uuid

import jsonschema.exceptions
from matplotlib import pyplot as plt
import numpy as np
import pinocchio as pin
import scipy.interpolate

from timor.Module import ModuleAssembly
from timor.Robot import RobotBase
from timor.task import Constraints, CostFunctions, Goals, Task
from timor.utilities import logging
from timor.utilities.dtypes import Lazy, TypedHeader, fuzzy_dict_key_matching  # noqa: F401
from timor.utilities.errors import TimeNotFoundError
from timor.utilities.file_locations import map2path, schema_dir
from timor.utilities.json_serialization_formatting import compress_json_vectors
from timor.utilities.jsonable import JSONable_mixin
from timor.utilities.schema import DEFAULT_DATE_FORMAT, get_schema_validator
from timor.utilities.trajectory import Trajectory
from timor.utilities.transformation import Transformation, TransformationLike
from timor.utilities.visualization import MeshcatVisualizerWithAnimation, animation, plot_time_series


@dataclass
class SolutionHeader(TypedHeader):
    """The header every solution contains"""

    taskID: str  # This is NOT the Solution ID, but identifies the task the solution was crafted for!
    version: str = "2023"
    author: List[str] = field(default_factory=lambda: [''])
    email: List[str] = field(default_factory=lambda: [''])
    affiliation: List[str] = field(default_factory=lambda: [''])
    publication: str = ''
    date: datetime.date = datetime.date.today()
    computationTime: float = -1.
    processorName: str = ''

    @classmethod
    def empty(cls) -> SolutionHeader:
        """Creates an empty solution header with a random taskID"""
        return cls(taskID=f'tmp_{uuid.uuid4()}')


class SolutionBase(abc.ABC, JSONable_mixin):
    """A solution describes an assembly and a trajectory it follows in order to reach all goals in a task"""

    def __init__(self,
                 header: Union[Dict, SolutionHeader],
                 task: Task.Task,
                 assembly: ModuleAssembly,
                 cost_function: CostFunctions.CostFunctionBase,
                 base_pose: TransformationLike = Transformation.neutral(),
                 ):
        """
        Build a solution that shows how to solve a Task.

        :param header: The header of the solution, containing information about the task it was crafted for and
            some more distinctive meta-information.
        :param task: The task the solution was crafted for
        :param assembly: The assembly that solves the task
        :param cost_function: The cost function used to evaluate the solution (usually the lower, the better)
        :param base_pose: The base placement of the assembly
        """
        self.header: SolutionHeader = header if isinstance(header, SolutionHeader) else SolutionHeader(**header)
        self.cost_function: CostFunctions.CostFunctionBase = cost_function
        self._module_assembly: ModuleAssembly = assembly
        self._base_pose: Transformation = Transformation(base_pose)
        self._cost: Union[Lazy, bool] = Lazy(self._evaluate_cost)
        self._valid: Union[Lazy, bool] = Lazy(self._check_valid)  # Will be evaluated when needed only
        self._task: Task.Task = task
        self.robot.set_base_placement(base_pose)

    def __str__(self):
        """String representation of the solution"""
        return f"Solution for task {self.header.taskID}"

    @classmethod
    @abc.abstractmethod
    def empty(cls) -> SolutionBase:
        """Create an empty solution"""

    @classmethod
    def from_json_file(cls, json_path: Union[Path, str], tasks: Dict[str, Task.Task]) -> SolutionBase:
        """Factory method to load a class instance from a json file."""
        json_path = map2path(json_path)
        with json_path.open('r') as f:
            content = json.load(f)
        _, validator = get_schema_validator(schema_dir.joinpath("SolutionSchema.json"))
        try:
            validator.validate(content)
        except jsonschema.exceptions.ValidationError:
            raise ValueError(f"Invalid solution json provided. Details: {tuple(validator.iter_errors(content))}.")
        return cls.from_json_data(content, tasks)

    @staticmethod
    def from_json_data(content: Dict, tasks: Dict[str, Task.Task], *args, **kwargs) -> SolutionBase:
        """Factory method to load a class instance from a dictionary."""
        _header = fuzzy_dict_key_matching(content, desired_only=SolutionHeader.fields())
        header = SolutionHeader(**_header)
        try:
            sol_task = tasks[header.taskID]
        except KeyError:
            raise KeyError(f"Got solution for task {header.ID}, but there is no such task.")

        assembly = ModuleAssembly.from_json_data(content)

        cost_func = CostFunctions.CostFunctionBase.from_descriptor(content["costFunction"])
        base_pose = np.array(content['basePose'][0]).squeeze()
        if base_pose.shape != (4, 4):
            raise ValueError("Cannot handle assembly with multiple bases")

        if 'trajectory' in content:
            trajectory = Trajectory.from_json_data(content['trajectory'])
            return SolutionTrajectory(trajectory, header, sol_task, assembly, cost_func, base_pose)
        else:
            raise NotImplementedError("Only trajectory from json so far ")

    @abc.abstractmethod
    def to_json_data(self) -> Dict[str, any]:
        """Should be implemented by children to dump a solution"""

    def to_json_string(self) -> str:
        """Convert the solution to a json string"""
        return compress_json_vectors(json.dumps(self.to_json_data(), indent=2))

    @property
    def module_assembly(self) -> ModuleAssembly:
        """The robot / module assembly set to solve this task."""
        return self._module_assembly

    @property
    def cost(self) -> float:
        """Evaluating lazy variable _cost"""
        return self._cost()

    @property
    def failed_constraints(self) -> Tuple[Constraints.ConstraintBase, ...]:
        """A tuple of all task-level constraints that were hurt by this solution"""
        return tuple(c for c in self.task.constraints if not c.fulfilled(self))

    @property
    def failed_goals(self) -> Tuple[Goals.GoalBase, ...]:
        """A tuple of all goals that were not fulfilled by this solution"""
        return tuple(g for g in self.task.goals if not g.achieved(self))

    @property
    def t_resolution(self) -> float:
        """All goal times can be at most t_resolution apart from the next time step defined in self"""
        return self.task.header.timeStepSize

    @property
    def robot(self) -> RobotBase:
        """Provide read access to the robot. This call might be expensive, depending on whether the robot is cached."""
        self._module_assembly.robot.set_base_placement(self._base_pose)
        return self._module_assembly.robot

    @property
    def task(self):
        """Prevent setting task directly"""
        return self._task

    @property
    def valid(self) -> bool:
        """Evaluating lazy variable _lazy"""
        return self._valid()

    # --------------- Fast access properties for important verification sizes ---------------
    @property
    @abc.abstractmethod
    def t_goals(self) -> Dict[str, float]:
        """Mapping from goal IDs to the time at which they were achieved"""
        pass

    @property
    @abc.abstractmethod
    def time_steps(self) -> np.ndarray:
        """Time steps of the trajectory"""
        pass

    @property
    @abc.abstractmethod
    def q(self) -> np.ndarray:
        """Joint positions of the trajectory"""
        pass

    @property
    @abc.abstractmethod
    def dq(self) -> np.ndarray:
        """Joint velocities of the trajectory"""
        pass

    @property
    @abc.abstractmethod
    def ddq(self) -> np.ndarray:
        """Joint accelerations of the trajectory"""
        pass

    @property
    @abc.abstractmethod
    def torques(self) -> np.ndarray:
        """Torques of the trajectory"""
        pass

    @property
    def tau(self) -> np.ndarray:
        """Alias for torques"""
        return self.torques

    def _check_valid(self) -> bool:
        """
        Returns true if all all constraints are fulfilled by this solution.

        :note: add allGoalsFulfilled Constraint if you want to ensure all goals and their constraints are fulfilled
        """
        global_constraints = all(constraint.fulfilled(self) for constraint in self.task.constraints)
        return global_constraints

    def _evaluate_cost(self) -> float:
        """Evaluates the cost function - without checking validity though"""
        return self.cost_function.evaluate(self)

    def get_time_id(self, t: float) -> int:
        """Get the index of the time step at which the given time is located"""
        idx_candidates = np.argwhere(np.isclose(self.time_steps, t))
        if idx_candidates.shape[0] != 1:
            raise TimeNotFoundError(f"Time {t} is found at index {idx_candidates}.")
        return int(idx_candidates)

    def tcp_pose_at(self, t: float) -> Transformation:
        """
        Returns the placement of the tool center point at time t

        :param t: The time in the solution at which the tcp position should be given. Must be one of the time steps in
            the provided (trajectory | torques).
        """
        q = self.q[self.get_time_id(t)]
        return self.robot.fk(q, kind='tcp')

    def tcp_velocity_at(self, t: float) -> np.ndarray:
        """
        Returns the velocity of the tool center point at time t

        :param t: The time in the solution at which the tcp velocity should be given. Must be one of the time steps in
            the provided (trajectory | torques).
        :return: Velocity [m/s]
        """
        q = self.q[self.get_time_id(t)]
        dq = self.dq[self.get_time_id(t)]
        self.robot.update_configuration(q=q, dq=dq)
        return self.robot.tcp_velocity

    def tcp_acceleration_at(self, t: float) -> np.ndarray:
        """
        Returns the velocity of the tool center point at time t

        :param t: The time in the solution at which the tcp acceleration should be given.
            Must be one of the time steps in the provided (trajectory | torques).
        :return: Acceleration [m / s**2]
        """
        q = self.q[self.get_time_id(t)]
        dq = self.dq[self.get_time_id(t)]
        ddq = self.ddq[self.get_time_id(t)]
        self.robot.update_configuration(q=q, dq=dq, ddq=ddq)
        return self.robot.tcp_acceleration

    def visualize(self,
                  viz: Union[pin.visualize.MeshcatVisualizer, MeshcatVisualizerWithAnimation] = None,
                  fps: float = 30.,
                  center_view: bool = True) \
            -> pin.visualize.MeshcatVisualizer:
        """
        Visualize a solution trajectory

        :param viz: Visualizer to use (default create new)
        :param fps: fps to use for visualization
        :param center_view: Whether to try to set camera such that robot / base pose constraint visible
        """
        if self.robot.njoints == 0:
            logging.warning("You tried to visualize a solution for a robot with no joints. Aborting.")
            return viz
        if self.q.size == 0:
            logging.warning("You tried to visualize a solution with an empty trajectory. Aborting.")
            return viz

        if viz is None:
            viz = MeshcatVisualizerWithAnimation()
            viz.initViewer()

        if isinstance(viz, pin.visualize.MeshcatVisualizer):
            viz = MeshcatVisualizerWithAnimation.from_MeshcatVisualizer(viz)

        viz = self.task.visualize(viz, robots=self.robot, center_view=center_view)
        for goal in self.task.goals:
            goal.visualize(viz)

        # resample to capture statics better
        if (self.time_steps is not None) and (self.time_steps.size > 1) and (self.q.size > 1):
            q_new = scipy.interpolate.interp1d(self.time_steps, self.q, axis=0)(np.arange(self.time_steps[0],
                                                                                          self.time_steps[-1], 1 / fps))
        else:
            q_new = self.q

        animation(self.robot, q_new, 1 / fps, visualizer=viz)

        return viz

    def __getstate__(self):
        """State of solution contains json-able part and the associated task object."""
        return self.to_json_data(), {self.header.taskID: self.task}

    def __setstate__(self, state):
        """To recreate solution use the state Tuple including the solution json and associated task object."""
        cpy = self.__class__.from_json_data(*state)
        self.__dict__ = cpy.__dict__


class SolutionTrajectory(SolutionBase):
    """A solution based on a given trajectory (as opposed to a solution based on torques given)"""

    def __init__(self,
                 trajectory: Trajectory,
                 header: Union[Dict, SolutionHeader],
                 task: Task.Task,
                 assembly: ModuleAssembly,
                 cost_function: CostFunctions.CostFunctionBase,
                 base_pose: TransformationLike = Transformation.neutral(),
                 ):
        """Build a solution based on a given trajectory

        :param trajectory: The solution trajectory
        :param header: The header of the solution
        :param task: The task that the solution belongs to
        :param assembly: The assembly of modules representing the assembly that performs the given trajectory
        :param cost_function: The cost function that should be used to evaluate the solution
        :param base_pose: The base placement of the assembly
        """
        super().__init__(header, task, assembly, cost_function, base_pose)
        self.trajectory: Trajectory = trajectory
        self._torques = Lazy(self._get_torques)
        self._link_side_torques = Lazy(lambda: self._get_torques(motor_inertia=False, friction=False))

    @classmethod
    def empty(cls) -> SolutionTrajectory:
        """Returns an empty solution with an empty trajectory"""
        task = Task.Task.empty()
        header = SolutionHeader(taskID=task.id)
        return cls(Trajectory.empty(), header, task, ModuleAssembly.empty(), CostFunctions.CostFunctionBase.empty())

    def to_json_data(self) -> Dict[str, any]:
        """
        Creates jsonable dictionary that serializes this solution.

        :return: A dictionary that serializes this solution
        """
        data = {
            "cost": self.cost,
            "costFunction": self.cost_function.descriptor(),
            **self.header.asdict(),
            **self.module_assembly.to_json_data(),
            "basePose": [self.robot.placement.to_json_data()],
            "trajectory": self.trajectory.to_json_data()
        }
        data['date'] = data['date'].strftime(DEFAULT_DATE_FORMAT)
        return data

    def _get_torques(self, motor_inertia: bool = True, friction: bool = True) -> np.ndarray:
        """
        Utility method to calculate joint torques to command based on the trajectory.

        :param motor_inertia: Include motor side inertia in torque calculations to compensate them
          (details see robot.id)
        :param friction: Include joint friction in torque calculations to compensate for them (details see robot.id)
        :note: Disabling both motor_inertia and friction will lead to the torques provided to the physical links by
          the motor-gearbox subsystem.
        :note: Motor inertia and friction can be the dominating parts of the required torques espacially for robots with
          high gear ratios.
        """
        torques = np.zeros_like(self.q)
        for i, (q, dq, ddq) in enumerate(zip(self.q, self.dq, self.ddq)):
            torques[i, :] = self.robot.id(q, dq, ddq, motor_inertia=motor_inertia, friction=friction)
        return torques

    @property
    def link_side_torques(self):
        """
        Method to calculate the torques acting from the motor-gearbox unit on the different links.

        In comparison to self.torques excludes the torques required to counteract joint friction and rotor inertia
        """
        return self._link_side_torques()

    @property
    def t_goals(self) -> Dict[str, float]:
        """Mapping from goal ID to the time at which the goal was reached"""
        return self.trajectory.goal2time

    @property
    def time_steps(self) -> np.ndarray:
        """The time steps of the trajectory"""
        return self.trajectory.t

    @property
    def q(self) -> np.ndarray:
        """The joint positions of the trajectory"""
        return self.trajectory.q

    @property
    def dq(self) -> np.ndarray:
        """The joint velocities of the trajectory"""
        return self.trajectory.dq

    @property
    def ddq(self) -> np.ndarray:
        """The joint accelerations of the trajectory"""
        return self.trajectory.ddq

    @property
    def torques(self) -> np.ndarray:
        """The joint torques of the trajectory"""
        return self._torques()

    @property
    def trajectory(self) -> Trajectory:
        """The heart of a trajectory solution. Describes robot trajectory to follow in order to solve the task"""
        return self._trajectory

    @trajectory.setter
    def trajectory(self, t: Trajectory):
        """Small type check to prevent invalid solutions. Also resets the validity and calculated cost."""
        if not isinstance(t, Trajectory):
            raise TypeError(f"Invalid argument for solution trajectory of type {type(t)}")
        self._valid.value = None
        self._cost.value = None
        self._trajectory = t

    @property
    def manipulability_indices(self) -> np.ndarray:
        """Return manipulability index at each time in time_steps."""
        return np.fromiter((self.robot.manipulability_index(q) for q in self.trajectory.q),
                           dtype=np.single, count=len(self.trajectory))

    @property
    def tcp_poses(self) -> Tuple[Transformation, ...]:
        """Return TCP pose at each time in time_steps."""
        return tuple(self.tcp_pose_at(t) for t in self.time_steps)

    def get_power(self, motor_inertia: bool = True, friction: bool = True):
        """
        Get mechanical power used in all joints for each time step.

        :param motor_inertia: Include power needed to overcome motor side inertia.
        :param friction: Include power needed to overcome joint friction.
        :note: If both options enabled this is the mechanical power produced by all motors in the robot; if both options
          are disabled it is the mechanical power acting on all robot links.
        """
        if motor_inertia and friction:
            torques = self.torques
        elif not motor_inertia and not friction:
            torques = self.link_side_torques
        else:
            torques = self._get_torques(motor_inertia, friction)

        steps = np.abs(np.einsum("ij,ij->i", torques, self.dq))[:-1] * (self.time_steps[1:] - self.time_steps[:-1])

        return np.hstack((steps, [0.]))

    def get_mechanical_energy(self, motor_inertia: bool = True, friction: bool = True):
        """
        Get mechanical energy needed to follow this solution.

        :param motor_inertia: Include energy to overcome motor side inertia.
        :param friction: Include energy to overcome joint friction.
        :note: If both options enabled this is the mechanical energy expanded by all motors in the robot to follow this
          solution; if both options are disabled it is the mechanical energy expanded on moving all robot links.
        """
        return np.sum(self.get_power(motor_inertia, friction))

    def plot(self, data: Sequence[str] = ("q", "dq", "ddq"), show_figure: bool = True,
             subplot_kwargs: Optional[Dict] = None) -> plt.Figure:
        """
        Plot time series data about this solution

        :param data: Can be any combination of q, dq, ddq, tau, manipulability, goals, eef_position, eef_rotation
        :param show_figure: execute pyplot show() function on figure; might need to disable if further figure formating
          wanted later on; may need to activate text.usetex in the local environment where show is called.
        :param subplot_kwargs: Kwargs handed to subplots, including pyplot.figure kwargs
          (see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)
        :return: Figure object containing the desired plots
        """
        marker = {}
        time_series = []
        legends = []
        available_series_data = namedtuple("available_series_data", ["time_series", "name", "legend_labels"])
        available = {
            "q": available_series_data(
                time_series=self.q,
                name="Configuration q",
                legend_labels=tuple(r"$q_{" + str(i) + "}$" for i in range(self.trajectory.dof))),
            "dq": available_series_data(
                time_series=self.dq,
                name="Joint Velocity dq",
                legend_labels=tuple(r"$\dot{{q}}_{" + str(i) + "}$" for i in range(self.trajectory.dof))),
            "ddq": available_series_data(
                time_series=self.ddq,
                name="Joint Acceleration ddq",
                legend_labels=tuple(r"$\ddot{{q}}_{" + str(i) + "}$" for i in range(self.trajectory.dof))),
            "tau": available_series_data(
                time_series=self.torques,
                name="Joint Torques",
                legend_labels=tuple(r"$\tau_{" + str(i) + "}$" for i in range(self.trajectory.dof))),
            "manipulablility": available_series_data(
                time_series=self.manipulability_indices,
                name="Manipulability Index",
                legend_labels=("Manipulability", )),
            "eef_position": available_series_data(
                time_series=tuple(self.tcp_pose_at(t).translation for t in self.time_steps),
                name="EEF Position",
                legend_labels=(r"$x$", r"$y$", r"$z$")),
            "eef_rotation": available_series_data(
                time_series=tuple(self.tcp_pose_at(t).projection.roto_translation_vector for t in self.time_steps),
                name="EEF Rotation",
                legend_labels=("$n_x$", r"$n_y$", "$n_z$"))
        }

        for d in data:
            if d == "goals":
                marker = copy(marker)
                marker.update({t: (g, "red" if g in (failed.id for failed in self.failed_goals) else "green")
                               for g, t in self.trajectory.goal2time.items()})
                logging.info("Showing fulfilled goals in green and failed in red.")
                continue
            if d not in available:
                raise ValueError(f"Cannot plot {d} in solution")
            time_series.append((available[d].time_series, available[d].name))
            legends.append(available[d].legend_labels)

        f = plot_time_series(self.time_steps, time_series, marker, subplot_kwargs=subplot_kwargs)
        for i, l in enumerate(legends):
            f.axes[i].legend(l, loc="upper right")

        if show_figure:
            with plt.rc_context({'text.usetex': True}):
                f.show()

        return f
