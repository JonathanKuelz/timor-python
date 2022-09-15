import abc
import datetime
import json
from pathlib import Path
from typing import Dict, NamedTuple, Tuple, Union
import warnings

import numpy as np
import pinocchio as pin

from timor.Module import ModuleAssembly, ModulesDB
from timor.Robot import RobotBase
from timor.scenario import Constraints, CostFunctions, Goals, Scenario
from timor.utilities.dtypes import Lazy, TorqueInput, Trajectory, fuzzy_dict_key_matching  # noqa: F401
from timor.utilities.file_locations import get_module_db_files
from timor.utilities.transformation import TransformationLike, Transformation
from timor.utilities.visualization import animation, MeshcatVisualizerWithAnimation


class SolutionHeader(NamedTuple):
    """The header every solution contains"""

    scenarioID: str  # This is NOT the Solution ID, but identifies the scenario the solution was crafted for!
    version: str = "Py2022"
    author: str = ''
    authorEMail: str = ''
    affiliation: str = ''
    publication: str = ''
    date: datetime.datetime = datetime.datetime(1970, 1, 1)
    computationTime: float = -1.
    processor: str = ''


class SolutionBase(abc.ABC):
    """A solution describes a robot and a trajectory it follows in order to reach all goals in a scenario"""

    def __init__(self,
                 header: Union[Dict, SolutionHeader],
                 scenario: 'Scenario.Scenario',
                 robot: RobotBase,
                 cost_function: 'CostFunctions.CostFunctionBase',
                 base_pose: TransformationLike = Transformation.neutral(),
                 ):
        """Build a solution that shows how to solve a Task.

        :param header: The header of the solution, containing information about the scenario it was crafted for and
          some more distinctive meta-information.
        :param scenario: The scenario the solution was crafted for
        :param robot: The robot that solves the scenario
        :param cost_function: The cost function used to evaluate the solution (usually the lower, the better)
        :param base_pose: The base placement of the robot
        """
        self.header: SolutionHeader = header if isinstance(header, SolutionHeader) else SolutionHeader(**header)
        self.cost_function: 'CostFunctions.CostFunctionBase' = cost_function
        self._cost: Union[Lazy, bool] = Lazy(self._evaluate_cost)
        self._valid: Union[Lazy, bool] = Lazy(self._check_valid)  # Will be evaluated when needed only

        self._scenario: 'Scenario.Scenario' = scenario
        robot.set_base_placement(base_pose)
        self._robot: RobotBase = robot

    def __str__(self):
        """String representation of the solution"""
        return f"Solution for scenario {self.header.scenarioID}"

    @staticmethod
    def from_json(json_path: Path, package_dir: Path, scenarios: Dict[str, 'Scenario.Scenario']) -> 'SolutionBase':
        """Factory method to load a class instance from a json file."""
        content = json.load(json_path.open('r'))
        _header = fuzzy_dict_key_matching(content, desired_only=SolutionHeader._fields)
        _header['scenarioID'] = str(_header['scenarioID'])
        header = SolutionHeader(**_header)
        try:
            sol_scenario = scenarios[header.scenarioID]
        except KeyError:
            raise KeyError(f"Got solution for scenario {header.scenarioID}, but there is no such scenario.")

        modules, package = get_module_db_files(content['moduleSet'])
        db = ModulesDB.from_file(modules, package)

        if 'moduleOrder' in content:
            warnings.warn("Module Order should be replaced by proper module arrangement definition",
                          DeprecationWarning)
            robot = ModuleAssembly.from_serial_modules(db, tuple(map(str, content['moduleOrder'])))
        else:
            robot = ModuleAssembly(db)

        robot = robot.to_pin_robot()
        cost_func = CostFunctions.CostFunctionBase.from_crok_descriptor(content["costFunction"])
        base_pose = np.array(content['basePose'])

        if 'trajectory' in content:
            _trajectory = fuzzy_dict_key_matching(content['trajectory'], {'goal2time': 'goals'},
                                                  desired_only=tuple(Trajectory.__dataclass_fields__.keys()))
            trajectory = Trajectory(**_trajectory)
            return SolutionTrajectory(trajectory, header, sol_scenario, robot, cost_func, base_pose)
        else:
            raise NotImplementedError("Only trajectory from json so far ")

    def to_dict(self) -> Dict[str, any]:
        """Should be implemented by children to dump a solution"""
        raise NotImplementedError

    @property
    def cost(self) -> float:
        """Evaluating lazy variable _cost"""
        return self._cost()

    @property
    def failed_constraints(self) -> Tuple['Constraints.ConstraintBase', ...]:
        """A tuple of all scenario-level constraints that were hurt by this solution"""
        return tuple(c for c in self.scenario.constraints if not c.fulfilled(self))

    @property
    def failed_goals(self) -> Tuple['Goals.GoalBase', ...]:
        """A tuple of all goals that were not fulfilled by this solution"""
        return tuple(g for g in self.scenario.goals if not g.achieved(self))

    @property
    def t_resolution(self) -> float:
        """All goal times can be at most t_resolution apart from the next time step defined in self"""
        return self.scenario.header.timeStepSize

    @property
    def robot(self):
        """Prevent setting robot directly"""
        return self._robot

    @property
    def scenario(self):
        """Prevent setting scenario directly"""
        return self._scenario

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
        """Evaluates the validity of this solution.

        Returns true if all goals of the internal scenario are solved while all constraints
        are fulfilled by this solution.
        """
        goals = all(goal.achieved(self) for goal in self.scenario.goals)
        global_constraints = all(constraint.fulfilled(self) for constraint in self.scenario.constraints)
        return goals and global_constraints

    def _evaluate_cost(self) -> float:
        """Evaluates the cost function - without checking validity though"""
        return self.cost_function.evaluate(self)

    def get_time_id(self, t: float) -> int:
        """Get the index of the time step at which the given time is located"""
        try:
            return int(np.argwhere(np.isclose(self.time_steps, t)))
        except TypeError:  # Casting empty array to int fails
            raise ValueError(f"Time {t} is not in the time steps of the solution.")

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
                  viz: Union[pin.visualize.MeshcatVisualizer, MeshcatVisualizerWithAnimation] = None) \
            -> pin.visualize.MeshcatVisualizer:
        """Visualize a solution trajectory"""
        if viz is None:
            viz = MeshcatVisualizerWithAnimation()
            viz.initViewer()

        if isinstance(viz, pin.visualize.MeshcatVisualizer):
            viz = MeshcatVisualizerWithAnimation.from_MeshcatVisualizer(viz)

        viz = self.scenario.visualize(viz, robots=self.robot)
        for goal in self.scenario.goals:
            goal.visualize(viz)

        animation(self.robot, self.q, 1 / (self.time_steps[-1] - self.time_steps[0]), visualizer=viz)

        return viz


class SolutionTrajectory(SolutionBase):
    """A solution based on a given trajectory (as opposed to a solution based on torques given)"""

    def __init__(self,
                 trajectory: Trajectory,
                 header: Union[Dict, SolutionHeader],
                 scenario: 'Scenario.Scenario',
                 robot: RobotBase,
                 cost_function: 'CostFunctions.CostFunctionBase',
                 base_pose: TransformationLike = Transformation.neutral(),
                 ):
        """Build a solution based on a given trajectory

        :param trajectory: The solution trajectory
        :param header: The header of the solution
        :param scenario: The scenario that the solution belongs to
        :param robot: The robot that performs the given trajectory
        :param cost_function: The cost function that should be used to evaluate the solution
        :param base_pose: The base placement of the robot
        """
        super().__init__(header, scenario, robot, cost_function, base_pose)
        self.trajectory: Trajectory = trajectory
        self._torques = Lazy(self._get_torques)

    def to_dict(self) -> Dict[str, any]:
        """Future todo"""
        raise NotImplementedError()

    def _get_torques(self) -> np.ndarray:
        """Utility method to calculate joint torques based on the trajectory"""
        torques = np.zeros_like(self.q)
        for i, (q, dq, ddq) in enumerate(zip(self.q, self.dq, self.ddq)):
            torques[i, :] = self.robot.id(q, dq, ddq)
        return torques

    @property
    def t_goals(self) -> Dict[str, float]:
        """Mapping from goal ID to the time at which the goal was reached"""
        return self.trajectory.goals

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
        """The heart of a trajectory solution. Describes robot trajectory to follow in order to solve the scenario"""
        return self._trajectory

    @trajectory.setter
    def trajectory(self, t: Trajectory):
        """Small type check to prevent invalid solutions. Also resets the validity and calculated cost."""
        if not isinstance(t, Trajectory):
            raise TypeError(f"Invalid argument for solution trajectory of type {type(t)}")
        self._valid.value = None
        self._cost.value = None
        self._trajectory = t
