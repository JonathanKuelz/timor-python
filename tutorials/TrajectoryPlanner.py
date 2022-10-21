from time import time
from typing import Optional, Tuple

import numpy as np

from timor.Robot import RobotBase
from timor.task import Constraints, Goals
from timor.task.Task import Task
from timor.utilities.dtypes import Trajectory
import timor.utilities.logging as logging

class TrajectoryPlanner:
    """
    Takes a task to solve and a robot and returns a motion trajectory.
    """

    timeout: float  # Global timeout in seconds, maximum time spent for solving a task
    maximum_speed: float  # For all joints in the trajectory
    trajectory_resolution: float  # Time step size in seconds
    trajectory_acceleration_time: float

    def __init__(self,
                 task: Task,
                 robot: Optional[RobotBase] = None,
                 trajectory_resolution: float = .01,
                 trajectory_acceleration_time: float = 2.5,
                 timeout: float = float('inf'),
                 time_limit_per_goal: float = 60.):
        """The Trajectory planner is a naive approach for finding trajectories to solve goals

        :param timeout: Maximum time for all goals
        :param time_limit_per_goal: time in seconds allowed to solve each goal
        """
        self._task: Task = task
        self._robot: Optional[RobotBase] = robot
        self._evaluated: bool = False  # Indicator whether current task has been evaluated using current robot
        self.timeout: float = timeout
        self.trajectory_resolution = trajectory_resolution
        self.trajectory_acceleration_time = trajectory_acceleration_time
        self.maximum_speed = float('inf')
        self.time_limit_per_goal = time_limit_per_goal
    
    def solve(self, robot: RobotBase = None) -> Optional[Trajectory]:
        """
        Compute a trajectory that solves the task with a given robot if possible.

        :param robot: If given, sets the robot of the planner
        :return: A trajectory if found or None if not
        :raise ValueError: If solution cannot be found at specific point
        :raise NotImplementedError: If goal type is not yet solvable
        """
        if robot is not None:
            self.robot = robot
        elif self.robot is None:
            raise ValueError("There is no robot assigned to this planner.")

        if self._evaluated:
            raise ValueError(f"Task {self.task} has already been evaluated using robot {self.robot}.")
        self._set_base()
        solution = self._solve()
        self._evaluated = True
        return solution
    
    @property
    def task(self) -> Task:
        """Keep the task private, as it should not be changed."""
        return self._task

    @property
    def robot(self) -> Optional[RobotBase]:
        """Default getter"""
        return self._robot

    @robot.setter
    def robot(self, new_robot: RobotBase):
        """Sets the maximum velocity in addition to the robot"""
        self.maximum_speed = np.min(new_robot.joint_velocity_limits)
        self._evaluated = False
        self._robot = new_robot
        

    def _set_base(self):
        """Don't optimize, but just take the center of the possible space the base can be placed into"""
        if isinstance(self.task.base_constraint, Constraints.BasePlacement):  # No optimization at all
            self.robot.set_base_placement(self.task.base_constraint.base_pose.nominal)

    def _solve(self) -> Optional[Trajectory]:
        """
        Solves a task by solving the goals one by one.

        :return: Trajectory that solves the task or None if it could not be found
        :raise ValueError: if IK solution / planner fails
        """
        t0 = time()

        goal_order: Tuple[str, ...]
        goal_order_constraint = [x for x in self.task.constraints if isinstance(x, Constraints.GoalOrderConstraint)]
        if len(goal_order_constraint) > 1:
            raise ValueError("Found multiple goal orders")
        if goal_order_constraint:
            goal_order = goal_order_constraint[0].order
        else:
            goal_order = tuple(self.task.goals_by_id.keys())

        q_start = np.expand_dims(self.robot.q, 0)
        q_previous = q_start
        goal_traj = {-1: q_start}
        for goal_id in goal_order:  # Find places where goals are fulfilled
            goal = self.task.goals_by_id[goal_id]
            if not isinstance(goal, Goals.At):
                raise NotImplementedError(f"This planner cannot solve goal of type {type(goal)}")
            time_limit_per_goal = min(self.timeout - (time() - t0), self.time_limit_per_goal)
            q_previous = self._solve_reach_goal(goal_id, q_previous, time_limit_per_goal)
            goal_traj[goal_id] = q_previous

        trajectory = Trajectory(t=np.asarray((0,)), q=q_start)

        for goal_id in goal_order:
            q_previous = np.expand_dims(trajectory.q[-1], 0)
            goal = self.task.goals_by_id[goal_id]
            if isinstance(goal, Goals.Pause):
                trajectory = trajectory + self._solve_pause_goal(goal_id, q_previous)
                continue
            dq_end = np.zeros(self.robot.njoints,)

            # Only add a new trajectory if you are not already pretty much at the goal
            if np.linalg.norm(q_previous[-1, :] - goal_traj[goal_id][0, :]) > 1e-4:
                traj = Trajectory.from_mstraj(via_points=np.vstack((trajectory.q[-1, :], goal_traj[goal_id])),
                                              dt=self.trajectory_resolution,
                                              t_acc=self.trajectory_acceleration_time,
                                              qd_max=self.maximum_speed,
                                              qdf=dq_end
                                              )
            else:  # We assume a linear motion to the new goal works if the distance is that small
                traj = Trajectory.from_mstraj(via_points=np.vstack((trajectory.q[-1, :], goal_traj[goal_id])),
                                              dt=self.trajectory_resolution,
                                              t_acc=self.trajectory_acceleration_time,
                                              qd_max=self.maximum_speed,
                                              qdf=dq_end)

            if type(self.task.goals_by_id[goal_id]) is Goals.Reach:
                traj = traj + Trajectory.stationary(traj.q[-1, :], 0.1)
            traj.goals[goal_id] = traj.t[-1]

            trajectory = trajectory + traj
        
        return trajectory

    def _solve_reach_goal(self,
                          goal_id: str,
                          q_previous: np.array,
                          time_limit: float = None) -> np.ndarray:
        """
        Solves a reach and (inefficiently) At goal.

        Returns a nxnDoF path to go to next goal; PTP movements between path's via points should be collision free
        (at least with the OMPL planner)

        :param goal_id: At or Reach goal to plan a path to
        :param q_previous: robot configuration to start planning the path to the goal from
        :param time_limit: custom time limit for this goal
        :return: path with n steps for each of robot.DoF's joints
        """
        goal: Goals.At = self.task.goals_by_id[goal_id]
        if time_limit is None:
            time_limit = self.time_limit_per_goal
        t_start = time()
        q_init = q_previous[-1]
        while True:
            q, success = self.robot.ik(goal.goal_pose, q_init)
            if success:
                break
            q_init = self.robot.random_configuration()
            if time() - t_start > time_limit:
                raise TimeoutError(f"IK not found for {str(goal)}.")
        # TODO : Check constraints at q and if necessary try to find valid ik

        return np.expand_dims(q, 0)