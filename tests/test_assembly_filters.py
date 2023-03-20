import logging
import random
import unittest

import numpy as np

from timor import Transformation
from timor.Geometry import Box
from timor.Module import ModuleAssembly, ModulesDB
from timor.Robot import PinRobot
from timor.configuration_search import AssemblyFilter
from timor.task import Goals, Task, Tolerance, Obstacle
from timor.utilities.trajectory import Trajectory
from timor.utilities.file_locations import get_test_tasks, robots
from timor.utilities.prebuilt_robots import get_six_axis_assembly
from timor.utilities.tolerated_pose import ToleratedPose


class TestFilterRobotsForTasks(unittest.TestCase):
    """Test all robot filters."""
    def setUp(self) -> None:
        random.seed(99)
        np.random.seed(99)
        panda_urdf = robots['panda'].joinpath('urdf').joinpath('panda.urdf')
        self.panda_assembly = \
            ModuleAssembly.from_monolithic_robot(PinRobot.from_urdf(panda_urdf, robots['panda'].parent))
        self.modrob_assembly = get_six_axis_assembly()
        self.panda_fk = tuple(self.panda_assembly.robot.fk(q)
                              for q in (self.panda_assembly.robot.random_configuration() for _ in range(30))
                              if not self.panda_assembly.robot.has_self_collision(q)
                              )

        gen2_db = ModulesDB.from_name('modrob-gen2')
        self.zero_joint_assembly = ModuleAssembly(gen2_db, ['105'])  # This is only a base

        test_data = get_test_tasks()
        self.task_files = test_data["task_files"]
        self.asset_dir = test_data["asset_dir"]

    def test_ik_filter(self):
        test_filter = AssemblyFilter.InverseKinematicsSolvable(max_n_tries=5)
        modrob_solved = 0
        num_tests = 20
        for test_task in range(num_tests):
            num_goals = random.randint(1, 5)
            logging.debug(f"Test goal with {num_goals} goals")
            # Relaxed tolerance to faster compute an ik
            position_tolerance = Tolerance.CartesianXYZ(x=(-.1, .1), y=(-.1, .1), z=(-.1, .1))
            goals = tuple(Goals.At(ID=str(i),
                                   goalPose=ToleratedPose(random.choice(self.panda_fk), position_tolerance))
                          for i in range(num_goals))
            header = Task.TaskHeader(ID=str(test_task))

            solves_all = Task.Task(header, goals=goals)
            solves_some = Task.Task(header, goals=goals)
            solves_none = Task.Task(header, goals=goals)

            intermediate_result = AssemblyFilter.IntermediateFilterResults(robot=self.panda_assembly.robot)
            solves = test_filter.check(self.panda_assembly, solves_all, intermediate_result)
            self.assertTrue(solves)
            self.assertEqual(len(intermediate_result.q_goal_pose), num_goals)
            self.assertFalse(test_filter.check(self.zero_joint_assembly, solves_none,
                                               AssemblyFilter.IntermediateFilterResults(
                                                   robot=self.zero_joint_assembly.robot)))

            solves = test_filter.check(self.modrob_assembly, solves_some,
                                       AssemblyFilter.IntermediateFilterResults(robot=self.modrob_assembly.robot))
            if solves:
                logging.debug("Was successfully solved")
                modrob_solved += 1

        self.assertLessEqual(modrob_solved, num_tests)  # Can at most solve all tests
        self.assertLess(0, modrob_solved)  # Expect at least one successful ik

    def test_InverseKinematicSolvableWithObstacle(self, n_iter: int = 100):
        obstacle = Obstacle.Obstacle("tmp", collision=Box(parameters={"x": 2., "y": 2., "z": 2.},
                                     pose=Transformation.from_translation((1.5, 0., 1.))))
        for _ in range(n_iter):
            q_sample = self.panda_assembly.robot.random_configuration()
            task = Task.Task(Task.TaskHeader("tmp"), obstacles=(obstacle, ),
                             goals=(Goals.At("1", ToleratedPose(self.panda_assembly.robot.fk(q_sample))), ))
            filter = AssemblyFilter.InverseKinematicsSolvable(task=task, q_init=q_sample)  # Just check collision free
            self.assertTrue(any(p is AssemblyFilter.ResultType.collision_free_goal for p in filter.provides))
            self.panda_assembly.robot.update_configuration(q_sample)
            intermediate_result = AssemblyFilter.IntermediateFilterResults()
            intermediate_result.robot = self.panda_assembly.robot
            ik_found = filter.check(self.panda_assembly, task, intermediate_result)
            # ik is found (maybe at other q)
            if ik_found:
                self.panda_assembly.robot.update_configuration(intermediate_result.q_goal_pose["1"])
                self.assertFalse(self.panda_assembly.robot.has_collisions(task))
                self.assertTrue(intermediate_result.collision_free_goal["1"])
            else:  # or at least q_sample was in collision
                self.panda_assembly.robot.update_configuration(q_sample)
                self.assertTrue(self.panda_assembly.robot.has_collisions(task))
                self.assertFalse(intermediate_result.collision_free_goal["1"])

    def test_test_InverseKinematicSolvableWithTau(self, n_iter: int = 100):
        self.panda_assembly.robot.model.inertias[-1].mass *= 10  # Make eef heavier such that static fails
        for _ in range(n_iter):
            q_sample = self.panda_assembly.robot.random_configuration()
            goal_pose = ToleratedPose(self.panda_assembly.robot.fk(q_sample))
            task = Task.Task(Task.TaskHeader("tmp"), obstacles=(),
                             goals=(Goals.At("1", goal_pose),))
            filter = AssemblyFilter.InverseKinematicsSolvable(check_static_torques=True, q_init=q_sample)

            self.assertTrue(any(p is AssemblyFilter.ResultType.tau_static for p in filter.provides))
            self.panda_assembly.robot.update_configuration(q_sample)
            intermediate_result = AssemblyFilter.IntermediateFilterResults()
            intermediate_result.robot = self.panda_assembly.robot
            ik_found = filter.check(self.panda_assembly, task, intermediate_result)
            tau = self.panda_assembly.robot.static_torque(intermediate_result.q_goal_pose["1"])
            if intermediate_result.tau_static["1"]:
                self.assertTrue((np.abs(tau) <= self.panda_assembly.robot.joint_torque_limits).all())
            else:
                self.assertFalse(ik_found)
                # Fail due to force or ik broken
                self.assertTrue(
                    not goal_pose.valid(self.panda_assembly.robot.fk(intermediate_result.q_goal_pose["1"]))
                    or (np.abs(tau) >= self.panda_assembly.robot.joint_torque_limits).any())

        self.panda_assembly.robot.model.inertias[-1].mass /= 10  # Reset

    def test_limits_met_filter(self):
        """Partially tests the filter by omitting the task and just plugging trajectories in"""
        test_filter = AssemblyFilter.JointLimitsMet()

        time_steps = 1000
        robot = self.modrob_assembly.robot
        dt = .1
        t_wait = Trajectory(t=dt, q=np.zeros((time_steps, robot.njoints)))
        q_edge_case = np.empty((time_steps, robot.njoints), dtype=float)
        dq_edge_case = np.empty((time_steps, robot.njoints), dtype=float)
        ddq_realistic = np.empty((time_steps, robot.njoints), dtype=float)
        acc_limit = dt * .2 * robot.joint_velocity_limits
        for i in range(time_steps):
            # Create a trajectory with random but valid accelerations
            sign = -1 if random.random() < .5 else 1
            factor = random.random() * sign * 0.5
            ddq = factor * acc_limit
            # Provisional integration
            dq = robot.dq + dt * ddq
            q = (robot.q + dt * dq + np.pi) % (2 * np.pi) - np.pi
            robot.update_configuration(q, dq, ddq)
            q_edge_case[i, :] = q
            dq_edge_case[i, :] = dq
            ddq_realistic[i, :] = ddq
        t_edge_case = Trajectory(t=dt, q=q_edge_case, dq=dq_edge_case, ddq=ddq_realistic)
        t_failing = Trajectory(t=dt, q=q_edge_case, dq=100 * dq_edge_case, ddq=100 * ddq_realistic)

        for trajectory, passes in ((t_wait, True), (t_edge_case, True), (t_failing, False)):
            intermediate_result = AssemblyFilter.IntermediateFilterResults(
                trajectory=AssemblyFilter.EternalResult({'some_goal_id': trajectory}),
                robot=robot)
            self.assertEqual(passes, test_filter._check(self.modrob_assembly, None, intermediate_result))


if __name__ == '__main__':
    unittest.main()
