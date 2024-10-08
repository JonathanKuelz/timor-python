#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.02.22
import unittest

import numpy as np
import numpy.testing as np_test
import pinocchio as pin

from timor import ModuleAssembly, Robot
from timor.task import Constraints, CostFunctions, Goals, Solution, Task, Tolerance
from timor.task.CostFunctions import QDist
from timor.task.Solution import SolutionHeader, SolutionTrajectory
from timor.task.Task import TaskHeader
from timor.utilities import prebuilt_robots, visualization
from timor.utilities.file_locations import get_test_tasks, robots
from timor.utilities.frames import Frame, WORLD_FRAME
from timor.utilities.prebuilt_robots import get_six_axis_assembly
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.trajectory import Trajectory
from timor.utilities.transformation import Transformation
from timor.utilities.visualization import clear_visualizer


class TestSolution(unittest.TestCase):
    """Test everything related to solutions here: are they valid, can we correctly identify valid solutions, ..."""

    def setUp(self) -> None:
        self.task_header = Task.TaskHeader('test_case', 'py', tags=['debug', 'test'])
        self.solution_header = Solution.SolutionHeader('test_case', author=['Jonathan'])
        self.empty_task = Task.Task(self.task_header)

        self.panda_urdf = robots['panda'].joinpath('urdf').joinpath('panda.urdf')
        self.robot = Robot.PinRobot.from_urdf(self.panda_urdf, robots['panda'].parent)

        file_locations = get_test_tasks()
        self.task_files = file_locations["task_files"]
        self.task_dir = file_locations["task_dir"]
        self.asset_dir = file_locations["asset_dir"]

        self.rng = np.random.default_rng(123)
        self.random_configurations = np.asarray([self.robot.random_configuration(self.rng) for _ in range(200)])

    def test_cost_parsing(self):
        """
        Tests parsing costs from strings.
        """
        abbreviations = ('cyc', 'mechEnergy', 'qDist', 'effort')

        for _ in range(100):
            # Generate random descriptor string
            num_costs = self.rng.integers(1, len(abbreviations))
            num_weights = self.rng.integers(1, len(abbreviations))
            cost_string = ''
            for i in range(num_costs):
                if i >= num_weights:
                    weight = str(float(self.rng.random()))
                    cost_string += self.rng.choice(abbreviations) + '_' + weight
                else:
                    cost_string += self.rng.choice(abbreviations)
                if i < num_costs - 1:
                    cost_string += ','

            c_1 = CostFunctions.CostFunctionBase.from_descriptor(cost_string)
            c_2 = CostFunctions.ComposedCost(
                *tuple(CostFunctions.CostFunctionBase.from_descriptor(single) for single in cost_string.split(','))
            )

            if num_costs > 1:
                self.assertIs(type(c_1), CostFunctions.ComposedCost)
                costs = c_1._internal
            else:
                costs = [c_1]

            self.assertEqual(len(costs), len(c_2._internal))
            for internal_1, internal_2 in zip(
                    sorted(costs, key=lambda x: x.__class__.__name__),
                    sorted(c_2._internal, key=lambda x: x.__class__.__name__)
            ):
                self.assertEqual(internal_1.weight, internal_2.weight)
                self.assertIs(type(internal_1), type(internal_2))

            # Test serialize
            c1_string = c_1.descriptor()
            c2_string = c_2.descriptor()
            self.assertEqual(c1_string, c2_string)
            self.assertEqual(c_1, CostFunctions.CostFunctionBase.from_descriptor(c1_string))
            self.assertEqual(c_2, CostFunctions.CostFunctionBase.from_descriptor(c1_string))
            self.assertEqual(c_2, CostFunctions.CostFunctionBase.from_descriptor(c2_string))

        with self.assertRaises(KeyError):
            CostFunctions.CostFunctionBase.from_descriptor('not_a_cost')

    def test_cost_de_serialization(self):
        """
        Test serialization and deserialization of costs
        """
        for abbrev, cost_class in CostFunctions.abbreviations.items():
            c = cost_class()
            c_from_descriptor = CostFunctions.CostFunctionBase.from_descriptor(c.descriptor())
            self.assertEqual(c, c_from_descriptor)
            parts = c.descriptor().split('_')
            self.assertEqual(parts[0], abbrev)
            if len(parts) > 1:
                self.assertEqual(float(parts[1]), c.weight)
            self.assertEqual(c_from_descriptor.descriptor(), c.descriptor())

    def test_goals(self):
        task = Task.Task({'ID': 'dummy'})
        cost_function = CostFunctions.GoalsFulfilled()
        robot = prebuilt_robots.get_six_axis_modrob()

        pause_duration = .4
        dt = 0.01
        time_steps = 100
        q_array_shape = (time_steps, robot.dof)
        # Add constraint such that _valid can be tested as well
        pause_goal = Goals.Pause(ID='pause', duration=pause_duration,
                                 constraints=[Constraints.JointAngles(robot.joint_limits)])

        for _ in range(10):
            bad_trajectory = Trajectory(t=dt, q=self.rng.random(q_array_shape))
            t_start = self.rng.choice(list(bad_trajectory.t[bad_trajectory.t < max(bad_trajectory.t) - pause_duration]))
            t_end = t_start + pause_duration
            q_good = bad_trajectory.q.copy()

            # Manually ensure there is a pause
            pause_mask = np.logical_and(t_start <= bad_trajectory.t, bad_trajectory.t <= t_end)
            q_good[pause_mask] = self.rng.random()
            good_trajectory = Trajectory(t=dt, q=q_good)
            good_trajectory.dq[pause_mask] = 0
            good_trajectory.ddq[pause_mask] = 0

            bad_trajectory.goal2time = {pause_goal.id: t_end}
            good_trajectory.goal2time = {pause_goal.id: t_end}

            valid = Solution.SolutionTrajectory(good_trajectory, {'taskID': task.id}, task=task,
                                                assembly=ModuleAssembly.from_monolithic_robot(robot),
                                                cost_function=cost_function)
            invalid = Solution.SolutionTrajectory(bad_trajectory, {'taskID': task.id}, task=task,
                                                  assembly=ModuleAssembly.from_monolithic_robot(robot),
                                                  cost_function=cost_function)

            self.assertTrue(pause_goal.achieved(valid))
            self.assertFalse(pause_goal.achieved(invalid))
            pause_goal.constraints.append(Constraints.AlwaysTrueConstraint())  # should not change anything
            self.assertTrue(pause_goal.achieved(valid))
            self.assertFalse(pause_goal.achieved(invalid))

            good_trajectory.dq = self.rng.random(q_array_shape)
            self.assertFalse(pause_goal.achieved(valid))

            bad_trajectory_to_short = Trajectory.stationary(q=np.zeros(q_array_shape[1]),
                                                            time=0.9 * pause_goal.duration)
            bad_trajectory_to_short.goal2time = {pause_goal.id: bad_trajectory_to_short.t[-1]}
            invalid_too_short = Solution.SolutionTrajectory(bad_trajectory_to_short, {'taskID': task.id},
                                                            task=task, cost_function=cost_function,
                                                            assembly=ModuleAssembly.from_monolithic_robot(robot))
            self.assertFalse(pause_goal.achieved(invalid_too_short))

            # Test that times in goal duration rightly calculated
            sol_times, sol_times_indices = pause_goal.get_time_range_goal(valid)
            self.assertListEqual(np.where(pause_mask)[0].tolist(), list(sol_times_indices))
            np_test.assert_array_less(t_start - dt, np.asarray(sol_times))
            np_test.assert_array_less(np.asarray(sol_times), t_end + dt)

            # Test constraint in duration - replace a single time-step inside pause with joint limit + a_bit_over_0
            q_bad_outside_joint_limits = good_trajectory.q.copy()
            q_bad_outside_joint_limits[self.rng.choice(np.argwhere(pause_mask).squeeze())] = \
                robot.joint_limits[1, :] + self.rng.random(robot.dof)
            bad_trajectory_outside_joint_limits = Trajectory(t=dt, q=q_bad_outside_joint_limits,
                                                             goal2time=good_trajectory.goal2time)
            bad_sol_outside_joint_limits = \
                Solution.SolutionTrajectory(bad_trajectory_outside_joint_limits, {'taskID': task.id}, task=task,
                                            assembly=ModuleAssembly.from_monolithic_robot(robot),
                                            cost_function=cost_function)
            self.assertFalse(pause_goal._valid(bad_sol_outside_joint_limits, 0., 0))  # time / index ignored anyway

    def test_instantiation(self):
        """Instantiate Solutions, Goals, Constraints"""
        # Set up a couple of realistic constraints
        constraint_joints = Constraints.JointAngles(self.robot.joint_limits)
        pos_tolerance = Tolerance.CartesianXYZ(*[[-.1, .1]] * 3)
        rot_tolerance = Tolerance.RotationAbsolute.default()
        base_tolerance = pos_tolerance + rot_tolerance
        constraint_base = Constraints.BasePlacement(ToleratedPose(WORLD_FRAME, base_tolerance))
        all_goals = Constraints.AllGoalsFulfilled()

        # Set up two goals
        at_goal = Goals.At('At pos z',
                           ToleratedPose(Frame("", Transformation.from_translation([.1, .1, .5]), WORLD_FRAME)),
                           constraints=[constraint_joints])
        reach_goal = Goals.Reach('Reach z',
                                 ToleratedPose(Frame("", Transformation.from_translation([.1, .1, .5]), WORLD_FRAME)),
                                 constraints=[constraint_joints])

        # Combine the goals in a new task (without obstacles for now)
        task = Task.Task(self.task_header, goals=[at_goal, reach_goal], constraints=[constraint_base, all_goals])
        task_without_all_goals = Task.Task(self.task_header, goals=[at_goal, reach_goal], constraints=[constraint_base])

        # This random trajectory certainly will not be valid - nevertheless, times at which goals are supposedly
        #  achieved must be stated
        random_trajectory = Trajectory(t=.01, q=self.random_configurations, goal2time={'At pos z': .58, 'Reach z': .65})

        # A cost function that reassembles Whitman2020
        cost_whitman = CostFunctions.NumJoints() + CostFunctions.RobotMass() - CostFunctions.GoalsFulfilled()
        mass_cost = CostFunctions.RobotMass()
        joint_cost = CostFunctions.NumJoints()
        composed_cost = CostFunctions.RobotMass() + CostFunctions.NumJoints() - CostFunctions.GoalsFulfilled()
        composed_cost = composed_cost * 2
        composed_cost = composed_cost / 5

        # So here is our (invalid) solution
        random_solution = Solution.SolutionTrajectory(random_trajectory,
                                                      self.solution_header,
                                                      task,
                                                      ModuleAssembly.from_monolithic_robot(self.robot),
                                                      cost_whitman)

        random_solution.plot(show_figure=False)
        random_solution.plot(["q", "dq", "ddq", "tau", "manipulablility", "eef_position", "eef_rotation"],
                             show_figure=False)
        random_solution.plot(["q", "dq", "ddq", "tau", "manipulablility", "eef_position", "eef_rotation"],
                             show_figure=False, subplot_kwargs={"facecolor": "black"})  # Check propagation of kwargs

        random_solution_without_all_goals = Solution.SolutionTrajectory(
            random_trajectory, self.solution_header, task_without_all_goals,
            ModuleAssembly.from_monolithic_robot(self.robot), cost_whitman)

        # Make sure lazy variables are not evaluated without explicitly calling them
        self.assertIsNone(random_solution._valid.value)
        self.assertIsNone(random_solution._cost.value)

        # Check whether the trajectory solves the solution
        self.assertFalse(random_solution.valid)
        self.assertTrue(random_solution_without_all_goals.valid)  # Should be fine even if one goal missed
        self.assertTrue(random_solution.cost > 0)

        # Lazy variables should be evaluated now
        self.assertIsNotNone(random_solution._valid.value)
        self.assertIsNotNone(random_solution._cost.value)

        # Test math operators on cost classes (composed cost evaluation should be equal to manually calculating
        # a more complex cost function)
        eval_composed_cost = list()
        for cst in (mass_cost, joint_cost):
            random_solution._cost.value = None  # Reset the cost calculation manually
            random_solution.cost_function = cst
            eval_composed_cost.append(random_solution.cost)
        random_solution._cost.value = None
        random_solution.cost_function = composed_cost
        self.assertAlmostEqual((eval_composed_cost[0] + eval_composed_cost[1]) * 2 / 5,
                               random_solution.cost)

    def test_follow_goal(self):
        """Test cases for follow goal."""
        robot_assembly = get_six_axis_assembly()
        qs = []  # 4 self-collision free robot configurations
        for _ in range(4):
            q = robot_assembly.robot.random_configuration(self.rng)
            while robot_assembly.robot.has_self_collision(q):
                q = robot_assembly.robot.random_configuration(self.rng)
            qs.append(q)
        pose_trajectory = Trajectory(pose=np.asarray(tuple(ToleratedPose(robot_assembly.robot.fk(q)) for q in qs)))
        timed_pose_trajectory = Trajectory(pose=pose_trajectory.pose, t=np.asarray((0., .1, .2, .3)))

        for traj in (pose_trajectory, timed_pose_trajectory):
            follow_goal = Goals.Follow("0", traj)
            task = Task.Task(TaskHeader("tmp", timeStepSize=0.1), goals=(follow_goal,))
            valid_sol = SolutionTrajectory(Trajectory(t=np.asarray((0., .1, .2, .3)), q=np.asarray(qs),
                                                      goal2time={"0": 0.3}),
                                           SolutionHeader("tmp"), task, robot_assembly,
                                           QDist(), Transformation.neutral())
            self.assertTrue(follow_goal.achieved(valid_sol))
            self.assertEqual(follow_goal._duration, 0.3)
            self.assertIsNotNone(valid_sol.tau)

            # Still ok with additional qs
            valid_sol = SolutionTrajectory(
                Trajectory(t=np.asarray((0., .1, .15, .2, .3)),
                           q=np.asarray((*qs[:2], robot_assembly.robot.random_configuration(self.rng), *qs[2:])),
                           goal2time={"0": 0.3}),
                SolutionHeader("tmp"), task, robot_assembly, QDist(), Transformation.neutral())
            self.assertTrue(follow_goal.achieved(valid_sol))
            self.assertEqual(follow_goal._duration, 0.3)

            # Break solution by removing point
            valid_sol = SolutionTrajectory(
                Trajectory(t=np.asarray((0., .1, .3)),
                           q=np.asarray((*qs[:2], *qs[3:])),
                           goal2time={"0": 0.3}),
                SolutionHeader("tmp"), task, robot_assembly, QDist(), Transformation.neutral())
            self.assertFalse(follow_goal.achieved(valid_sol))
            if traj.is_timed:
                self.assertEqual(follow_goal._duration, 0.3)
            else:
                self.assertEqual(follow_goal._duration, None)

            # Break by changing timing
            valid_sol = SolutionTrajectory(
                Trajectory(t=np.asarray((0., .1, .15, .23, .4)),
                           q=np.asarray((*qs[:2], robot_assembly.robot.random_configuration(self.rng), *qs[2:])),
                           goal2time={"0": 0.4}),
                SolutionHeader("tmp"), task, robot_assembly, QDist(), Transformation.neutral())
            if traj.is_timed:
                self.assertFalse(follow_goal.achieved(valid_sol))
                self.assertEqual(follow_goal._duration, 0.3)
            else:
                self.assertTrue(follow_goal.achieved(valid_sol))
                self.assertEqual(follow_goal._duration, 0.4)  # Changed timing -> other duration

            # Break altering q
            valid_sol = SolutionTrajectory(
                Trajectory(t=np.asarray((0., .1, .2, .3)),
                           q=np.asarray((*qs[:2], robot_assembly.robot.random_configuration(self.rng), *qs[3:])),
                           goal2time={"0": 0.3}),
                SolutionHeader("tmp"), task, robot_assembly, QDist(), Transformation.neutral())
            self.assertFalse(follow_goal.achieved(valid_sol))
            if traj.is_timed:
                self.assertEqual(follow_goal._duration, 0.3)
            else:
                self.assertEqual(follow_goal._duration, None)

        # Test constraints on whole duration with adding / removing self-collision during execution
        while not robot_assembly.robot.has_self_collision():
            robot_assembly.robot.update_configuration(robot_assembly.robot.random_configuration(self.rng))
        q_self_collision = robot_assembly.robot.configuration
        follow_goal_with_constraint = Goals.Follow("0", traj, constraints=(Constraints.SelfCollisionFree(),))
        task = Task.Task(TaskHeader("tmp", timeStepSize=0.1), goals=(follow_goal_with_constraint,))
        valid_sol = SolutionTrajectory(Trajectory(t=np.asarray((0., .1, .2, .3)),
                                                  q=np.asarray(qs), goal2time={"0": 0.3}),
                                       SolutionHeader("tmp"), task, robot_assembly,
                                       QDist(), Transformation.neutral())
        self.assertTrue(follow_goal_with_constraint.achieved(valid_sol))
        self.assertTrue(follow_goal_with_constraint._valid(
            valid_sol, valid_sol.t_goals["0"], len(valid_sol.trajectory)))

        # Adding a random, non-colliding should be fine without time
        q_rand_coll_free = robot_assembly.robot.random_configuration(self.rng)
        while robot_assembly.robot.has_self_collision(q_rand_coll_free):
            q_rand_coll_free = robot_assembly.robot.random_configuration(self.rng)
        validish_sol = SolutionTrajectory(
            Trajectory(t=np.asarray((0., .1, .2, .3, .4)),
                       q=np.asarray((*qs[:2], q_rand_coll_free, *qs[2:])), goal2time={"0": 0.4}),
            SolutionHeader("tmp"), task, robot_assembly, QDist(), Transformation.neutral())
        if traj.is_timed:
            self.assertFalse(follow_goal_with_constraint.achieved(validish_sol))
        else:
            self.assertTrue(follow_goal_with_constraint.achieved(validish_sol))

        # Constraint fulfilled in both cases
        self.assertTrue(follow_goal_with_constraint._valid(
            validish_sol, validish_sol.t_goals["0"], len(validish_sol.trajectory)))

        # Invalid by adding self colliding during trajectory
        invalid_sol = SolutionTrajectory(
            Trajectory(t=np.asarray((0., .1, .2, .3, .4)),
                       q=np.asarray((*qs[:2], q_self_collision, *qs[2:])), goal2time={"0": 0.4}),
            SolutionHeader("tmp"), task, robot_assembly, QDist(), Transformation.neutral())
        self.assertFalse(follow_goal_with_constraint.achieved(invalid_sol))
        self.assertFalse(follow_goal_with_constraint._valid(
            invalid_sol, invalid_sol.t_goals["0"], len(invalid_sol.trajectory)))

    def test_follow_force_torque_constraint(self):
        assembly = get_six_axis_assembly()
        robot = assembly.robot
        robot.model.effortLimit = 10 * np.ones(robot.dof)  # high torque limits
        len_desired = 15
        len_additional = 5
        dt = .1
        q = []
        while len(q) < len_desired + len_additional:
            new = q[-1] + self.rng.random((robot.dof,)) * 0.01 if len(q) > 0 \
                else robot.random_configuration(self.rng)
            if not robot.has_self_collision(new) \
                    and robot.tau_in_torque_limits(robot.static_torque(new)) \
                    and robot.q_in_joint_limits(new):
                q.append(new)

        desired = Trajectory(pose=np.asarray(tuple(ToleratedPose(robot.fk(qi)) for qi in q[:len_desired])), t=dt)
        follow_goal_with_constraint = Goals.Follow("0", desired,
                                                   constraints=(Constraints.JointLimits(parts=('q',)),))
        task = Task.Task(TaskHeader("tmp", timeStepSize=dt),
                         goals=(follow_goal_with_constraint,),
                         constraints=(Constraints.JointLimits(parts=('q', 'tau')),
                                      Constraints.AllGoalsFulfilled(),
                                      Constraints.CollisionFree()
                                      )
                         )

        trajectory = Trajectory(q=np.asarray(q), t=dt, goal2time={"0": (len_desired - 1) * dt})
        solution = SolutionTrajectory(trajectory, {'taskID': task.id}, task, assembly, QDist())

        # Thus far, the solution should be valid (assuming the robot is not moving too fast yet)
        self.assertTrue(solution.valid)

        # Add high external force
        follow_goal_with_constraint._external_forces = self.rng.random((3,)) * 1000
        solution = SolutionTrajectory(trajectory, {'taskID': task.id}, task, assembly, QDist())
        self.assertFalse(solution.valid)
        follow_goal_with_constraint._external_forces = np.zeros((3,))  # reset

        # Add high external torques
        follow_goal_with_constraint._external_torques = self.rng.random((3,)) * 1000
        solution = SolutionTrajectory(trajectory, {'taskID': task.id}, task, assembly, QDist())
        self.assertFalse(solution.valid)

        # Remove the torque constraint
        task.constraints = (Constraints.AllGoalsFulfilled(), Constraints.CollisionFree())
        solution = SolutionTrajectory(trajectory, {'taskID': task.id}, task, assembly, QDist())
        self.assertTrue(solution.valid)

    def test_solution_derived_properties(self, num_test=10, epsilon=1e-12):
        assembly = get_six_axis_assembly()  # Use IMPROV with significant joint inertia and friction
        for _ in range(num_test):
            qs = np.asarray((assembly.robot.random_configuration(), assembly.robot.random_configuration()))
            sol = SolutionTrajectory(Trajectory.from_mstraj(qs, 0.01, 1., 1.), SolutionHeader("tmp"),
                                     Task.Task(TaskHeader("tmp")), assembly, CostFunctions.RobotMass(),
                                     assembly.robot.placement)

            np_test.assert_array_less(np.abs(sol.link_side_torques), np.abs(sol.torques) + epsilon)
            np_test.assert_array_less(sol._get_torques(False, True), sol.torques + epsilon)
            np_test.assert_array_less(sol._get_torques(True, False), sol.torques + epsilon)
            np_test.assert_array_less(sol.get_power(False, False), sol.get_power() + epsilon)
            self.assertLess(sol.get_mechanical_energy(False, False), sol.get_mechanical_energy() + epsilon)

    def test_task_visuals(self):
        """Needs manual inspection for the plots"""
        robot = Robot.PinRobot.from_urdf(self.panda_urdf, robots['panda'].parent)
        robot.move(Transformation.from_translation((-2., -2., .5)))
        q0 = pin.neutral(robot.model)
        q1 = pin.randomConfiguration(robot.model)
        q2 = pin.randomConfiguration(robot.model)
        viz = None
        color_viz = None

        empty_task = Task.Task.empty()
        empty_solution = Solution.SolutionTrajectory.empty()
        empty_viz = empty_task.visualize()
        empty_viz = empty_task.visualize(empty_viz, robots=robot)
        empty_solution.visualize(empty_viz)

        for description in self.task_files:
            robot.update_configuration(q0)
            task = Task.Task.from_json_file(description, self.asset_dir)
            robot.update_configuration(q1)
            viz = task.visualize(viz, robots=robot, center_view=False)  # Should have view centered on origin
            clear_visualizer(viz)
            viz = task.visualize(viz, robots=robot, center_view=True)  # Should have a view including more of the robot
            clear_visualizer(viz)
            viz = task.visualize(viz, center_view=True)  # Should have a view including base constraint, e.g. PTP_1
            clear_visualizer(viz)
            robot.update_configuration(q2)
            self.assertIs(robot.data, viz.data)

            assembly = get_six_axis_assembly()
            assembly.robot.update_configuration(assembly.robot.random_configuration())
            color_viz = task.visualize(color_viz, robots=assembly.robot)
            visualization.color_visualization(color_viz, assembly)

            id_cmap = {mid: np.random.rand(4) for mid in assembly.original_module_ids}
            visualization.color_visualization(color_viz, assembly, color_map=id_cmap)

            custom_id_cmap = {m.id: np.random.rand(4) for m in assembly.module_instances}
            visualization.color_visualization(color_viz, assembly, color_map=custom_id_cmap)


if __name__ == '__main__':
    unittest.main()
