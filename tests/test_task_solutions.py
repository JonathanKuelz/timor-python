#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.02.22
import math
import random
import unittest

import numpy as np
import numpy.testing as np_test
import pinocchio as pin

from timor import ModuleAssembly, Robot
from timor.task import Constraints, CostFunctions, Goals, Solution, Task, Tolerance
from timor.utilities import dtypes, prebuilt_robots, spatial
from timor.utilities.file_locations import get_test_tasks, robots
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.transformation import Transformation


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

        random.seed(123)
        np.random.seed(123)
        self.random_configurations = np.asarray([pin.randomConfiguration(self.robot.model) for _ in range(200)])

    def test_constraint_abstraction_level(self):
        panda = Robot.PinRobot.from_urdf(self.panda_urdf, robots['panda'].parent)

        limits = panda.joint_limits
        base_constraint = Constraints.BasePlacement(ToleratedPose(Transformation.neutral(),
                                                                  tolerance=Tolerance.DEFAULT_SPATIAL))
        basic_joint_constraint = Constraints.JointLimits(parts=('q',))
        additional_joint_constraint = Constraints.CoterminalJointAngles(limits * 0.95)

        start_conf = np.mean(limits * np.random.random(limits.shape), axis=0)  # Around middle of limits
        end_conf = np.mean(limits * np.random.random(limits.shape), axis=0)  # Around middle of limits

        goal_pose = ToleratedPose(panda.fk(end_conf))
        self.assertFalse(panda.has_self_collision(end_conf))
        at_goal = Goals.At('At_Goal', goal_pose, constraints=[additional_joint_constraint])

        task = Task.Task(self.task_header, goals=[at_goal],
                         constraints=[base_constraint, basic_joint_constraint, Constraints.AllGoalsFulfilled()])
        task_without_all_goals_fulfilled = Task.Task(self.task_header, goals=[at_goal],
                                                     constraints=[base_constraint, basic_joint_constraint])

        steps = 100
        dt = .1
        easy_trajectory = dtypes.Trajectory(dt, q=np.linspace(start_conf, end_conf, steps + 1),
                                            goals={at_goal.id: steps * dt})
        solution_header = Solution.SolutionHeader(task.id)
        cost_function = CostFunctions.CycleTime()
        solution = Solution.SolutionTrajectory(easy_trajectory, solution_header, task=task,
                                               assembly=ModuleAssembly.from_monolithic_robot(panda),
                                               cost_function=cost_function)
        solution_without_all_goals_fulfilled = Solution.SolutionTrajectory(
            easy_trajectory, solution_header, task=task_without_all_goals_fulfilled,
            assembly=ModuleAssembly.from_monolithic_robot(panda), cost_function=cost_function)

        self.assertTrue(solution.valid)
        solution._valid.value = None  # Dirty but quick reset of solution

        # Test that local constraints can fail a solution
        previous_limits = additional_joint_constraint.limits.copy()
        additional_joint_constraint.limits = np.zeros_like(additional_joint_constraint.limits)
        self.assertFalse(solution.valid)
        # Should be valid as not every goal needs to be valid
        self.assertTrue(solution_without_all_goals_fulfilled.valid)
        solution._valid.value = None  # Dirty but quick reset of solution

        # Reset to valid
        additional_joint_constraint.limits = previous_limits

        # Now hurt these local constraints, but outside the goal
        addon_trajectory = dtypes.Trajectory(dt, q=np.linspace(end_conf, panda.joint_limits[1, :], steps + 1))
        solution.trajectory = solution.trajectory + addon_trajectory
        self.assertTrue(solution.valid)
        solution._valid.value = None  # Dirty but quick reset of solution

        # But it fails if the same constraint is a global one
        task.constraints = tuple(list(task.constraints) + [additional_joint_constraint])
        self.assertFalse(solution.valid)

        for global_constraint in (Constraints.GoalOrderConstraint(['At_Goal']),
                                  base_constraint):
            with self.assertRaises(ValueError):
                Goals.At('At_Goal', goal_pose, constraints=[global_constraint])

    def test_cost_parsing(self):
        """
        Tests parsing costs from strings.
        """
        abbreviations = ('cyc', 'mechEnergy', 'qDist', 'effort')

        for _ in range(100):
            # Generate random descriptor string
            num_costs = random.randint(1, len(abbreviations))
            num_weights = random.randint(1, len(abbreviations))
            cost_string = ''
            for i in range(num_costs):
                if i >= num_weights:
                    weight = str(float(random.random()))
                    cost_string += random.choice(abbreviations) + '_' + weight
                else:
                    cost_string += random.choice(abbreviations)
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

    def test_eef_constraint(self):
        eef_constraint_allow_any_pose = Constraints.EndEffector(
            pose=ToleratedPose(nominal=Transformation.neutral(), tolerance=Tolerance.AlwaysValidTolerance()))
        qs = np.asarray([self.robot.random_configuration() for _ in range(10)])
        eef_constraint_allow_first_pose = Constraints.EndEffector(
            pose=ToleratedPose(self.robot.fk(qs[0, :]), tolerance=Tolerance.DEFAULT_SPATIAL)
        )

        # Check pose
        sol = Solution.SolutionTrajectory(dtypes.Trajectory(1., qs, {}), Solution.SolutionHeader("tmp"),
                                          Task.Task(Task.TaskHeader('tmp'), ),
                                          ModuleAssembly.from_monolithic_robot(self.robot), CostFunctions.RobotMass(),
                                          Transformation.neutral())
        self.assertTrue(eef_constraint_allow_first_pose.is_valid_until(sol, 0.))
        self.assertFalse(eef_constraint_allow_first_pose.is_valid_until(sol, 1.))
        self.assertFalse(eef_constraint_allow_first_pose.fulfilled(sol))
        self.assertTrue(eef_constraint_allow_any_pose.is_valid_until(sol, 0.))
        self.assertTrue(eef_constraint_allow_any_pose.is_valid_until(sol, 1.))
        self.assertTrue(eef_constraint_allow_any_pose.fulfilled(sol))

        # Constructed case - movement along line with invariant rotation about line, e.g. drilling
        orientation_tolerance = Tolerance.RotationAxisAngle(n_x=(-math.pi / 100, math.pi / 100),
                                                            n_y=(-math.pi / 100, math.pi / 100),
                                                            n_z=(-1, 1),  # Any rotation about local z-axis is ok
                                                            theta_r=(0, np.pi))
        ik_tolerance = Tolerance.Composed((orientation_tolerance, Tolerance.CartesianXYZ.default()))
        start_pose = Transformation.from_translation((0.5, 0., 0.5)) @ spatial.rotY(math.pi / 2)
        # Construct movement steps - q_start, q_end, q_mid_ok, q_mid_nok (small rot), q_end_to_far
        qs = []
        N_middle = 10  # How many random valid position inbetween start and end pose
        t_good = 2 + N_middle  # start, end + valid in the middle
        for T in (start_pose,  # q_start
                  Transformation.from_translation((.2, 0., 0.)) @ start_pose,  # q_end
                  # q_mid - all with random translation in 0, 0.2 and rotation about drill axis
                  *(Transformation.from_translation((random.random() * 0.2, 0., 0.)) @
                    start_pose @ spatial.rotZ(random.random()) for _ in range(N_middle)),
                  Transformation.from_translation((.11, 0., 0.)) @ start_pose @ spatial.rotX(0.02),  # q_mid_not_ok
                  Transformation.from_translation((.11, 0., 0.)) @ start_pose @ spatial.rotY(-0.02),  # q_mid_not_ok
                  Transformation.from_translation((.22, 0., 0.)) @ start_pose  # q_end_to_far
                  ):
            q, success = self.robot.ik(ToleratedPose(T, ik_tolerance))
            self.assertTrue(success, f"ik for {str(T)} failed")
            qs.append(q)
        drill_eef_constraint = Constraints.EndEffector(pose=ToleratedPose(
            nominal=start_pose,
            tolerance=Tolerance.Composed((
                Tolerance.CartesianXYZ((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.2)),  # Drill along z
                orientation_tolerance
            ))
        ))
        drill_test_traj = dtypes.Trajectory(1., np.asarray(qs), {})
        sol = Solution.SolutionTrajectory(drill_test_traj, Solution.SolutionHeader("tmp"),
                                          Task.Task(Task.TaskHeader("tmp")),
                                          ModuleAssembly.from_monolithic_robot(self.robot), CostFunctions.RobotMass(),
                                          Transformation.neutral())
        for t in np.arange(0, t_good, 1):
            self.assertTrue(drill_eef_constraint.is_valid_until(sol, t), f"Error at step {t}")
        for t in np.arange(t_good, t_good + 3, 1):
            self.assertFalse(drill_eef_constraint.is_valid_until(sol, t))

        # Check velocities
        self.robot.update_configuration(np.zeros((7,)))  # ~0.6 m from first joint to eef
        eef_constraint_v = Constraints.EndEffector(velocity_lim=np.asarray((.55, 0.65)))
        eef_constraint_o = Constraints.EndEffector(rotation_velocity_lim=np.asarray((.95, 1.05)))
        vel_test_traj = dtypes.Trajectory(np.asarray([0., 1., 2., 3., 4.]), np.zeros((5, 7)), {},
                                          dq=np.asarray(((0., 1., 0., 0., 0., 0., 0.),  # eef ~ 0.6 m/s
                                                         (0., .9, 0., 0., 0., 0., 0.),  # eef ~ 0.54 m/s - invalid
                                                         (0., 1.1, 0., 0., 0., 0., 0.),  # eef ~ 0.66 m/s - invalid
                                                         (0., -1., 0., 0., 0., 0., 0.),  # eef ~ 0.6 m/s
                                                         (0., -.85, 0., 0., 0., 0., 0.),  # eef ~ 0.5 m/s
                                                         )))
        sol = Solution.SolutionTrajectory(vel_test_traj, Solution.SolutionHeader("tmp"),
                                          Task.Task(Task.TaskHeader("tmp"), constraints=(eef_constraint_v,)),
                                          ModuleAssembly.from_monolithic_robot(self.robot), CostFunctions.RobotMass(),
                                          Transformation.neutral())
        for c in (eef_constraint_v, eef_constraint_o):
            self.assertTrue(c.is_valid_until(sol, 0.))
            self.assertTrue(c.is_valid_until(sol, 3.))
            self.assertFalse(c.is_valid_until(sol, 1.))
            self.assertFalse(c.is_valid_until(sol, 2.))
            self.assertFalse(c.is_valid_until(sol, 4.))
        for t in np.arange(0., 3., 1.):
            self.assertTrue(eef_constraint_allow_any_pose.is_valid_until(sol, t))
            self.assertFalse(eef_constraint_allow_first_pose.is_valid_until(sol, t))

    def test_goals(self):
        task = Task.Task({'ID': 'dummy'})
        cost_function = CostFunctions.GoalsFulfilled()
        robot = prebuilt_robots.get_six_axis_modrob()

        pause_duration = .4
        dt = 0.01
        time_steps = 100
        q_array_shape = (time_steps, robot.njoints)
        # Add constraint such that _valid can be tested as well
        pause_goal = Goals.Pause(ID='pause', duration=pause_duration,
                                 constraints=[Constraints.JointAngles(robot.joint_limits)])

        Solution.SolutionBase.t_resolution = float('inf')  # Monkey patch to being able testing invalid solutions
        for _ in range(10):
            bad_trajectory = dtypes.Trajectory(t=dt, q=np.random.random(q_array_shape))
            t_start = random.choice(tuple(bad_trajectory.t[bad_trajectory.t < max(bad_trajectory.t) - pause_duration]))
            t_end = t_start + pause_duration
            q_good = bad_trajectory.q.copy()

            # Manually ensure there is a pause
            pause_mask = np.logical_and(t_start <= bad_trajectory.t, bad_trajectory.t <= t_end)
            q_good[pause_mask] = random.random()
            good_trajectory = dtypes.Trajectory(t=dt, q=q_good)
            good_trajectory.dq[pause_mask] = 0
            good_trajectory.ddq[pause_mask] = 0

            bad_trajectory.goals = {pause_goal.id: t_end}
            good_trajectory.goals = {pause_goal.id: t_end}

            valid = Solution.SolutionTrajectory(good_trajectory, {'taskID': task.id}, task=task,
                                                assembly=ModuleAssembly.from_monolithic_robot(robot),
                                                cost_function=cost_function)
            invalid = Solution.SolutionTrajectory(bad_trajectory, {'taskID': task.id}, task=task,
                                                  assembly=ModuleAssembly.from_monolithic_robot(robot),
                                                  cost_function=cost_function)

            self.assertTrue(pause_goal.achieved(valid))
            self.assertFalse(pause_goal.achieved(invalid))
            good_trajectory.dq = np.random.random(q_array_shape)
            self.assertFalse(pause_goal.achieved(valid))

            bad_trajectory_to_short = dtypes.Trajectory.stationary(np.zeros(q_array_shape), 0.9 * pause_goal.duration)
            bad_trajectory_to_short.goals = {pause_goal.id: bad_trajectory_to_short.t[-1]}
            invalid_too_short = Solution.SolutionTrajectory(bad_trajectory_to_short, {'taskID': task.id},
                                                            task=task, cost_function=cost_function,
                                                            assembly=ModuleAssembly.from_monolithic_robot(robot))
            self.assertFalse(pause_goal.achieved(invalid_too_short))

            # Test that times in goal duration rightly calculated
            sol_times, sol_times_indices = pause_goal._get_time_range_goal(valid)
            self.assertListEqual(np.where(pause_mask)[0].tolist(), list(sol_times_indices))
            np_test.assert_array_less(t_start - dt, np.asarray(sol_times))
            np_test.assert_array_less(np.asarray(sol_times), t_end + dt)

            # Test constraint in duration - replace a single time-step inside pause with joint limit + a_bit_over_0
            q_bad_outside_joint_limits = good_trajectory.q.copy()
            q_bad_outside_joint_limits[np.random.choice(np.argwhere(pause_mask).squeeze())] = \
                robot.joint_limits[1, :] + np.random.random(robot.njoints)
            bad_trajectory_outside_joint_limits = dtypes.Trajectory(t=dt, q=q_bad_outside_joint_limits,
                                                                    goals=good_trajectory.goals)
            bad_sol_outside_joint_limits = \
                Solution.SolutionTrajectory(bad_trajectory_outside_joint_limits, {'taskID': task.id}, task=task,
                                            assembly=ModuleAssembly.from_monolithic_robot(robot),
                                            cost_function=cost_function)
            self.assertFalse(pause_goal._valid(bad_sol_outside_joint_limits))

    def test_instantiation(self):
        """Instantiate Solutions, Goals, Constraints"""
        # Set up a couple of realistic constraints
        constraint_joints = Constraints.JointAngles(self.robot.joint_limits)
        pos_tolerance = Tolerance.CartesianXYZ(*[[-.1, .1]] * 3)
        rot_tolerance = Tolerance.RotationAxisAngle.default()
        base_tolerance = pos_tolerance + rot_tolerance
        constraint_base = Constraints.BasePlacement(ToleratedPose(Transformation.neutral(), base_tolerance))
        all_goals = Constraints.AllGoalsFulfilled()

        # Set up two goals
        at_goal = Goals.At('At pos z', ToleratedPose(spatial.homogeneous([.1, .1, .5])),
                           constraints=[constraint_joints])
        reach_goal = Goals.Reach('Reach z', ToleratedPose(spatial.homogeneous([.1, .1, .5])),
                                 constraints=[constraint_joints])

        # Combine the goals in a new task (without obstacles for now)
        task = Task.Task(self.task_header, goals=[at_goal, reach_goal], constraints=[constraint_base, all_goals])
        task_without_all_goals = Task.Task(self.task_header, goals=[at_goal, reach_goal], constraints=[constraint_base])

        # This random trajectory certainly will not be valid - nevertheless, times at which goals are supposedly
        #  achieved must be stated
        random_trajectory = dtypes.Trajectory(.01, self.random_configurations, {'At pos z': .58, 'Reach z': .65})

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

    def test_task_visuals(self):
        """Needs manual inspection for the plots"""
        robot = Robot.PinRobot.from_urdf(self.panda_urdf, robots['panda'].parent)
        q0 = pin.neutral(robot.model)
        q1 = pin.randomConfiguration(robot.model)
        q2 = pin.randomConfiguration(robot.model)
        for description in self.task_files:
            robot.update_configuration(q0)
            task = Task.Task.from_json_file(description, self.asset_dir)
            robot.update_configuration(q1)
            viz = task.visualize(robots=robot)  # Machine should be red
            robot.update_configuration(q2)
            self.assertIs(robot.data, viz.data)


if __name__ == '__main__':
    unittest.main()
