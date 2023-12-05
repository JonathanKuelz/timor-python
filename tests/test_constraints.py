import unittest

import numpy as np

from timor import ModuleAssembly, Robot
from timor.task import Constraints, CostFunctions, Goals, Solution, Task, Tolerance
from timor.utilities import spatial
from timor.utilities.file_locations import robots
from timor.utilities.frames import WORLD_FRAME
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.trajectory import Trajectory
from timor.utilities.transformation import Transformation


class TestConstraints(unittest.TestCase):

    def setUp(self):
        self.task_header = Task.TaskHeader('test_case', 'py', tags=['debug', 'test'])
        self.panda_urdf = robots['panda'].joinpath('urdf').joinpath('panda.urdf')
        self.robot = Robot.PinRobot.from_urdf(self.panda_urdf, robots['panda'].parent)
        self.rng = np.random.default_rng(420)

    def test_constraint_abstraction_level(self):
        """Tests that the differentiation between goal-level and task-level constraints works properly"""
        panda = Robot.PinRobot.from_urdf(self.panda_urdf, robots['panda'].parent)

        base_constraint = Constraints.BasePlacement(ToleratedPose(Transformation.neutral(),
                                                                  tolerance=Tolerance.DEFAULT_SPATIAL))
        basic_joint_constraint = Constraints.JointLimits(parts=('q',))
        additional_joint_constraint = Constraints.JointAngles(panda.joint_limits * 0.99)

        start_conf = panda.random_configuration(self.rng)
        end_conf = panda.random_configuration(self.rng)

        goal_pose = ToleratedPose(panda.fk(end_conf))
        self.assertFalse(panda.has_self_collision(end_conf))
        at_goal = Goals.At('At_Goal', goal_pose, constraints=[additional_joint_constraint])

        task = Task.Task(self.task_header, goals=[at_goal],
                         constraints=[base_constraint, basic_joint_constraint, Constraints.AllGoalsFulfilled()])
        task_without_all_goals_fulfilled = Task.Task(self.task_header, goals=[at_goal],
                                                     constraints=[base_constraint, basic_joint_constraint])

        steps = 100
        dt = .1
        easy_trajectory = Trajectory(t=dt, q=np.linspace(start_conf, end_conf, steps + 1),
                                     goal2time={at_goal.id: steps * dt})
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
        addon_trajectory = Trajectory(t=dt, q=np.linspace(end_conf, panda.joint_limits[1, :], steps + 1))
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

    def test_constraint_scoping(self):
        """
        Tests that the different time scopes (one time, up until t, full solution) work properly

        This should work for the following constraints:

        - Joint limits
        - Collision free
        - Self Collision free
        - End effector (tested in its own test case)
        - Always true (trivial: always true)
        - Goals by same robot (TODO, not yet implemented constraint)
        """
        robot = self.robot

        # We test the non-trivial constraints all in the same manner: The solution hurts the constraints at one specific
        # time, but not at the others. This is then checked vie the constraints methods.
        num_via_points = 5
        non_trivial_constraints = [
            Constraints.JointLimits(parts=('q',)),
            Constraints.CollisionFree(),
            Constraints.SelfCollisionFree(),
        ]
        trivial_constraints = [
            Constraints.AlwaysTrueConstraint(),
            Constraints.AllGoalsFulfilled(),
        ]

        #  First, we construct a completely valid solution
        via_points = list()
        while len(via_points) < num_via_points:
            q = robot.random_configuration(self.rng)
            if not robot.has_self_collision(q):
                via_points.append(q)

        q_trajectory = np.vstack([
            *[np.linspace(via_points[i], via_points[i + 1], 10, endpoint=False) for i in range(num_via_points - 1)],
            via_points[-1]
        ])
        for i, q in enumerate(q_trajectory):
            # We are cheating a bit here, as this creates unrealistic trajectories with jittery movements, but
            #  this is not the concern of this test case. After all, we don't care about joint velocity and acceleration
            if robot.has_self_collision(q):
                q_trajectory[i, :] = q_trajectory[i - 1, :]
        goals = [Goals.At(f'At_{i}', ToleratedPose(robot.fk(q))) for i, q in enumerate(via_points)]
        trajectory = Trajectory(
            t=.1,
            q=q_trajectory,
            goal2time={goals[i].id: i for i in range(num_via_points)}
        )

        all_constraints = non_trivial_constraints + trivial_constraints
        solution = Solution.SolutionTrajectory(
            trajectory=trajectory,
            header=Solution.SolutionHeader.empty(),
            task=Task.Task(Task.TaskHeader('tmp'), goals=goals, constraints=all_constraints),
            assembly=ModuleAssembly.from_monolithic_robot(robot),
            cost_function=CostFunctions.RobotMass()
        )
        self.assertTrue(solution.valid)

        # Manipulate the solution:
        t_manipulate = 3.1
        t_idx = solution.get_time_id(t_manipulate)

        q_collision = robot.random_configuration(self.rng)
        while not robot.has_self_collision(q_collision):
            q_collision = robot.random_configuration(self.rng)
        q_falsify = q_collision + np.ones_like(q_collision) * 4 * np.pi

        trajectory.q[t_idx] = q_falsify

        for constraint in non_trivial_constraints:
            self.assertFalse(constraint.fulfilled(solution))
            for t in trajectory.t:
                if t != t_manipulate:
                    self.assertTrue(constraint.is_valid_at(solution, t))
                else:
                    self.assertFalse(constraint.is_valid_at(solution, t))
                if t < t_manipulate:
                    self.assertTrue(constraint.is_valid_until(solution, t))
                else:
                    self.assertFalse(constraint.is_valid_until(solution, t))

        for t in trajectory.t:
            self.assertTrue(Constraints.AlwaysTrueConstraint().is_valid_at(solution, t))
            self.assertTrue(Constraints.AlwaysTrueConstraint().is_valid_until(solution, t))
            self.assertTrue(Constraints.AllGoalsFulfilled().is_valid_until(solution, t))

        t_final = solution.trajectory.t[-1]
        solution.task.goals = solution.task.goals + (Goals.At('error', ToleratedPose(Transformation.neutral())),)
        self.assertFalse(Constraints.AllGoalsFulfilled().fulfilled(solution))
        self.assertFalse(Constraints.AllGoalsFulfilled().is_valid_until(solution, t_final))
        self.assertTrue(Constraints.AlwaysTrueConstraint().fulfilled(solution))

    def test_eef_constraint(self):
        """Test the functionality of the EEF constraint"""
        eef_constraint_allow_any_pose = Constraints.EndEffector(
            pose=ToleratedPose(nominal=WORLD_FRAME, tolerance=Tolerance.AlwaysValidTolerance()))
        qs = np.asarray([self.robot.random_configuration(self.rng) for _ in range(10)])
        eef_constraint_allow_first_pose = Constraints.EndEffector(
            pose=ToleratedPose(self.robot.fk(qs[0, :]), tolerance=Tolerance.DEFAULT_SPATIAL)
        )

        # Check pose
        sol = Solution.SolutionTrajectory(Trajectory(t=1., q=qs, goal2time={}), Solution.SolutionHeader("tmp"),
                                          Task.Task(Task.TaskHeader('tmp'), ),
                                          ModuleAssembly.from_monolithic_robot(self.robot), CostFunctions.RobotMass(),
                                          Transformation.neutral())
        self.assertTrue(eef_constraint_allow_first_pose.is_valid_until(sol, 0.))
        self.assertTrue(eef_constraint_allow_first_pose.is_valid_at(sol, 0.))
        self.assertFalse(eef_constraint_allow_first_pose.is_valid_at(sol, 1.))
        self.assertFalse(eef_constraint_allow_first_pose.is_valid_until(sol, 1.))
        self.assertFalse(eef_constraint_allow_first_pose.fulfilled(sol))
        self.assertTrue(eef_constraint_allow_any_pose.is_valid_until(sol, 0.))
        self.assertTrue(eef_constraint_allow_any_pose.is_valid_at(sol, 0.))
        self.assertTrue(eef_constraint_allow_any_pose.is_valid_until(sol, 1.))
        self.assertTrue(eef_constraint_allow_any_pose.is_valid_at(sol, 1.))
        self.assertTrue(eef_constraint_allow_any_pose.fulfilled(sol))

        # Constructed case - movement along line with invariant rotation about line, e.g. drilling
        orientation_tolerance = Tolerance.RotationAxis(n_x=(-np.pi / 100, np.pi / 100),
                                                       n_y=(-np.pi / 100, np.pi / 100),
                                                       n_z=(-np.pi, np.pi))
        ik_tolerance = Tolerance.Composed((orientation_tolerance, Tolerance.CartesianXYZ.default()))
        start_pose = Transformation.from_translation((0.5, 0., 0.5)) @ spatial.rotY(np.pi / 2)
        # Construct movement steps - q_start, q_end, q_mid_ok, q_mid_nok (small rot), q_end_to_far
        qs = []
        N_middle = 10  # How many random valid position inbetween start and end pose
        t_good = 2 + N_middle  # start, end + valid in the middle
        for T in (start_pose,  # q_start
                  Transformation.from_translation((.2, 0., 0.)) @ start_pose,  # q_end
                  # q_mid - all with random translation in 0, 0.2 and rotation about drill axis
                  *(Transformation.from_translation((self.rng.random() * 0.2, 0., 0.)) @
                    start_pose @ spatial.rotZ(self.rng.random()) for _ in range(N_middle)),
                  Transformation.from_translation((.11, 0., 0.)) @ start_pose @ spatial.rotX(0.05),  # q_mid_not_ok
                  Transformation.from_translation((.11, 0., 0.)) @ start_pose @ spatial.rotY(-0.05),  # q_mid_not_ok
                  Transformation.from_translation((.21, 0., 0.)) @ start_pose  # q_end_to_far
                  ):
            q, success = self.robot.ik(ToleratedPose(T, ik_tolerance))
            self.assertTrue(success, f"ik for {str(T)} failed")
            qs.append(q)
        drill_eef_constraint = Constraints.EndEffector(pose=ToleratedPose(
            nominal=start_pose,
            tolerance=Tolerance.Composed((
                Tolerance.CartesianXYZ((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.201)),  # Drill along z
                orientation_tolerance
            ))
        ))
        drill_test_traj = Trajectory(1., q=np.asarray(qs), goal2time={})
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
        vel_test_traj = Trajectory(t=np.asarray([0., 1., 2., 3., 4.]), q=np.zeros((5, 7)), goal2time={},
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
            self.assertTrue(c.is_valid_at(sol, 0.))
            self.assertTrue(c.is_valid_until(sol, 0.))
            self.assertFalse(c.is_valid_at(sol, 1.))
            self.assertFalse(c.is_valid_at(sol, 2.))
            self.assertTrue(c.is_valid_at(sol, 3.))
            self.assertFalse(c.is_valid_at(sol, 4.))
            for t in (1, 2, 3, 4):
                self.assertFalse(c.is_valid_until(sol, t))
        for t in np.arange(0., 3., 1.):
            self.assertTrue(eef_constraint_allow_any_pose.is_valid_until(sol, t))
            self.assertFalse(eef_constraint_allow_first_pose.is_valid_until(sol, t))

    def test_robot_constraints(self):
        """Test the functionality of the robot constraints"""
        q_rand = self.robot.random_configuration(self.rng)
        dq_rand = self.rng.random(self.robot.dof) * self.robot.joint_velocity_limits
        dq0 = np.zeros_like(dq_rand)

        c = Constraints.AlwaysTrueConstraint()
        self.assertTrue(c.check_single_state(None, self.robot, q_rand, dq_rand))

        # Test limits
        c = Constraints.JointLimits(parts=('q', 'dq', 'tau'))
        self.assertTrue(c.check_single_state(None, self.robot, q_rand, dq0))
        self.assertTrue(c.check_single_state(None, self.robot, q_rand, dq_rand))
        self.assertFalse(c.check_single_state(None, self.robot, q_rand, self.robot.joint_velocity_limits + 1))
        # Ensure that constraint check does not modify the robot's placement
        old_placement = self.robot.placement
        self.robot.set_base_placement(Transformation.from_translation((1., 2., 3.)))
        c.check_single_state(None, self.robot, q_rand, dq_rand)
        self.assertEqual(self.robot.placement, Transformation.from_translation((1., 2., 3.)))
        self.robot.set_base_placement(old_placement)

        # Test collision
        c1 = Constraints.CollisionFree()
        c2 = Constraints.SelfCollisionFree()
        while not self.robot.has_self_collision():
            q_rand = self.robot.random_configuration(self.rng)
            self.robot.update_configuration(q_rand)
        self.assertFalse(c1.check_single_state(None, self.robot, q_rand, dq0))
        self.assertFalse(c2.check_single_state(None, self.robot, q_rand, dq0))
