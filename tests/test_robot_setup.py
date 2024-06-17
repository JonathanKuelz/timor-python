import random
import unittest

import numpy as np
import numpy.testing as np_test
import pinocchio as pin

from timor.Module import ModuleAssembly, ModulesDB
from timor.Robot import PinRobot, RobotBase
from timor.task import Constraints, Tolerance
from timor.utilities import logging, prebuilt_robots, spatial
from timor.utilities.file_locations import robots
from timor.utilities.frames import Frame, NOMINAL_FRAME, WORLD_FRAME
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.transformation import Transformation


class PinocchioRobotSetup(unittest.TestCase):
    """Test if robots can be instantiated, and works as expected."""

    @staticmethod
    def get_random_config(rng: np.random.Generator, robot: RobotBase):
        """Random configuration generator depending on numpy only (no pinocchio, so randomness is controlled)"""
        lim_lower = np.maximum(robot.joint_limits[0, :], -2 * np.pi)
        lim_upper = np.minimum(robot.joint_limits[1, :], 2 * np.pi)
        return (lim_upper - lim_lower) * rng.random(robot.dof) + lim_lower

    def setUp(self) -> None:
        self.package_dir = robots['panda'].parent
        self.urdf = robots['panda'].joinpath('urdf').joinpath('panda.urdf')
        random.seed(1234)
        self.rng = np.random.default_rng(1234)

    def test__first_load_robot(self):
        """Double underscore because tests are run alphabetically and this one should run first"""
        wrapper = pin.RobotWrapper.BuildFromURDF(str(self.urdf), str(self.package_dir))
        robot = PinRobot(wrapper=wrapper)
        self.assertIsInstance(robot, RobotBase)

    def test_base_placement(self):
        new_placement = Transformation.random()

        # Check that moving the robot to a new placement works
        robot_one = PinRobot.from_urdf(self.urdf, self.package_dir)
        robot_two = PinRobot.from_urdf(self.urdf, self.package_dir, base_placement=new_placement)
        robot_one.move(new_placement)
        np_test.assert_array_equal(robot_one.placement.homogeneous, robot_two.placement.homogeneous)

        for _ in range(5):
            robot_one = PinRobot.from_urdf(self.urdf, self.package_dir)
            displacement = Transformation.random()
            inverse = spatial.inv_homogeneous(displacement)
            original = robot_one.placement.homogeneous.copy()
            robot_one.move(displacement)
            robot_one.move(inverse)
            np_test.assert_array_almost_equal(original, robot_one.placement.homogeneous)
            robot_one.set_base_placement(original)  # Should change nothing
            np_test.assert_array_almost_equal(original, robot_one.placement.homogeneous)

        # Test displacement for a robot built from a module
        db = ModulesDB.from_name('geometric_primitive_modules')
        mini_assembly = ModuleAssembly(db, ['J2'], base_connector=(0, 'J2_proximal'))
        jnt_displacement = (Transformation(spatial.rotX(np.pi)) @
                            db.all_connectors_by_id[('J2', 'J2_proximal', 'J2_proximal')].connector2body @
                            list(db.by_id['J2'].joints)[0].parent2joint).homogeneous
        mini_robot = mini_assembly.to_pin_robot(base_placement=new_placement)
        np_test.assert_array_almost_equal(mini_robot.model.jointPlacements[1].homogeneous,
                                          new_placement.homogeneous @ jnt_displacement)

    def test_empty_robot(self):
        wrapper = pin.RobotWrapper(pin.Model())
        robot = PinRobot(wrapper=wrapper)
        self.assertIsInstance(robot, RobotBase)
        self.assertEqual(robot.mass, 0)

    def test_multiple_base_placement(self):
        robot = prebuilt_robots.get_six_axis_modrob()
        fk_eye4 = robot.fk()  # Should be placement of this robot eef without base being moved

        T = Transformation.from_translation((1, 2, 3))
        robot.set_base_placement(T)
        self.assertTrue(Tolerance.DEFAULT_SPATIAL.valid(T @ fk_eye4, robot.fk()))
        T = Transformation.from_translation((1, 0, 0))
        robot.set_base_placement(T)
        self.assertTrue(Tolerance.DEFAULT_SPATIAL.valid(T @ fk_eye4, robot.fk()))

        for i in range(100):
            T = Transformation(Transformation.random())
            robot.set_base_placement(T)
            self.assertTrue(Tolerance.DEFAULT_SPATIAL.valid(T, robot._base_placement))
            self.assertTrue(Tolerance.DEFAULT_SPATIAL.valid(T @ fk_eye4, robot.fk()),
                            msg=f"Could not reconstruct fk after moving base {i + 1} times")

    def test_robot_eef_velocity(self):
        robot = PinRobot.from_urdf(self.urdf, self.package_dir)
        dq0 = np.asarray((1., 0., 0., 0., 0., 0., 0.))
        dq1 = np.asarray((0., 1., 0., 0., 0., 0., 0.))

        # Velocity should be close to omega x r if single joint moves
        for _ in range(100):
            q = self.get_random_config(self.rng, robot)
            # First joint movement. Needs relative movement of 5th (= 1st joint) and EEF frame; specific to Panda robot
            robot.update_configuration(q, dq0)
            fks = robot.fk(kind="full")
            v_rob = robot.tcp_velocity
            J_1toEEF = fks[5].inv @ fks[-1]  # From 1st joint frame (=5th in complete hierarchy) frame to eef
            v_eef_from_J_1 = np.cross(np.asarray((0., 0., 1.)), J_1toEEF.translation)
            # Rotate velocity from 1st joint frame into base (= world) frame for comparison with tcp_velocity output
            np_test.assert_array_almost_equal(v_rob[:3], fks[5].rotation @ v_eef_from_J_1)
            np_test.assert_array_almost_equal(v_rob[3:], np.asarray((0., 0., 1.)))

            # Second joint movement (= 7th frame in complete FK)
            robot.update_configuration(q, dq1)
            v_rob = robot.tcp_velocity
            J_2toEEF = fks[7].inv @ fks[-1]
            v_eef_from_J_2 = np.cross(np.asarray((0., 0., 1.)), J_2toEEF.translation)
            # Rotate velocity from 2nd joint frame into base (= world) frame for comparison
            np_test.assert_array_almost_equal(v_rob[:3], fks[7].rotation @ v_eef_from_J_2)
            np_test.assert_array_almost_equal(v_rob[3:], fks[7].rotation @ np.asarray((0., 0., 1.)))

            # Velocity should not change with translation of base
            robot.set_base_placement(Transformation.from_translation(self.rng.random((3,))))
            robot.update_configuration(q, dq1)
            v_rob_moved = robot.tcp_velocity
            np_test.assert_array_almost_equal(v_rob, v_rob_moved)

            # Velocity should rotate with rotation of base
            randon_transformation = Transformation.random()
            robot.set_base_placement(randon_transformation)
            robot.update_configuration(q, dq1)
            v_rob_moved = robot.tcp_velocity
            np_test.assert_array_almost_equal(v_rob[:3], randon_transformation.inv.rotation @ v_rob_moved[:3])
            np_test.assert_array_almost_equal(v_rob[3:], randon_transformation.inv.rotation @ v_rob_moved[3:])

            # reset
            robot.set_base_placement(Transformation.neutral())

    def test_jacobian_ik_masked(self):
        robot = PinRobot.from_urdf(self.urdf, self.package_dir, rng=self.rng)
        # Create some goals that are not reachable and then try to optimize for certain error terms only:
        N = 50
        fails = 0

        def cost_rot(_r: RobotBase, _q: np.ndarray, _g: Transformation):
            """A custom cost function to optimize for rotation only"""
            return _r.fk(_q, 'tcp').distance(_g).rotation_angle

        def cost_trans(_r: RobotBase, _q: np.ndarray, _g: Transformation):
            """A custom cost function to optimize for translation only"""
            return _r.fk(_q, 'tcp').distance(_g).translation_euclidean

        def get_trans_z_cost(f: str):
            def cost_trans_z(_r: RobotBase, _q: np.ndarray, _g: Transformation):
                """A custom cost function to optimize for translation in z-direction only"""
                if f == 'tcp':
                    ref = NOMINAL_FRAME
                else:
                    ref = WORLD_FRAME
                error = _g.inv @ _r.fk(_q)
                return np.abs(ref.rotate_to_this_frame(error, _g).translation[2])
            return cost_trans_z

        trans_mask = np.array([True, True, True, False, False, False])
        rot_mask = np.array([False, False, False, True, True, True])
        z_mask = np.array([False, False, True, False, False, False])

        def make_distance_matrix(q_s, costs, _g):
            """Compute distance matrix for a set of joint configurations and a cost function"""
            return np.asarray([[c(robot, q, _g.nominal.in_world_coordinates()) for q in q_s] for c in costs])

        for _ in range(N):
            for frame in ('tcp', 'world'):
                goal = ToleratedPose(Transformation.random(self.rng) @
                                     Transformation.from_translation(self.rng.random((3,)) * 100))
                q_init = robot.random_configuration(self.rng)
                kwargs = {'eef_pose': goal, 'ik_method': 'jacobian', 'max_iter': 50, 'allow_random_restart': True,
                          'tcp_or_world_frame': frame, 'q_init': q_init}
                q_rot, s1 = robot.ik(ik_cost_function=cost_rot, error_term_mask=rot_mask, **kwargs)
                q_trans, s2 = robot.ik(ik_cost_function=cost_trans, error_term_mask=trans_mask, **kwargs)
                q_trans_z, s3 = robot.ik(ik_cost_function=get_trans_z_cost(frame), error_term_mask=z_mask, **kwargs)

                # If the goal was reachable, the error masking could be without effect
                self.assertFalse(s1 or s2 or s3)

                mat = make_distance_matrix([q_rot, q_trans, q_trans_z, q_init],
                                           [cost_rot, cost_trans, get_trans_z_cost(frame)], goal)

                self.assertEqual(np.argmin(mat[0, :]), 0)
                fails += np.min(mat[1, :]) != mat[1, 1]  # numeric IK is not super precise, fails _can_ happen
                # Minimizing the z-error in tcp frame is actually not trivial, but we can test for the world frame
                if frame == 'world':
                    self.assertLess(mat[2, 2], mat[2, 3])
            self.assertLess(fails, int(0.1 * (N * 2)), msg=f"Failed to optimize for position only {fails} times")

    def test_robot_fk(self):
        robot = PinRobot.from_urdf(self.urdf, self.package_dir)
        self.assertIsInstance(robot.fk(robot.configuration, 'tcp'), Transformation)

        conf = self.get_random_config(self.rng, robot)
        for kind in ('tcp', 'joints', 'full'):
            # Updating geometry placements should not change kinematics
            self.assertEqual(
                robot.fk(conf, kind, collision=False, visual=False),
                robot.fk(conf, kind, collision=True, visual=True)
            )

    def test_robot_friction(self):
        robot = PinRobot.from_urdf(self.urdf, self.package_dir)
        for i in range(100):
            dq = robot.random_configuration()
            tau_friction = robot.friction(dq)
            np_test.assert_array_almost_equal(tau_friction,
                                              robot.model.damping * dq + robot.model.friction * np.sin(
                                                  np.arctan(100 * dq)))

    def test_robot_id(self):
        robot = PinRobot.from_urdf(self.urdf, self.package_dir)

        def random_values(limits: np.ndarray, N: int = 100):
            """Expects positive limits end returns random values between -limits and +limits"""
            sign_mask = (self.rng.random([N, len(limits)]) > .5)
            rand_val = self.rng.random([N, len(limits)]) * limits
            rand_val[sign_mask] = rand_val[sign_mask] * -1
            return rand_val

        q = np.vstack([self.get_random_config(self.rng, robot) for _ in range(100)])
        dq = random_values(robot.joint_velocity_limits, 100)
        tau = random_values(robot.model.effortLimit, 100)

        for i in range(q.shape[0]):
            # Only considers robot bodies; ignores motor dynamics and friction
            ddq = pin.aba(robot.model, robot.data, q[i], dq[i], tau[i])
            tau_via_rnea = robot.id(q[i], dq[i], ddq)
            tau_via_rnea_without_motor_friction = robot.id(q[i], dq[i], ddq, motor_inertia=False, friction=False)
            np_test.assert_array_almost_equal(tau[i], tau_via_rnea - robot.motor_inertias(ddq) - robot.friction(dq[i]))
            np_test.assert_array_almost_equal(tau[i], tau_via_rnea_without_motor_friction)

            ddq = robot.fd(tau[i], q[i], dq[i], friction=False)
            tau_via_rnea = robot.id(q[i], dq[i], ddq)
            np_test.assert_array_almost_equal(tau[i], tau_via_rnea - robot.motor_inertias(ddq) - robot.friction(dq[i]))

            ddq_with_friction = robot.fd(tau[i], q[i], dq[i], friction=True)
            tau_via_rnea_with_friction = robot.id(q[i], dq[i], ddq_with_friction)
            np_test.assert_array_almost_equal(tau[i], tau_via_rnea_with_friction - robot.motor_inertias(ddq))

    def test_robot_ik(self):
        wrapper = pin.RobotWrapper.BuildFromURDF(str(self.urdf), str(self.package_dir))
        robot = PinRobot(wrapper=wrapper)
        panda_tcp = pin.SE3(spatial.homogeneous(np.array([0, 0, .21]), spatial.rotZ(-np.pi / 4)[:3, :3]))
        robot.add_tcp_frame('panda_joint7', panda_tcp)

        def translation_cost(_r: RobotBase, _q: np.ndarray, _g: np.ndarray):
            """A custom cost function to optimize for translation only"""
            return _r.fk(_q, 'tcp').distance(_g).translation_euclidean

        error_count = {"scipy": 0, "jacobian": 0}
        N = 50
        n = 0
        scipy_tolerance = Tolerance.CartesianXYZ.default()
        jacobian_tolerance = Tolerance.DEFAULT_SPATIAL
        fails = []
        inf_limits = np.ones_like(robot.joint_torque_limits) * np.inf
        zero_limits = np.zeros_like(robot.joint_torque_limits)
        for _ in range(N):
            # Using a random example for which we know, there's at least one goal configuration
            conf = self.get_random_config(self.rng, robot)
            q_init = self.get_random_config(self.rng, robot)
            goal = Frame("", robot.fk(conf), WORLD_FRAME)
            if robot.has_self_collision():
                # Don't test ik in these cases - it might fail due to avoiding self collisions
                continue
            n += 1  # Keep track of how many examples were really worked with
            robot.model.effortLimit = inf_limits  # Make it impossible to fail due to the torque limits
            robot.update_configuration(q_init)
            conf, success = robot.ik(ToleratedPose(goal, scipy_tolerance), q_init=q_init, ik_method='scipy',
                                     ik_cost_function=translation_cost)
            # Check that multiple runs produce different ik solutions
            q_s = [robot.ik(ToleratedPose(goal, scipy_tolerance), ik_cost_function=translation_cost, ik_method='scipy')
                   for _ in range(10)]
            q_s = np.asarray([q for q, s in q_s if s])
            dist = np.linalg.norm(q_s - q_s[0, :], axis=1)
            self.assertTrue(np.any(dist > 1e-3))
            if success:
                # Here, we only optimize for position, therefore the Cartesian XYZ tolerance
                self.assertTrue(scipy_tolerance.valid(goal.in_world_coordinates(), robot.fk(conf)))
                self.assertTrue(max(abs(conf)) < 2 * np.pi)  # IK result should be mapped to (-2pi, 2pi)
                self.assertFalse(robot.has_self_collision(conf))
            else:
                error_count['scipy'] += 1

            robot.update_configuration(q_init)  # Prevent using the scipy result as initial guess
            conf, success = robot.ik(ToleratedPose(goal, jacobian_tolerance), ik_method='jacobian')
            # Check that multiple runs produce different ik solutions
            q_s = [robot.ik(ToleratedPose(goal, jacobian_tolerance), ik_method='jacobian') for _ in range(10)]
            q_s = np.asarray([q for q, s in q_s if s])
            dist = np.linalg.norm(q_s - q_s[0, :], axis=1)
            self.assertTrue(np.any(dist > 1e-3))
            if success:
                np_test.assert_array_almost_equal(goal.in_world_coordinates().homogeneous, robot.fk(conf).homogeneous,
                                                  decimal=2)
                self.assertTrue(jacobian_tolerance.valid(goal.in_world_coordinates(), robot.fk(conf)))
                self.assertTrue(max(abs(conf)) < 2 * np.pi)
                self.assertFalse(robot.has_self_collision(conf))
            else:
                error_count['jacobian'] += 1
                fails.append((goal, conf))

            # Test that masking joints in the ik works
            q_start = self.get_random_config(self.rng, robot)
            mask = np.array([random.random() > 0.5 for _ in range(len(conf))])
            conf, success = robot.ik(ToleratedPose(goal, jacobian_tolerance), q_init=q_start, joint_mask=mask,
                                     ik_method='jacobian')
            np_test.assert_array_almost_equal(q_start[~mask], conf[~mask])
            conf, success = robot.ik(ToleratedPose(goal, jacobian_tolerance), q_init=q_start,
                                     ik_method='scipy', joint_mask=mask)
            np_test.assert_array_almost_equal(q_start[~mask], conf[~mask])

            # Test joint torque limits
            robot.model.effortLimit = zero_limits
            torque_constraint = Constraints.JointLimits(parts=('tau',))
            conf, success = robot.ik(ToleratedPose(goal, jacobian_tolerance), max_iter=50, ik_method='jacobian',
                                     outer_constraints=(torque_constraint,))
            self.assertFalse(success)
            conf, success = robot.ik(ToleratedPose(goal, jacobian_tolerance), max_iter=50, ik_method='scipy',
                                     outer_constraints=(torque_constraint,))
            self.assertFalse(success)

        for k, v in error_count.items():
            self.assertLess(v, 0.03 * n, f"ik should be able to resolve almost any fk; "
                                         f"not for {k} with {v / n * 100:.2f}% errors.")

        if error_count['scipy'] + error_count['jacobian'] > 0:
            logging.warning('IK test failed {} times for scipy, {} times for jacobian.'.format(
                error_count['scipy'], error_count['jacobian']
            ))

    def test_robot_move_base(self):
        """Creates two robots, one with the base moved to another origin and then checks whether
        frame displacement checks out."""
        displacement = pin.SE3(spatial.homogeneous(np.asarray([1, -.1, .75])))
        wrapper_one = pin.RobotWrapper.BuildFromURDF(str(self.urdf), str(self.package_dir))
        wrapper_two = pin.RobotWrapper.BuildFromURDF(str(self.urdf), str(self.package_dir))
        robot_default = PinRobot(wrapper=wrapper_one)
        robot_moved = PinRobot(wrapper=wrapper_two, base_placement=displacement.copy())
        conf = self.get_random_config(self.rng, robot_default)

        self.assertEqual(
            tuple(p.multiply_from_left(displacement) for p in robot_default.fk(conf, 'full')[1:]),
            robot_moved.fk(conf, 'full')[1:]
        )
        # sanity check (should be covered by test above already, but tcp is important!)
        np_test.assert_array_almost_equal(
            displacement.homogeneous @ robot_default.fk(conf, 'tcp').homogeneous,
            robot_moved.fk(conf, 'tcp').homogeneous
        )

    def test_robot_random_conf(self):
        robot = PinRobot.from_urdf(self.urdf, self.package_dir)
        for _ in range(1000):
            self.assertTrue(robot.q_in_joint_limits(robot.random_configuration()))

    def test_robot_viz(self):
        wrapper = pin.RobotWrapper.BuildFromURDF(str(self.urdf), str(self.package_dir))
        robot = PinRobot(wrapper=wrapper)
        self.assertIsNotNone(robot.visualize())
        self.assertIsNotNone(robot.visualize_self_collisions())

    def test_robot_manip(self):
        robot = PinRobot.from_urdf(self.urdf, self.package_dir)
        # Some best guess singular / non-singular panda poses
        self.assertAlmostEqual(0., robot.manipulability_index(np.asarray((0., 0., 0., 0., 0., 0., 0.))))
        self.assertGreater(
            0.5, robot.manipulability_index(np.asarray((0., -np.pi / 4, 0., np.pi / 2, 0., np.pi / 2, 0.))))


if __name__ == '__main__':
    unittest.main()
