from time import process_time
import unittest

import numpy as np
import pinocchio as pin

from timor import Transformation
from timor.task.Task import Task
import timor.utilities.prebuilt_robots
from timor.utilities import ik_cpp
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.file_locations import test_data


class CustomTypeUnitTests(unittest.TestCase):
    """Test custom cpp ik utilities."""

    def setUp(self) -> None:
        """Set up a six axis robot for testing."""
        self.assembly = timor.utilities.prebuilt_robots.get_six_axis_assembly()
        self.q0 = self.assembly.robot.configuration

    def test_fk(self, n_iter: int = 100):
        """Check that fk and model are transferred correctly."""
        for _ in range(n_iter):
            q = self.assembly.robot.random_configuration()
            T = ik_cpp.fk(self.assembly.robot.model,
                          self.assembly.robot.data,
                          q,
                          self.assembly.robot.tcp)
            T = Transformation.from_roto_translation(T.rotation, T.translation)
            self.assembly.robot.update_configuration(self.q0)
            self.assertEqual(self.assembly.robot.fk(q), T)

    def test_ik(self, n_iter: int = 100, max_iter_ik: int = 1000):
        """Check that ik and model are transferred correctly."""
        err_count_cpp = 0
        err_count_py = 0
        t_cpp = []
        t_py = []
        for _ in range(n_iter):
            q = self.assembly.robot.random_configuration()
            T_des = self.assembly.robot.fk(q)
            T_des_tolerated = ToleratedPose(T_des)
            self._reset_robot()
            # q_init = q should always converge
            q_ik, success, iter_left = ik_cpp.ik(
                self.assembly.robot.model, self.assembly.robot.data,
                pin.SE3(T_des.homogeneous), self.assembly.robot.tcp,
                q, pin.GeometryModel(),  # Pass empty if no collision detection desired
                max_iter=max_iter_ik)
            self.assertTrue(T_des_tolerated.valid(self.assembly.robot.fk(q_ik)))
            self.assertGreater(iter_left, max_iter_ik * 0.9)  # Should at max use a few iterations

            # Start from 0 pose
            self._reset_robot()
            t_start = process_time()
            q_ik, success, iter_left = ik_cpp.ik(
                self.assembly.robot.model, self.assembly.robot.data,
                pin.SE3(T_des.homogeneous), self.assembly.robot.tcp,
                self.q0, self.assembly.robot.collision,
                max_iter=max_iter_ik * 10,  # Is dumber but way faster - give a few more steps
                random_restart=True)
            t_cpp.append(process_time() - t_start)
            self.assertGreaterEqual(iter_left, -1)
            if success:  # CPP Code should be conservative and really just return valid if sure
                self.assertTrue(T_des_tolerated.valid(self.assembly.robot.fk(q_ik)))
                self.assertTrue(np.all(self.assembly.robot.joint_limits[0, :] < q_ik), f"q_ik: {q_ik}")
                self.assertTrue(np.all(q_ik < self.assembly.robot.joint_limits[1, :]), f"q_ik: {q_ik}")
                self.assertFalse(self.assembly.robot.has_self_collision(q_ik))
            else:
                err_count_cpp += 1

            # Compare to python implementation
            self._reset_robot()
            t_start = process_time()
            q_ik, success = self.assembly.robot.ik(T_des_tolerated, q_init=self.q0, allow_random_restart=True,
                                                   max_iter=max_iter_ik)
            t_py.append(process_time() - t_start)
            err_count_py += not success

        print(f"C++ Failed {err_count_cpp} times out of {n_iter} iterations")
        print(f"C++ Average time {np.mean(t_cpp)} +/- {np.std(t_cpp)}")
        print(f"Python Failed {err_count_py} times out of {n_iter} iterations")
        print(f"Python Average time {np.mean(t_py)} +/- {np.std(t_py)}")

    def test_ik_cpp_in_robot(self, n_iter: int = 100, max_iter_ik: int = 1000):
        """Test that ik_cpp usable via high-level robot ik interface."""
        err_count_cpp = 0
        err_count_py = 0
        t_cpp = []
        t_py = []
        rng = np.random.default_rng(1)
        task = Task.from_id('simple/PTP_2', test_data.joinpath('sample_tasks'))
        self.assembly.robot.set_base_placement(task.base_constraint.base_pose.nominal)
        for i in range(n_iter):
            q = self.assembly.robot.random_configuration(rng)
            T_des = self.assembly.robot.fk(q)
            T_des_tolerated = ToleratedPose(T_des)
            self._reset_robot()

            q_ik, success = self.assembly.robot.ik(T_des_tolerated, q_init=q, max_iter=max_iter_ik,
                                                   ik_method='jacobian_cpp',
                                                   # Should be able to set about any weight here
                                                   weight_translation=1.0 + rng.uniform(-.5, .5),
                                                   weight_rotation=.5 / np.pi + rng.uniform(0., 1.))
            if not self.assembly.robot.has_self_collision(q) or success:
                self.assertTrue(T_des_tolerated.valid(self.assembly.robot.fk(q_ik)))

            # Start from 0 pose
            self._reset_robot()
            t_start = process_time()
            q_ik, success = self.assembly.robot.ik(T_des_tolerated, q_init=self.q0, max_iter=max_iter_ik * 10,
                                                   ik_method='jacobian_cpp', task=task)
            t_cpp.append(process_time() - t_start)
            err_count_cpp += not success
            if success:
                self.assertTrue(T_des_tolerated.valid(self.assembly.robot.fk(q_ik)))
                self.assertTrue(np.all(self.assembly.robot.joint_limits[0, :] < q_ik), f"q_ik: {q_ik}")
                self.assertTrue(np.all(q_ik < self.assembly.robot.joint_limits[1, :]), f"q_ik: {q_ik}")
                self.assertFalse(self.assembly.robot.has_self_collision(q_ik))
                self.assertFalse(self.assembly.robot.has_collisions(task))

            self._reset_robot()
            t_start = process_time()
            q_ik, success = self.assembly.robot.ik(T_des_tolerated, q_init=self.q0, ik_method='jacobian',
                                                   task=task)
            t_py.append(process_time() - t_start)
            err_count_py += not success

        print(f"C++ Failed {err_count_cpp} times out of {n_iter} iterations")
        print(f"C++ Average time {np.mean(t_cpp)} +/- {np.std(t_cpp)}")
        print(f"Python Failed {err_count_py} times out of {n_iter} iterations")
        print(f"Python Average time {np.mean(t_py)} +/- {np.std(t_py)}")

    def _reset_robot(self):
        self.assembly.robot.update_configuration(self.q0)
        self.assembly.robot.fk(self.q0)  # Force that robot data struct is zeroed
        self.assertTrue(np.allclose(self.assembly.robot.data.oMf[self.assembly.robot.tcp].translation,
                                    np.asarray((0.09, 0., -0.1)) + self.assembly.robot.placement.translation))

    def test_cpp_collision(self, iterations: int = 10000):
        """Test cpp collision checking."""
        cpp_time = py_time = 0
        qs = []
        colls = []
        for _ in range(iterations):
            qs.append(self.assembly.robot.random_configuration())
            self._reset_robot()
            t0 = process_time()
            colls.append(ik_cpp.has_self_collision(self.assembly.robot.model,
                                                   self.assembly.robot.data,
                                                   self.assembly.robot.collision,
                                                   qs[-1]))
            cpp_time += process_time() - t0
            self._reset_robot()
            t0 = process_time()
            collision_py = self.assembly.robot.has_self_collision(qs[-1])
            py_time += process_time() - t0
            self.assertEqual(colls[-1], collision_py)

        t0 = process_time()
        colls_vec = ik_cpp.has_self_collision_vec(self.assembly.robot.model,
                                                  self.assembly.robot.data,
                                                  self.assembly.robot.collision,
                                                  np.asarray(qs))
        vec_time = process_time() - t0
        self.assertTrue(np.all(colls == colls_vec))

        print(f"CPP collision check took {cpp_time} seconds.")
        print(f"Python collision check took {py_time} seconds.")
        print(f"Vectorized CPP collision check took {vec_time} seconds.")
