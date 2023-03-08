from copy import deepcopy
import random
from typing import Callable
import unittest

import numpy as np
import numpy.testing as np_test

from timor import Transformation
from timor.utilities import logging
from timor.utilities.dtypes import Lazy
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.trajectory import Trajectory
import timor.utilities.errors as err


class TestTrajectoryClasses(unittest.TestCase):
    """Tests trajectory classes in utilities."""

    def setUp(self) -> None:
        # Fix random seeds to have deterministic tests
        random.seed(123)
        np.random.seed(123)

    def test_Trajectory(self):
        t = np.linspace(0, 1, 1001)
        q = np.random.random((1001, 6))
        t_goal = .51
        goals = {'Epic': t_goal}

        traj_with_lazy_dq = Trajectory(t, q=q, goal2time=goals)

        # Test lazy evaluation of dq, ddq
        self.assertIsNone(traj_with_lazy_dq._dq.value)
        self.assertIsNone(traj_with_lazy_dq._ddq.value)

        # Test property shapes
        self.assertEqual(traj_with_lazy_dq.t.size, t.size)
        self.assertEqual(traj_with_lazy_dq.q.size, traj_with_lazy_dq.dq.size)
        self.assertEqual(traj_with_lazy_dq.q.size, traj_with_lazy_dq.ddq.size)

        # test de-serialize pipeline
        self.assertEqual(Trajectory.from_json_data(traj_with_lazy_dq.to_json_data()), traj_with_lazy_dq)
        self.assertTrue("dq" not in traj_with_lazy_dq.to_json_data())
        self.assertTrue("ddq" not in traj_with_lazy_dq.to_json_data())

        traj_with_dq_array = Trajectory(traj_with_lazy_dq.t, q=traj_with_lazy_dq.q, dq=traj_with_lazy_dq.dq)
        self.assertIsInstance((traj_with_lazy_dq + traj_with_lazy_dq)._dq, Lazy)
        self.assertIsInstance((traj_with_lazy_dq + traj_with_dq_array)._dq, np.ndarray)
        self.assertIsInstance((traj_with_dq_array + traj_with_lazy_dq)._dq, np.ndarray)
        self.assertIsInstance((traj_with_dq_array + traj_with_dq_array)._dq, np.ndarray)

        traj_with_lazy_dq = Trajectory(1 / 1000, q=q.tolist(),
                                       goal2time=goals)  # Not best practice, but giving a list should be possible
        self.assertEqual(traj_with_lazy_dq.t.size, t.size)
        np_test.assert_array_equal(traj_with_lazy_dq.t, t)

        self.assertEqual(traj_with_lazy_dq[0].t, t[0])
        np_test.assert_array_equal(q, traj_with_lazy_dq[:].q)
        includes_goal = int(np.where(t == t_goal)[0])
        self.assertEqual(dict(), traj_with_lazy_dq[:includes_goal].goal2time)
        self.assertIn('Epic', traj_with_lazy_dq[includes_goal:].goal2time.keys())

        with self.assertRaises(ValueError):  # Expect time to be increasing 1
            traj_with_lazy_dq = Trajectory(t=np.asarray((1., 3., 2.)), q=np.random.random((3, 3)))
        with self.assertRaises(ValueError):  # Expect time to be increasing 2
            traj_with_lazy_dq = Trajectory(t * 0, q=q, goal2time=goals)
        with self.assertRaises(TypeError):
            traj_with_lazy_dq = Trajectory()
        with self.assertRaises(IndexError):
            traj_with_lazy_dq = traj_with_lazy_dq[len(t)]
        with self.assertRaises(ValueError):
            traj_with_lazy_dq = Trajectory.stationary(1., q=np.ones((2, 2)))
        with self.assertRaises(ValueError):  # Goals should be within time
            traj_with_lazy_dq = Trajectory(t=np.asarray((0., 1.)), q=np.random.random((2, 2)), goal2time={"test": 2.})
        # Barely in time
        traj_with_lazy_dq = Trajectory(t=np.asarray((0., 1.)), q=np.random.random((2, 2)), goal2time={"test": 1.})
        with self.assertRaises(ValueError):  # _allowed_deviation_in_time > 0!
            traj_with_lazy_dq = Trajectory(t=t, q=q, _allowed_deviation_in_time=-1e-5)

        empty = Trajectory.empty()
        self.assertEqual(empty, empty)
        self.assertEqual(empty, empty + empty)
        self.assertEqual(traj_with_lazy_dq, traj_with_lazy_dq + empty)
        self.assertEqual(traj_with_lazy_dq, empty + traj_with_lazy_dq)

    def test_untimed(self):
        """Based on issue #23."""
        arr = np.random.rand(100, 1)
        t = Trajectory(q=arr)
        logging.debug(t.dq)
        t.plot()
        self.assertEqual(t, t)
        self.assertEqual(t, Trajectory.from_json_data(t.to_json_data()))
        t2 = t + t
        self.assertEqual(len(t2), 2 * len(t))
        self.assertFalse(t.has_dq)
        self.assertFalse(t.has_ddq)

    def test_scalar_t(self):
        """Based on issue #24."""
        q = np.random.rand(100)  # Should be 100 time-steps of 1 DoF trajectory
        dt = .1
        t = Trajectory(t=dt, q=q)
        self.assertTupleEqual(t.q.shape, (len(q), 1))
        self.assertTupleEqual(t.t.shape, (len(q),))
        self.assertEqual(t.t[0], 0.)
        self.assertEqual(t.t[-1], (q.shape[0] - 1) * dt)

    def test_deepcopy(self):
        """Based on issue #25."""
        def _test_deepcopy(original: Trajectory, new: Trajectory, change_field: Callable, is_not_none_fields):
            for f in is_not_none_fields:
                self.assertIsNotNone(f)
            self.assertEqual(original, new)
            self.assertFalse(original is new)
            change_field()
            self.assertNotEqual(original, new)

        def _inc_last_element(array):
            array[-1] += 1

        q = np.random.rand(100, 1)
        t = np.arange(100)

        traj = Trajectory(t, q, q, q)
        traj_2 = deepcopy(traj)
        self.assertIsInstance(traj_2._dq, np.ndarray)
        self.assertIsInstance(traj_2._ddq, np.ndarray)
        _test_deepcopy(traj, traj_2, lambda: _inc_last_element(traj_2.ddq),
                       [traj_2.ddq, traj_2._ddq, traj_2.dq, traj_2._dq])

        traj = Trajectory(t, q, q)
        traj_2 = deepcopy(traj)
        self.assertIsInstance(traj_2._dq, np.ndarray)
        self.assertIsInstance(traj_2._ddq, Lazy)
        _test_deepcopy(traj, traj_2, lambda: _inc_last_element(traj_2.dq),
                       [traj_2.ddq, traj_2._ddq, traj_2.dq, traj_2._dq])

        traj = Trajectory(t, q)
        traj_2 = deepcopy(traj)
        self.assertIsInstance(traj_2._dq, Lazy)
        self.assertIsInstance(traj_2._ddq, Lazy)
        _test_deepcopy(traj, traj_2, lambda: _inc_last_element(traj_2.dq),
                       [traj_2.ddq, traj_2._ddq, traj_2.dq, traj_2._dq])

    def test_PoseTrajectory(self):
        example_pose_trajectory = Trajectory(pose=np.asarray([
            ToleratedPose(Transformation.neutral()),
            ToleratedPose(Transformation.from_translation((0, 0, 1))),
            ToleratedPose(Transformation.from_translation((0, 1, 1)))]))
        example_timed_pose_trajectory = Trajectory(t=np.asarray((0., 0.1, 0.2)),
                                                   pose=example_pose_trajectory.pose,
                                                   _allowed_deviation_in_time=0.01)

        self.assertEqual(3, len(example_pose_trajectory))
        self.assertTrue(example_pose_trajectory[2].pose[0].valid(Transformation.from_translation((0, 0.9999, 1))))
        t_new = Trajectory.from_json_data(example_pose_trajectory.to_json_data())
        self.assertEqual(example_pose_trajectory, t_new)
        with self.assertRaises(ValueError):  # len(t) != len(pose)
            _ = Trajectory(t=np.asarray((0., 0.1, 0.2, 0.3)),
                           pose=example_pose_trajectory.pose,
                           _allowed_deviation_in_time=0.01)
        with self.assertRaises(err.TimeNotFoundError):  # Index out of _allowed_deviation_in_time
            example_timed_pose_trajectory.get_at_time(0.15)
        with self.assertRaises(ValueError):
            example_pose_trajectory.get_at_time(0.1)
        self.assertTrue(example_timed_pose_trajectory.get_at_time(0.1).pose[0].
                        valid(example_pose_trajectory.pose[1].nominal))
        # Check access via index instead of time
        self.assertTrue(example_timed_pose_trajectory[1].pose[0].valid(example_pose_trajectory.pose[1].nominal))
        self.assertFalse(example_timed_pose_trajectory.get_at_time(0.1).pose[0].
                         valid(example_pose_trajectory.pose[2].nominal))
        t_timed_new = Trajectory.from_json_data(example_timed_pose_trajectory.to_json_data())
        self.assertEqual(example_timed_pose_trajectory, t_timed_new)
        self.assertNotEqual(example_pose_trajectory, example_timed_pose_trajectory)


if __name__ == '__main__':
    unittest.main()
