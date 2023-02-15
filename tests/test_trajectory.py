import random
import unittest

import numpy as np
import numpy.testing as np_test

from timor import Transformation
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
        traj_with_dq_array = Trajectory.from_json_data(traj_with_lazy_dq.to_json_data())
        self.assertEqual(traj_with_lazy_dq, traj_with_dq_array)

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

        empty = Trajectory(t=[], q=[], goal2time={})
        self.assertEqual(traj_with_lazy_dq, traj_with_lazy_dq + empty)
        self.assertEqual(traj_with_lazy_dq, empty + traj_with_lazy_dq)

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
