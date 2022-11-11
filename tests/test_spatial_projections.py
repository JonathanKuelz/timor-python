import random
import unittest

import numpy as np
import numpy.testing as np_test

from timor.utilities import spatial
from timor.utilities.spatial import dh_extended_to_homogeneous, homogeneous, rotX, rotZ
from timor.utilities.transformation import Transformation


class TestSpatialutilities(unittest.TestCase):
    """Place to test spatial projections."""
    def setUp(self) -> None:
        np.random.seed(0)
        random.seed(0)

    @staticmethod
    def test_dh_extended_to_homogeneous():
        np_test.assert_array_almost_equal(dh_extended_to_homogeneous(a=1), homogeneous((1, 0, 0)))
        np_test.assert_array_almost_equal(dh_extended_to_homogeneous(alpha=1), rotX(1))
        np_test.assert_array_almost_equal(dh_extended_to_homogeneous(delta=1), rotZ(-1))
        np_test.assert_array_almost_equal(dh_extended_to_homogeneous(p=1), homogeneous((0, 0, -1)))
        np_test.assert_array_almost_equal(dh_extended_to_homogeneous(n=1), homogeneous((0, 0, 1)))

        np_test.assert_array_almost_equal(dh_extended_to_homogeneous(p=1, n=1), Transformation.neutral().homogeneous)
        np_test.assert_array_almost_equal(dh_extended_to_homogeneous(n=1, a=1), homogeneous((1, 0, 1)))

    @staticmethod
    def test_euler():
        for _ in range(100):
            seq = 'xxx'
            while seq[0] == seq[1] or seq[1] == seq[2]:  # Euler angles cannot be consecutive
                seq = ''.join(random.choice('xyz') for _ in range(3))
            R = spatial.random_rotation()
            R_dash = spatial.euler2mat(spatial.mat2euler(R, seq), seq)
            np_test.assert_array_almost_equal(R, R_dash)
        for _ in range(100):
            seq = 'xxx'
            while seq[0] == seq[1] or seq[1] == seq[2]:  # Euler angles cannot be consecutive
                seq = ''.join(random.choice('xyz') for _ in range(3))
            angles = list(random.random() for _ in range(3))
            angles_dash = spatial.mat2euler(spatial.euler2mat(angles, seq), seq)
            np_test.assert_array_almost_equal(angles, angles_dash)

    @staticmethod
    def test_projections():
        cartesian = (
            [spatial.cartesian2spherical, spatial.spherical2cartesian],
            [spatial.cartesian2cylindrical, spatial.cylindrical2cartesian]
        )
        rotation = ([spatial.rot_mat2axis_angle, spatial.axis_angle2rot_mat],)

        for _ in range(100):
            T = Transformation.random()
            translation = T[:3, 3]
            R = T[:3, :3]
            for _ in range(5):
                # Apply 5 random projections and their inverse
                c_func, c_inv_func = random.choice(cartesian)
                r_func, r_inv_func = random.choice(rotation)
                translation = c_inv_func(c_func(translation))
                R = r_inv_func(r_func(R))

            np_test.assert_array_almost_equal(translation, T[:3, 3])
            np_test.assert_array_almost_equal(R, T[:3, :3])

    def test_advanced_projections(self):
        for _ in range(100):
            T = Transformation.random()
            axis_angles = T.projection.axis_angles
            axis_angles3 = T.projection.axis_angles3
            vec = T.projection.roto_translation_vector
            np_test.assert_array_almost_equal(T.translation, vec[3:])
            np_test.assert_array_almost_equal(axis_angles3, vec[:3])
            np_test.assert_array_almost_equal(axis_angles3, axis_angles[:3] * axis_angles[3])

            T_dash = Transformation.from_roto_translation_vector(vec)
            self.assertEqual(T, T_dash)

            rot = spatial.axis_angle2rot_mat(axis_angles)
            T_dash = Transformation.from_roto_translation(rot, T.translation)
            self.assertEqual(T, T_dash)

        no_rotation = np.zeros((4,))
        no_rotation3 = np.zeros((3,))
        identity = Transformation.neutral()
        should_be_identity = spatial.axis_angle2rot_mat(no_rotation)
        np_test.assert_array_equal(should_be_identity, identity.rotation)
        np_test.assert_array_equal(identity.projection.axis_angles3, no_rotation3)
        np_test.assert_array_equal(identity.projection.axis_angles, no_rotation)

        should_be_identity = Transformation.from_roto_translation_vector(np.zeros(6,))
        self.assertEqual(should_be_identity, identity)

    @staticmethod
    def test_rotations():
        for _ in range(100):
            alpha = random.uniform(-np.pi, np.pi)
            beta = random.uniform(-np.pi, np.pi)
            gamma = random.uniform(-np.pi, np.pi)
            R = spatial.random_rotation()
            original = R.copy()
            rotations = list(f(angle)
                             for f, angles in zip(
                (spatial.rotX, spatial.rotY, spatial.rotZ),
                ([alpha, -alpha], [beta, -beta], [gamma, -gamma])
            ) for angle in angles)
            for manipulation in rotations:
                R = R @ manipulation[:3, :3]

            np_test.assert_array_almost_equal(R, original)

    def test_rotation_mapping(self):
        def in_bounds(arr: np.ndarray, bounds: np.ndarray) -> bool:
            return (bounds[0, :] <= arr).all() and (bounds[1, :] >= arr.all())

        for _ in range(1000):
            rot = np.random.randint(-100, 100, size=(1, 20)) * np.random.random((1, 20))
            limits = np.random.randint(-10, 10, size=(2, 20)) * np.random.random((2, 20))
            limits[1, :] = limits[0, :] + np.random.randint(0, 10, size=(1, 20))  # make sure upper bounds are higher
            projected = spatial.rotation_in_bounds(rot, limits)
            if in_bounds(rot, limits):
                np_test.assert_array_equal(rot, projected)
            elif not in_bounds(projected, limits):
                # The only valid reason to fail is that the projection is not possible
                self.assertTrue((limits[1, :] - limits[0, :] < 2 * np.pi).any())

    def test_skew_vector(self):
        for _ in range(50):
            v = np.random.random(3)
            x = np.random.random(3)
            M = spatial.skew(v)
            np_test.assert_array_almost_equal(M @ x, np.cross(v, x))
            np_test.assert_array_almost_equal(M, -M.T)


if __name__ == '__main__':
    unittest.main()
