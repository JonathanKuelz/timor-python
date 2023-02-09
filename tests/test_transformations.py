import math
from random import random
import unittest

import numpy as np
import numpy.testing as np_test

from timor.utilities import spatial
from timor.utilities.transformation import Transformation


class TestSpatialTransformations(unittest.TestCase):
    """Tests the capabilities of roto-translation manipulations (Transformations)"""

    def setUp(self) -> None:
        np.random.seed(85748)
        rand = np.eye(4)
        rand[:3, 3] = np.random.rand(3)
        rand[:3, :3] = spatial.random_rotation()
        self.random_homogeneous = rand
        self.random_transformation = Transformation(self.random_homogeneous)

    def test_magic_methods_and_properties(self):
        """Tests behavior of the Transformation class when magic methods are used"""

        np_test.assert_array_equal(self.random_homogeneous, self.random_transformation.homogeneous)
        np_test.assert_array_equal(self.random_transformation[:4, :4],
                                   self.random_homogeneous[:4, :4])  # np-like indexing

        for _ in range(20):
            # Equality operator and inverse
            t = Transformation.random()
            self.assertEqual(t, t.inv.inv)

    def test_multiplication(self):
        """Makes sure multiplication of homogeneous transformations work as expected"""

        np_test.assert_array_almost_equal(self.random_homogeneous, self.random_transformation.inv.inv.homogeneous)
        translation = Transformation.from_translation(self.random_homogeneous[:3, 3])
        rotation = Transformation.from_rotation(self.random_homogeneous[:3, :3])
        roto_translation = translation @ rotation
        self.assertEqual(roto_translation, self.random_transformation)

        # A Transformation can also be given as a sequence of transformations
        sequence = [Transformation.random() for _ in range(10)]

        final_transform = Transformation.neutral()
        for t in sequence:
            final_transform = final_transform @ t

        self.assertEqual(final_transform, Transformation(sequence))

    def test_transformation_distance(self):
        """Test distance for transformations."""
        # combined
        T = Transformation.from_translation((1, 1, 1)) @ Transformation(spatial.rotX(math.pi / 2))
        self.assertAlmostEqual(T.distance(Transformation.neutral()).translation_euclidean, math.sqrt(3))
        self.assertAlmostEqual(T.distance(Transformation.neutral()).rotation_angle, math.pi / 2)

        for _ in range(100):  # Test distance symmetry and unity
            T1 = Transformation.random()
            T2 = Transformation.random()
            dT = T1.inv @ T2
            self.assertAlmostEqual(T1.distance(T1).translation_euclidean, 0.)
            self.assertAlmostEqual(T1.distance(T1).rotation_angle, 0.)
            self.assertAlmostEqual(T1.distance(T2).translation_euclidean, T2.distance(T1).translation_euclidean)
            self.assertAlmostEqual(T1.distance(T2).rotation_angle, T2.distance(T1).rotation_angle)
            self.assertAlmostEqual(T1.distance(T2).translation_euclidean, dT.norm.translation_euclidean)
            self.assertAlmostEqual(T1.distance(T2).rotation_angle, dT.norm.rotation_angle)

        # Test 180° rotation
        T1 = Transformation.neutral()
        T2 = Transformation.from_rotation(spatial.rotX(math.pi)[:3, :3])
        T3 = Transformation.from_rotation(spatial.rotY(-math.pi)[:3, :3])
        T4 = Transformation.from_rotation(spatial.rotZ(math.pi + 1e-9)[:3, :3])
        self.assertAlmostEqual(T1.distance(T2).rotation_angle, math.pi)
        self.assertAlmostEqual(T1.distance(T2).rotation_angle, math.pi)
        self.assertAlmostEqual(T1.distance(T3).rotation_angle, math.pi + 1e-9)

    def test_transformation_interpolate(self):
        """Test interpolation between transformations."""
        start = Transformation.neutral()
        end_trans = Transformation.from_translation((1, 1, 1))
        end_rot = Transformation(spatial.rotX(math.pi / 2))
        for i in np.arange(0, 1, 0.1):
            # translation
            self.assertEqual(start.interpolate(end_trans, i),
                             Transformation.from_translation((i, i, i)))
            # rotation
            self.assertEqual(start.interpolate(end_rot, i),
                             Transformation(spatial.rotX(i * math.pi / 2)))
            # combined
            T = start.interpolate(end_trans @ end_rot, i)

        for _ in range(10):
            start = Transformation.random()
            end = Transformation.random()
            self.assertEqual(start.interpolate(end, 0), start)
            self.assertEqual(start.interpolate(end, 1), end)

    def test_transformation_norm(self):
        self.assertAlmostEqual(Transformation.neutral().norm.translation_euclidean, 0.0)
        self.assertAlmostEqual(Transformation.neutral().norm.rotation_angle, 0.0)
        for _ in range(100):
            # Translation only
            translation = np.random.random((3,))
            self.assertAlmostEqual(Transformation.from_translation(translation).norm.translation_euclidean,
                                   np.linalg.norm(translation),
                                   msg=f"Translation was {translation}")
            self.assertAlmostEqual(Transformation.from_translation(translation).norm.translation_maximum,
                                   np.max(np.abs(translation)),
                                   msg=f"Translation was {translation}")
            self.assertAlmostEqual(Transformation.from_translation(translation).norm.rotation_angle, 0.0)
            # Rotation only
            rot = math.pi * (random() - 0.5)
            self.assertAlmostEqual(Transformation(spatial.rotX(rot)).norm.translation_euclidean, 0.0,
                                   msg=f"Rotation angle x was {rot}")
            self.assertAlmostEqual(Transformation(spatial.rotX(rot)).norm.rotation_angle, abs(rot),
                                   msg=f"Rotation angle x was {rot}")
            self.assertAlmostEqual(Transformation(spatial.rotY(rot)).norm.translation_euclidean, 0.0,
                                   msg=f"Rotation angle y was {rot}")
            self.assertAlmostEqual(Transformation(spatial.rotY(rot)).norm.rotation_angle, abs(rot),
                                   msg=f"Rotation angle y was {rot}")
            self.assertAlmostEqual(Transformation(spatial.rotZ(rot)).norm.translation_euclidean, 0.0,
                                   msg=f"Rotation angle z was {rot}")
            self.assertAlmostEqual(Transformation(spatial.rotZ(rot)).norm.rotation_angle, abs(rot),
                                   msg=f"Rotation angle z was {rot}")
            # Composite
            T = Transformation.from_translation(translation) @ spatial.rotX(rot)
            self.assertAlmostEqual(T.norm.translation_euclidean, np.linalg.norm(translation))
            self.assertAlmostEqual(T.norm.rotation_angle, abs(rot))
            T = Transformation.from_translation(translation) @ spatial.rotY(rot)
            self.assertAlmostEqual(T.norm.translation_euclidean, np.linalg.norm(translation))
            self.assertAlmostEqual(T.norm.rotation_angle, abs(rot))
            T = Transformation.from_translation(translation) @ spatial.rotZ(rot)
            self.assertAlmostEqual(T.norm.translation_euclidean, np.linalg.norm(translation))
            self.assertAlmostEqual(T.norm.rotation_angle, abs(rot))

            self.assertLessEqual(T.norm.translation_euclidean, T.norm.translation_manhattan)
            self.assertAlmostEqual(T.norm.rotation_angle * 180 / np.pi, T.norm.rotation_angle_degree)


if __name__ == '__main__':
    unittest.main()
