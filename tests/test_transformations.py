import unittest

import numpy as np
import numpy.testing as np_test

from timor.utilities import spatial
from timor.utilities.transformation import Transformation


class TestSpatialTransformations(unittest.TestCase):
    """Tests the capabilities of roto-translation manipulations (Transformations)"""

    def setUp(self) -> None:
        np.random.seed(85748)
        self.random_homogeneous = spatial.random_homogeneous()
        self.random_transformation = Transformation(self.random_homogeneous)

    def test_magic_methods_and_properties(self):
        """Tests behavior of the Transformation class when magic methods are used"""

        np_test.assert_array_equal(self.random_homogeneous, self.random_transformation.homogeneous)
        np_test.assert_array_equal(self.random_transformation[:4, :4],
                                   self.random_homogeneous[:4, :4])  # np-like indexing
        self.assertTrue(self.random_transformation == self.random_transformation.inv.inv)  # equality operator

    def test_multiplication(self):
        """Makes sure multiplication of homogeneous transformations work as expected"""

        np_test.assert_array_almost_equal(self.random_homogeneous, self.random_transformation.inv.inv.homogeneous)
        translation = Transformation.from_translation(self.random_homogeneous[:3, 3])
        rotation = Transformation.from_rotation(self.random_homogeneous[:3, :3])
        roto_translation = translation @ rotation
        self.assertEqual(roto_translation, self.random_transformation)

        # A Transformation can also be given as a sequence of transformations
        sequence = [spatial.random_homogeneous() for _ in range(10)]

        final_transform = Transformation.neutral()
        for t in sequence:
            final_transform = final_transform @ t

        self.assertEqual(final_transform, Transformation(sequence))


if __name__ == '__main__':
    unittest.main()
