import unittest

import numpy as np

from timor.Volume import BoxVolume, CylinderVolume, SphereVolume
from timor.utilities.transformation import Transformation


class TestVolumeProperties(unittest.TestCase):
    """Test different volume properties"""

    def setUp(self) -> None:
        np.random.seed(2022)

        self.r_default = 2
        self.dz_default = 2

        # initial full volumes
        self.sphere = SphereVolume(limits=np.array(((0, self.r_default), (0, np.pi), (-np.pi, np.pi))))
        self.cylinder = CylinderVolume(limits=np.array(((0, self.r_default), (-np.pi, np.pi),
                                                        (-self.dz_default, self.dz_default))))
        self.box = BoxVolume(limits=np.array((((-self.dz_default, self.dz_default),) * 3)))

        # initial two half-spheres, where s_1 is the inverse of s_2 w.r.t \theta
        self.s_1 = SphereVolume(limits=np.array(((0, self.r_default), (0, np.pi / 2), (-np.pi, np.pi))),
                                offset=Transformation.neutral())
        self.s_2 = SphereVolume(limits=np.array(((0, self.r_default), (np.pi / 2, np.pi), (-np.pi, np.pi))),
                                offset=Transformation.from_rotation(np.array(((1, 0, 0),
                                                                              (0, -1, 0),
                                                                              (0, 0, -1)))))
        # initial a sphere, which radius is halved
        self.s_3 = SphereVolume(limits=np.array(((0, 0.5 * self.r_default), (0, np.pi), (-np.pi, np.pi))))

        # initial a quarter sphere
        self.s_4 = SphereVolume(limits=np.array(((0, self.r_default), (0, np.pi / 2), (0, np.pi))))

        # initial two half-cylinders, where c_1 is the inverse of c_2 w.r.t \phi
        self.c_1 = CylinderVolume(limits=np.array(((0, self.r_default), (-np.pi, 0),
                                                   (-self.dz_default, self.dz_default))))
        self.c_2 = CylinderVolume(limits=np.array(((0, self.r_default), (0, np.pi),
                                                   (-self.dz_default, self.dz_default))),
                                  offset=Transformation.from_rotation(np.array(((1, 0, 0),
                                                                                (0, -1, 0),
                                                                                (0, 0, -1)))))

    def test_actual_volume(self):
        """Test the formular of the actual volume"""
        self.assertAlmostEqual(4 / 3 * np.pi * self.r_default ** 3, self.sphere.vol, places=5)
        self.assertAlmostEqual(4 * np.pi * self.r_default ** 2, self.cylinder.vol, places=5)
        self.assertAlmostEqual((2 * self.dz_default) ** 3, self.box.vol)
        self.assertEqual(self.s_1.vol, self.s_2.vol)
        self.assertEqual(self.sphere.vol, 8 * self.s_3.vol)
        self.assertEqual(self.sphere.vol, 4 * self.s_4.vol)
        self.assertEqual(self.c_1.vol, self.c_2.vol)

    def test_contains(self):
        """Test whether a given point is contained in the volume"""
        zero = np.zeros(3)
        x_max = np.array((self.dz_default, 0, 0))
        y_max = np.array((0, self.dz_default, 0))
        z_max = np.array((0, 0, self.dz_default))
        xz_max = np.array((self.dz_default, 0, self.dz_default))
        xyz_max = np.array((self.dz_default, self.dz_default, self.dz_default))

        # test sphere
        self.assertTrue(self.sphere.contains(zero))  # test center of the sphere
        self.assertTrue(self.sphere.contains(x_max))  # test the border point in x-direction
        self.assertTrue(self.sphere.contains(y_max))  # test the border point in y-direction
        self.assertTrue(self.sphere.contains(z_max))  # test the border point in z-direction
        self.assertFalse(self.sphere.contains(xyz_max))  # test point outside the sphere

        # test cylinder
        self.assertTrue(self.cylinder.contains(zero))  # test center of the cylinder
        self.assertTrue(self.cylinder.contains(x_max))  # test border points
        self.assertTrue(self.cylinder.contains(xz_max))  # test border points
        self.assertFalse(self.cylinder.contains(xyz_max))  # test point outside the cylinder

        # test box
        self.assertTrue(self.box.contains(zero))  # test center of the box
        self.assertTrue(self.box.contains(x_max))  # test border points
        self.assertTrue(self.box.contains(x_max + y_max + z_max))  # test border points
        self.assertFalse(self.box.contains(1.01 * x_max))  # test points outside the box

        # test partial sphere
        for _ in range(50):
            point = np.random.random((3,)) * self.r_default
            self.assertEqual(self.s_1.contains(point), self.s_2.contains(point))
            self.assertEqual(self.c_1.contains(point), self.c_2.contains(point))

    def test_sample_uniform(self):
        """Test uniform sampler"""
        num_samples = 10000
        samples = {'sphere': [], 'cylinder': [], 'box': []}
        in_partial = 0
        for _ in range(num_samples):
            s, c, b = self.sphere.sample_uniform(), self.cylinder.sample_uniform(), self.box.sample_uniform()
            samples['sphere'].append(s)
            samples['cylinder'].append(c)
            samples['box'].append(b)
            self.assertTrue(self.sphere.contains(s))
            self.assertTrue(self.cylinder.contains(c))
            self.assertTrue(self.box.contains(b))

            if self.s_1.contains(s):
                self.assertTrue(self.s_2.contains(s))
                in_partial += 1

        # Some basic checks on uniformity
        self.assertAlmostEqual(float(np.mean(samples['sphere'])), 0, places=1)
        self.assertAlmostEqual(float(np.mean(samples['cylinder'])), 0, places=1)
        self.assertAlmostEqual(float(np.mean(samples['box'])), 0, places=1)
        self.assertAlmostEqual(self.s_1.vol / self.sphere.vol, in_partial / num_samples, places=1)

    def test_sample_volume_grid(self):
        """Test grid sampler"""
        grid_size = (25, 25, 25)
        # Volume samples correct number of grid points.
        for vol in self.sphere, self.cylinder, self.box:
            samples = vol.discretize(grid_size=grid_size)
            vol_hypercuboid = np.prod(vol.limit_rectangular_cuboid[:, 1] - vol.limit_rectangular_cuboid[:, 0])
            self.assertAlmostEqual(vol.vol / vol_hypercuboid, (~np.isnan(samples)).sum() / samples.size, places=2)

            for idx in np.ndindex(samples.shape[:3]):
                sample = samples[idx]
                if np.isnan(sample).any():
                    self.assertTrue(np.isnan(sample).all())
                    continue
                self.assertTrue(vol.contains(sample))


if __name__ == '__main__':
    unittest.main()
