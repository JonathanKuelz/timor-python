import unittest

import itertools
import numpy as np

from timor.Volume import Volume, BoxVolume, CylinderVolume, SphereVolume
from timor.utilities.transformation import Transformation


def is_convex_sampling(num_samples: int, num_interpolation_steps: int, rng: np.random.Generator, vol: Volume) -> bool:
    """
    Check whether the volume is convex based on sampling approach.

    :param num_samples: The number of uniformly sampled points within the volume.
    :param num_interpolation_steps: The number of intermediate points between two samples to check.
    :param rng: Random number generator.
    :param vol: The volume instance to check.
    :return: True if the volume is convex, False otherwise.
    """
    samples = []
    steps = np.linspace(0, 1, num_interpolation_steps)
    for _ in range(num_samples):
        samples.append(vol.sample_uniform(rng))

    for (i, j) in itertools.combinations(range(num_samples), 2):
        for step in steps:
            if not vol.contains(samples[i] * step + samples[j] * (1 - step)):
                return False
    return True


def random_limit(min_val, max_val, rng: np.random.Generator):
    """
    Generate a random limit tuple (lower_limit, upper_limit). Serves as a helper function.

    :param min_val: Minimum possible value for the limits.
    :param max_val: Maximum possible value for the limits.
    :param rng: Random number generator.
    :return: Tuple (lower_limit, upper_limit)
    """
    random_vals = rng.random((2,)) * (max_val - min_val) + min_val
    lower_limit, upper_limit = np.sort(random_vals)
    return lower_limit, upper_limit


class TestVolumeProperties(unittest.TestCase):
    """Test different volume properties"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
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
        # initial partial cylinders, which should not be convex
        self.c_3_4 = CylinderVolume(limits=np.array(((0, self.r_default), (-np.pi, np.pi / 2),
                                                     (-self.dz_default, self.dz_default))))
        self.c_2_3 = CylinderVolume(limits=np.array(((0, self.r_default), (-np.pi, np.pi / 3),
                                                     (-self.dz_default, self.dz_default))))

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
            point = self.rng.random((3,)) * self.r_default
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

    def test_convexity(self):
        """Testing the convexity of the volume."""
        num_samples_per_volume = 20
        num_interpolation_steps = 20
        num_random_tests = 15

        for vol in (self.sphere, self.cylinder, self.box, self.s_1, self.s_2, self.s_3, self.s_4, self.c_1, self.c_2,
                    self.c_3_4, self.c_2_3):
            samples_are_convex = is_convex_sampling(num_samples_per_volume, num_interpolation_steps, self.rng, vol)
            if not samples_are_convex:
                self.assertFalse(vol.is_convex())

        # Randomized checks
        def random_sphere() -> SphereVolume:
            """Generate a random sphere volume."""
            lim_r = random_limit(0, self.r_default, self.rng)
            lim_theta = random_limit(0, np.pi, self.rng)
            lim_phi = random_limit(-np.pi, np.pi, self.rng)
            return SphereVolume(limits=np.array((lim_r, lim_theta, lim_phi)))

        def random_cylinder() -> CylinderVolume:
            """Generate a random cylinder volume."""
            lim_r = random_limit(0, self.r_default, self.rng)
            lim_phi = random_limit(-np.pi, np.pi, self.rng)
            dz = self.rng.uniform(0.1, self.dz_default)
            return CylinderVolume(limits=np.array((lim_r, lim_phi, (-dz, dz))))

        def random_box() -> BoxVolume:
            """Generate a random box volume."""
            x = self.rng.uniform(0.1, self.dz_default)
            y = self.rng.uniform(0.1, self.dz_default)
            z = self.rng.uniform(0.1, self.dz_default)
            return BoxVolume(limits=np.array(((-x, x), (-y, y), (-z, z))))

        for constructor in (random_sphere, random_cylinder, random_box):
            for _ in range(num_random_tests):
                volume = constructor()

                # We can't guarantee that a volume is convex just by sampling, but we can falsify convexity.
                samples_are_convex = is_convex_sampling(num_samples_per_volume,
                                                        num_interpolation_steps, self.rng, volume)
                if not samples_are_convex:
                    self.assertFalse(volume.is_convex())


if __name__ == '__main__':
    unittest.main()
