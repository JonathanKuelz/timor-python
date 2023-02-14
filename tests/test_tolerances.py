import abc
from copy import deepcopy
import inspect
import random
import unittest

import numpy as np
import numpy.testing as np_test
from pinocchio.visualize import MeshcatVisualizer

from timor.task import Tolerance
from timor.utilities import spatial
from timor.utilities.transformation import Transformation


class TestTolerances(unittest.TestCase):
    """Tests all kinds of tolerances"""

    def test_asymmetric_spatial_tolerances(self):
        xyz = Tolerance.CartesianXYZ([-.5, 0], [-1, .5], [-.35, 100])
        rot = Tolerance.RotationAxisAngle([-1, 1], [-1, 1], [0, 1], [0, np.pi])
        desired = Transformation.random()

        valid_offsets = (
            spatial.homogeneous(translation=[-.4, 0, 0]),
            spatial.homogeneous(translation=[0, 0, 99]),
            spatial.homogeneous(rotation=spatial.rotZ(0.9 * np.pi)[:3, :3]),
            spatial.homogeneous(rotation=spatial.rotZ(-1e-14)[:3, :3]),
            spatial.homogeneous(rotation=spatial.rotX(0.9 * np.pi)[:3, :3]),
        )
        invalid_offsets = (
            spatial.homogeneous(translation=[.4, 0, 0]),
            spatial.homogeneous(translation=[0, 0, -1]),
            spatial.homogeneous(rotation=spatial.rotZ(-0.9 * np.pi)[:3, :3]),
        )

        tolerance = xyz + rot

        for offset in valid_offsets:
            self.assertTrue(tolerance.valid(desired, desired @ offset))
        for offset in invalid_offsets:
            self.assertFalse(tolerance.valid(desired, desired @ offset))

        # Check negative theta works for rotation tolerance:
        rot = Tolerance.RotationAxisAngle([-1, 1], [-1, 1], [0, 1], [-np.pi, 0])
        offset = spatial.homogeneous(rotation=spatial.rotZ(-0.9 * np.pi)[:3, :3])
        self.assertTrue(rot.valid(desired, desired @ offset))

    def test_tolerance_combinations(self):
        cartesian_large = Tolerance.CartesianXYZ([-.5, .5], [-1, 1], [-.35, .35])
        cartesian_medium = Tolerance.CartesianXYZ([-.4, .4], [-.75, .75], [-.5, .5])
        cartesian_small = Tolerance.CartesianXYZ(*[[-1e-4, 1e-4]] * 3)

        # Make sure "deprecated" scalar and one-sided tolerances fail
        with self.assertRaises(ValueError):
            tol = Tolerance.CartesianXYZ(.3, .3, .3)
        with self.assertRaises(TypeError):
            tol = Tolerance.CartesianXYZ([.1, .1, .1])

        with self.assertRaises(TypeError):
            new_tolerance = cartesian_small + 42

        np_test.assert_array_equal((cartesian_large + cartesian_large).stacked, cartesian_large.stacked)
        np_test.assert_array_equal((cartesian_small + cartesian_medium).stacked, cartesian_small.stacked)

        goal = Transformation.from_translation(np.array([1, 1, 1], dtype=float))
        self.assertTrue(cartesian_large.valid(goal, Transformation.from_translation([.6, 1.5, 1.2])))
        self.assertFalse(cartesian_small.valid(goal, Transformation.from_translation([1.1, 1, 1])))
        self.assertFalse(cartesian_medium.valid(goal, Transformation.from_translation([0, -.5, 1.5])))
        self.assertFalse(
            (cartesian_large + cartesian_small).valid(goal, Transformation.from_translation([0, -.5, 1.5])))
        self.assertFalse((cartesian_large + cartesian_medium + cartesian_small)
                         .valid(goal, Transformation.from_translation(np.array([0, -.5, 1.5]))))

    def test_tolerance_types(self):
        # Get all classes defined in the Tolerance namespace

        available_tolerances = [ret[1] for ret in inspect.getmembers(Tolerance, inspect.isclass)]
        for tol_class in available_tolerances:
            if not type(tol_class) is abc.ABCMeta or inspect.isabstract(tol_class):
                continue
            if tol_class is Tolerance.Composed:
                continue

            # Make sure all default methods are running properly
            default = tol_class.default()

            # Make sure lower tolerances must be smaller than the upper ones
            default_dict = default.__dict__
            if len(default_dict) > 0 \
                    and all(isinstance(tol_val, np.ndarray) and tol_val.size == 2 for tol_val in default_dict.values()):
                # We have a tolerance with two values
                reverse = deepcopy(default_dict)
                for key, val in default_dict.items():
                    # Change tolerance values
                    reverse[key][0] = val[1]
                    reverse[key][1] = val[0]
                with self.assertRaises(ValueError):
                    reverse_tolerance = tol_class(*reverse.values())

            # Make sure adding the same tolerance twice doesn't alter it
            double_default = default + default
            self.assertTrue(all(np.all(  # np.all accepts "True" even if it is not iterable
                default.__dict__[key] == double_default.__dict__[key]) for key in default.__dict__))

            placement = Transformation.random()

            if isinstance(default, (Tolerance.Cartesian, Tolerance.Rotation)):
                # Check that the valid random deviation is indeed valid
                deviation = default.valid_random_deviation
                self.assertTrue(default.valid(placement, placement @ deviation))

                # Check the tolerance projection and instantiation from projection
                projection_data = default.to_projection()
                copy_from_specs = Tolerance.AlwaysValidTolerance()
                for projection, tol in zip(projection_data['toleranceProjection'], projection_data['tolerance']):
                    copy_from_specs += Tolerance.ToleranceBase.from_projection(projection, tol)
                self.assertIs(default.__class__, copy_from_specs.__class__)
                np_test.assert_array_equal(default.stacked, copy_from_specs.stacked)
                self.assertEqual(default, copy_from_specs)

    def test_tolerance_reference_frame(self):
        # Defines a narrow, long cylinder where only rotation around z-axis is really allowed. Could be drilling hole.

        cart = Tolerance.CartesianCylindrical(r=(0, .1), z=(-.01, 2.5))
        rot = Tolerance.RotationAxisAngle(n_x=(-.05, .05), n_y=(-.05, .05), n_z=(-1, 1), theta_r=(0, np.pi))
        tolerance = cart + rot

        t_desired = Transformation(spatial.homogeneous(translation=(1, 1, 2),
                                                       rotation=spatial.rotX(-np.pi / 2)[:3, :3]))

        valid_rot_deviations = [spatial.rotZ(np.pi * random.random()) for _ in range(100)]
        valid_trans_deviation = spatial.homogeneous(translation=(.01, .03, 2))
        valid_deviations = (*valid_rot_deviations, valid_trans_deviation,
                            *(valid_trans_deviation @ vrd for vrd in valid_rot_deviations))

        for deviation in valid_deviations:
            new_pose = t_desired @ deviation
            self.assertTrue(tolerance.valid(t_desired, new_pose))

        # Also add a manual sanity check
        valid_pose = Transformation([
            [1, 0, 0, 1],
            [0, 0, 1, 2.5],
            [0, -1, 0, 2],
            [0, 0, 0, 1]
        ])
        self.assertTrue(tolerance.valid(t_desired, valid_pose))

        failing = [Transformation.from_translation((1 + .11, 1, 2)),
                   Transformation.from_translation((1 + .08, 1 + .08, 2)),
                   t_desired @ Transformation(spatial.rotX(.1)), t_desired @ Transformation(spatial.rotY(.1))]
        for transform in failing:
            self.assertFalse(tolerance.valid(t_desired, transform))

    def test_visualization(self):
        """This unittest is running the visual utilities, but it only tests that the code runs without crashing.

        For correctness of visualization, human inspection is necessary.
        """
        viz = MeshcatVisualizer()
        viz.initViewer()
        box = Tolerance.CartesianXYZ(x=(0, .7), y=(0, .7), z=(0, .5))
        cylinder = Tolerance.CartesianCylindrical(r=(0, 1 / 2), z=(-.3, .3))
        partial_cylinder = Tolerance.CartesianCylindrical(r=(0, 1 / 2), z=(-.1, .1), phi_cyl=(np.pi / 4, np.pi / 2))
        sphere = Tolerance.CartesianSpheric(r=(0, .33))
        partial_sphere = Tolerance.CartesianSpheric(r=(0, .33), theta=(0, np.pi / 2), phi_sph=(-np.pi / 4, np.pi / 2))

        composed = Tolerance.Composed([
            Tolerance.CartesianXYZ(x=(0, .1), y=(0, .1), z=(0, .1)),
            Tolerance.CartesianSpheric(r=(0, .1)),
        ])

        tolerances = (box, cylinder, sphere, partial_cylinder, partial_sphere, composed)
        offsets = [Transformation.from_translation((0, y, 0)) for y in range(len(tolerances))]

        for i, (tol, offset) in enumerate(zip(tolerances, offsets)):
            tol.visualize(viz, offset.homogeneous, name='test_tolerance_{}'.format(i))


if __name__ == '__main__':
    unittest.main()
