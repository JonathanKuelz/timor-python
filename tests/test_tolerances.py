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
from timor.utilities.frames import WORLD_FRAME
from timor.utilities.transformation import Transformation


class TestTolerances(unittest.TestCase):
    """Tests all kinds of tolerances"""

    def setUp(self):
        self.rng = np.random.default_rng(123456789)

    def test_asymmetric_spatial_tolerances(self, n: int = 5):
        for _ in range(n):
            xyz = Tolerance.CartesianXYZ([-.5, 0], [-1, .5], [-.35, 100])
            rot = Tolerance.RotationAxis(n_x=(-np.pi / 10, np.pi), n_y=(-np.pi / 2, np.pi / 2),
                                         n_z=(-np.pi, np.pi / 10))
            desired = Transformation.random()

            valid_offsets = (
                spatial.homogeneous(translation=[-.4, 0, 0]),
                spatial.homogeneous(translation=[0, 0, 99]),
                spatial.homogeneous(rotation=spatial.rotZ(np.pi / 11)[:3, :3]),
                spatial.homogeneous(rotation=spatial.rotZ(-np.pi / 2)[:3, :3]),
                spatial.homogeneous(rotation=spatial.rotY(np.pi / 3)[:3, :3]),
                spatial.homogeneous(rotation=(spatial.rotX(0.3 * np.pi) @ spatial.rotX(0.1 * np.pi))[:3, :3]),
            )
            invalid_offsets = (
                spatial.homogeneous(translation=[.4, 0, 0]),
                spatial.homogeneous(translation=[0, 0, -1]),
                spatial.homogeneous(rotation=spatial.rotZ(0.9 * np.pi)[:3, :3]),
                Transformation.from_roto_translation_vector([-np.sqrt(1 / 2), np.sqrt(1 / 2), 0, 0, 0, 0]),
                Transformation.from_roto_translation_vector([-.5, 0, -np.sqrt(.75), 0, 0, 0]),
            )

            tolerance = xyz + rot

            for offset in valid_offsets:
                self.assertTrue(tolerance.valid(desired, desired @ offset))
            for offset in invalid_offsets:
                self.assertFalse(tolerance.valid(desired, desired @ offset))

    def test_rotation_tolerance_edge_cases(self, n_tries=100):
        desired = Transformation.random(rng=self.rng)
        no_rotation = Transformation.from_translation(self.rng.random((3,)))

        allow_all_angles = Tolerance.RotationAbsolute(np.pi)
        allow_all_axes = Tolerance.RotationAxis(n_x=(-np.pi, np.pi), n_y=(-np.pi, np.pi), n_z=(-np.pi, np.pi))
        discard_all_angles = Tolerance.RotationAbsolute(0)
        discard_all_axes = Tolerance.RotationAxis(n_x=(0, 0), n_y=(0, 0), n_z=(0, 0))
        super_tolerance = allow_all_angles + allow_all_axes + discard_all_angles + discard_all_axes
        tolerances = (allow_all_angles, allow_all_axes, discard_all_angles, discard_all_axes, super_tolerance)

        for tol in tolerances:
            self.assertTrue(tol.valid(desired, desired))
            self.assertTrue(tol.valid(desired, desired @ no_rotation))

        for _ in range(n_tries):
            rotation = Transformation.from_rotation(spatial.random_rotation(rng=self.rng))
            self.assertTrue(allow_all_angles.valid(desired, desired @ rotation))
            self.assertTrue(allow_all_axes.valid(desired, desired @ rotation))
            self.assertFalse(discard_all_angles.valid(desired, desired @ rotation))
            self.assertFalse(discard_all_axes.valid(desired, desired @ rotation))
            self.assertFalse(super_tolerance.valid(desired, desired @ rotation))

        allows_rotation_in_plane = Tolerance.RotationAxis(n_x=(-np.pi, np.pi), n_y=(-np.pi, np.pi), n_z=(0, 0))
        for _ in range(n_tries):
            diff_x = desired @ spatial.rotX(self.rng.random())
            diff_y = desired @ spatial.rotY(self.rng.random())
            diff_z = desired @ spatial.rotZ(self.rng.random())
            self.assertTrue(allows_rotation_in_plane.valid(desired, diff_x))
            self.assertTrue(allows_rotation_in_plane.valid(desired, diff_y))
            self.assertFalse(allows_rotation_in_plane.valid(desired, diff_z))

    def test_rotation_tolerance_use_cases(self):
        """
        Test the different modi of the Rotation tolerances:

        In the 3-dimensional vector space of possible rotations, we can constrain all valid ones either by:

        - an absolute angle (RotationAbsolute), which defines a sphere around the identity rotation
        - a scaled rotation axis (RotationAxis), which defines a rectangle around the identity rotation
        - an intersection of these two.
        """
        small_angle = np.pi / 180  # 1 Degree should be fine for many use cases
        half_degree = np.pi / 360
        five_degrees = np.pi / 36
        # Imagine different use cases
        arbitrary_but_accurate = Tolerance.RotationAbsolute(small_angle)
        drilling = Tolerance.RotationAxis(n_x=(-small_angle, small_angle), n_y=(-small_angle, small_angle), n_z=(-1, 1))
        surgery = Tolerance.RotationAxis(n_x=(0, 0), n_y=(0, 0), n_z=(-1, 1))  # assuming super precision
        avoid_roll = Tolerance.RotationAxis(n_x=(-small_angle, small_angle), n_y=(-1, 1), n_z=(-1, 1),
                                            frame=WORLD_FRAME)

        desired = Transformation.random(rng=self.rng)
        random_axis = np.cross(self.rng.random((3,)), np.array([0, 0, 1]))  # Avoid randomly getting a z-rotation
        half_degree_away = desired @ Transformation.from_rotation(spatial.axis_angle2rot_mat(
            np.concatenate((random_axis / np.linalg.norm(random_axis), [half_degree]))))
        five_degrees_away = desired @ Transformation.from_rotation(spatial.axis_angle2rot_mat(
            np.concatenate((random_axis / np.linalg.norm(random_axis), [five_degrees]))))

        rot_by_five_x = spatial.rotX(five_degrees)
        rot_by_five_z = spatial.rotZ(five_degrees)
        five_degrees_x = desired @ rot_by_five_x
        five_degrees_z = desired @ rot_by_five_z

        self.assertTrue(arbitrary_but_accurate.valid(desired, half_degree_away))
        self.assertFalse(arbitrary_but_accurate.valid(desired, five_degrees_away))

        self.assertTrue(drilling.valid(desired, half_degree_away))
        self.assertTrue(drilling.valid(desired, five_degrees_z))
        self.assertFalse(drilling.valid(desired, five_degrees_away))
        self.assertFalse(drilling.valid(desired, five_degrees_x))

        self.assertTrue(surgery.valid(desired, five_degrees_z))
        self.assertFalse(surgery.valid(desired, half_degree_away))

        roll = Transformation(rot_by_five_x) @ desired
        no_roll = Transformation(rot_by_five_z) @ desired
        self.assertTrue(avoid_roll.valid(desired, no_roll))
        self.assertFalse(avoid_roll.valid(desired, roll))

        # A rotation of 5 Degrees around the x-axis, then the z-axis still is not more than 5 Degrees around the initial
        # x-axis. It is a rotation of more than five degree, though
        rot_abs = Tolerance.RotationAbsolute(five_degrees)
        rot_axis_wise = Tolerance.RotationAxis(n_x=(-five_degrees, five_degrees), n_y=(-five_degrees, five_degrees),
                                               n_z=(-five_degrees, five_degrees))
        rot_xz = desired @ rot_by_five_x @ rot_by_five_z
        self.assertFalse(rot_abs.valid(desired, rot_xz))
        self.assertTrue(rot_axis_wise.valid(desired, rot_xz))

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

        composed = Tolerance.Composed((cartesian_large, cartesian_medium, cartesian_small))
        copy = composed + composed
        self.assertEqual(composed, copy)

    def test_tolerance_types(self):
        # Get all classes defined in the Tolerance namespace
        available_tolerances = [ret[1] for ret in inspect.getmembers(Tolerance, inspect.isclass)]
        for tol_class in available_tolerances:
            if not type(tol_class) is abc.ABCMeta or inspect.isabstract(tol_class):
                continue
            if tol_class is Tolerance.Composed:
                continue
            if not issubclass(tol_class, Tolerance.ToleranceBase):
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
        eps_rot = np.pi / 180
        cart = Tolerance.CartesianCylindrical(r=(0, .1), z=(-.01, 2.5))
        rot = Tolerance.RotationAxis(n_x=(-eps_rot, eps_rot), n_y=(-eps_rot, eps_rot), n_z=(-np.pi, np.pi))
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
