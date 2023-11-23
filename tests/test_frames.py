import json
import random
import unittest

import numpy as np

from timor.task import Tolerance
from timor.task.Task import Task
from timor.utilities import spatial
from timor.utilities.file_locations import get_test_tasks
from timor.utilities.spatial import rotX
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.transformation import Transformation
from timor.utilities.frames import Frame, FrameTree, NOMINAL_FRAME, WORLD_FRAME


class TestFrames(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)
        np.random.seed(0)
        self.data = {
            "1": {"transformation": Transformation.random().homogeneous.round(10).tolist(), "parent": "world"},
            "2": {"transformation": Transformation.random().homogeneous.round(10).tolist(), "parent": "1"},
            "3": {"transformation": Transformation.random().homogeneous.round(10).tolist(), "parent": "2"},
            "2a": {"transformation": Transformation.random().homogeneous.round(10).tolist(), "parent": "1"},
            "3a": {"transformation": Transformation.random().homogeneous.round(10).tolist(), "parent": "2a"}
        }  # keep to list such that compatible with bare json data

    def test_frame_tree_transformations(self):
        ft = FrameTree(self.data)
        # Check world
        with self.assertRaises(AttributeError):
            p2f = ft["world"].parent2frame  # noqa: F841
        with self.assertRaises(AttributeError):
            fp = ft["world"].parent  # noqa: F841
        # Check other frames
        for name, frame in self.data.items():
            self.assertTrue(name in ft)
            self.assertEqual(ft[name].parent2frame, Transformation(frame["transformation"]))
            self.assertEqual(ft[name].parent, ft[frame["parent"]])
            # Check world_to_frame consistent with parent
            self.assertEqual(ft[name].in_world_coordinates(),
                             ft[frame["parent"]].in_world_coordinates() @ ft[name].parent2frame)

    def test_frame_tree_add(self):
        ft = FrameTree(self.data)
        # Check world
        with self.assertRaises(ValueError):
            ft.add("world", Transformation.random(), None)
        # Check other frames
        for name, frame in self.data.items():
            with self.assertRaises(ValueError):
                ft.add(name, Transformation.random(), WORLD_FRAME)
        # Check new frame
        ft.add("new", Transformation.random(), "world")
        self.assertIn("new", ft)
        self.assertIn("new", ft.to_json_data())
        ft.add("special.new", Transformation.random(), "world")
        self.assertIn("special.new", ft)
        self.assertNotIn("special.new", ft.to_json_data())

    def test_matmul(self):
        ft = FrameTree(self.data)
        for frame in ft:
            T = Transformation.random()
            self.assertEqual(frame @ T, frame.absolute @ T)

    def test_eq_frame(self):
        ft1 = FrameTree(self.data)
        ft2 = FrameTree(self.data)
        for frame in ft1:
            self.assertEqual(frame, ft2[frame.ID])
            self.assertEqual(frame, frame)
        self.assertEqual(ft1["1"], ft2["1"])  # Frames in different trees can be equal if both trees have same structure
        f = f2 = Frame("test", Transformation.neutral(), WORLD_FRAME)
        f._parent = f2
        f2._parent = f
        with self.assertRaises(ValueError):
            _ = f == f2

    def test_eq_ft(self):
        ft1 = FrameTree(self.data)
        ft2 = FrameTree(self.data)
        self.assertEqual(ft1, ft2)

    def test_frame_circle(self):
        data = {
            "1": {"transformation": Transformation.random(), "parent": "2"},
            "2": {"transformation": Transformation.random(), "parent": "3"},
            "3": {"transformation": Transformation.random(), "parent": "1"}
        }
        with self.assertRaises(ValueError):
            _ = FrameTree(data)

    def test_special_frames(self):
        """Make sure nominal work as expected."""
        ft = FrameTree(self.data)
        with self.assertRaises(AttributeError):
            _ = ft["nominal"].in_world_coordinates()

        # Ensure that special frames are singletons
        ft2 = FrameTree(self.data)
        self.assertTrue(ft["world"] is ft2["world"])
        self.assertTrue(ft["nominal"] is ft2["nominal"])

    def test_frame_tree_reserved_keywords(self):
        with self.assertRaises(ValueError):
            FrameTree({
                "world": {  # Reserved keyword for the world frame; should have None parent
                    "transformation": Transformation.neutral(), "parent": "some_parent"
                },
                "another": {
                    "transformation": Transformation.from_translation((0, 0, 1)), "parent": "world"
                }
            })

        with self.assertRaises(ValueError):
            FrameTree({
                "world": {  # Reserved keyword for the world frame; should have neutral transformation
                    "transformation": Transformation.random(), "parent": None
                },
                "another": {
                    "transformation": Transformation.from_translation((0, 0, 1)), "parent": "world"
                }
            })

        with self.assertRaises(ValueError):
            FrameTree({
                "nominal": {},  # Reserved keyword for the nominal frame; cannot be a parent
                "another": {
                    "transformation": Transformation.from_translation((0, 0, 1)), "parent": "world"
                }
            })

        with self.assertRaises(ValueError):
            FrameTree({
                "special.dot": {},  # Reserved keyword for the nominal frame; cannot be a parent
                "another": {
                    "transformation": Transformation.from_translation((0, 0, 1)), "parent": "world"
                }
            })

    def test_serialization(self):
        ft = FrameTree(self.data)
        ft2 = FrameTree.from_json_data(ft.to_json_data())
        self.assertEqual(ft, ft2)

    def _get_test_task_data(self):
        task_files = get_test_tasks()["task_files"]

        # Check PTP_1 for handling of constraint and goal with frames
        PTP_1_task_file = [file for file in task_files if "PTP_1" in file.name][0]
        description = json.load(open(PTP_1_task_file))
        description["frames"] = self.data
        return description

    def test_frames_in_task(self):
        task = Task.from_json_data(self._get_test_task_data())
        for f in self.data.keys():
            self.assertIn(f, task.frames)
        self.assertEqual(task.base_constraint.base_pose.nominal.ID, 'constraint.base')
        for frame_id in ('constraint.base', *(f'goal.{id}' for id in task.goals_by_id.keys())):
            self.assertIn(frame_id, task.frames)
        self.assertEqual(task.frames.to_json_data(), self.data)
        self.assertIsInstance(task.base_constraint.base_pose.nominal, Frame)
        for g in task.goals:
            self.assertIsInstance(g.goal_pose.nominal, Frame)
        task_data = task.to_json_data()
        for f in task_data["frames"].values():
            self.assertIsInstance(f["transformation"], list)
            self.assertIsInstance(f["parent"], str)
        task2 = Task.from_json_data(task_data)
        self.assertEqual(task, task2)

    def test_frames_in_task_with_obstacles(self):
        # Check PTP_2 for handling of geometry with frames
        task_files = get_test_tasks()["task_files"]
        PTP_2_task_file = [file for file in task_files if "PTP_2.json" in file.name][0]
        description = Task.from_json_file(PTP_2_task_file).to_json_data()  # Generate valid local asset paths
        description["frames"] = self.data
        task = Task.from_json_data(description)
        task2 = Task.from_json_data(task.to_json_data())
        self.assertEqual(task, task2)

    def test_tolerance_frames_in_task(self):
        task_data = self._get_test_task_data()
        task_data["goals"][0]["goalPose"]["toleranceFrame"] = ("nominal", "world")
        task_data["goals"][1]["goalPose"]["toleranceFrame"] = ("nominal", "1")
        task = Task.from_json_data(task_data)
        self.assertIsInstance(
            tuple(task.goals_by_id["1"].goal_pose.tolerance.tolerances)[0], Tolerance.RotationAbsolute)
        self.assertIsInstance(
            tuple(task.goals_by_id["1"].goal_pose.tolerance.tolerances)[1], Tolerance.CartesianSpheric)
        self.assertEqual(tuple(task.goals_by_id["1"].goal_pose.tolerance.tolerances)[0]._frame, WORLD_FRAME)
        self.assertEqual(tuple(task.goals_by_id["1"].goal_pose.tolerance.tolerances)[1]._frame, NOMINAL_FRAME)
        self.assertIsInstance(
            tuple(task.goals_by_id["2"].goal_pose.tolerance.tolerances)[0], Tolerance.RotationAbsolute)
        self.assertIsInstance(
            tuple(task.goals_by_id["2"].goal_pose.tolerance.tolerances)[1], Tolerance.CartesianSpheric)
        self.assertEqual(tuple(task.goals_by_id["2"].goal_pose.tolerance.tolerances)[0]._frame, task.frames["1"])
        self.assertEqual(tuple(task.goals_by_id["2"].goal_pose.tolerance.tolerances)[1]._frame, NOMINAL_FRAME)
        task_data_new = task.to_json_data()
        for goal_idx in range(len(task.goals)):  # Make sure tolerance frames in same order as projections
            self.assertEqual(task_data_new["goals"][goal_idx]["ID"], task_data["goals"][goal_idx]["ID"])
            for projection, frame in zip(
                    task_data_new["goals"][goal_idx]["goalPose"]["toleranceProjection"],
                    task_data_new["goals"][goal_idx]["goalPose"]["toleranceFrame"]):
                self.assertTrue(np.allclose(
                    projection == np.asarray(task_data["goals"][goal_idx]["goalPose"]["toleranceProjection"]),
                    frame == np.asarray(task_data["goals"][goal_idx]["goalPose"]["toleranceFrame"])))

    def test_frames_with_tolerated_poses(self):
        ft = FrameTree(self.data)
        T1 = ToleratedPose(Frame("", Transformation.random(), WORLD_FRAME))
        self.assertTrue(T1.valid(T1.nominal.in_world_coordinates()))
        T2 = ToleratedPose(Frame("", Transformation.random(), ft["1"]))
        self.assertTrue(T2.valid(T2.nominal.in_world_coordinates()))
        self.assertFalse(T2.valid(T1.nominal.in_world_coordinates()))
        T3 = ToleratedPose.from_json_data({
            "nominal": [
                [1, 0, 0, -0.5],
                [0, 1, 0, -1.1],
                [0, 0, 1, 0.5],
                [0, 0, 0, 1]
            ],
            "nominalParent": "world"
        }, frames=ft)
        self.assertTrue(T3.valid(T3.nominal.in_world_coordinates()))
        T4 = ToleratedPose.from_json_data(T3.to_json_data(), frames=ft, frameName="some_name.2")
        self.assertTrue(T4.valid(T3.nominal.in_world_coordinates()))
        self.assertEqual(T4.nominal.ID, "some_name.2")
        self.assertIn("some_name.2", ft)
        self.assertNotIn("some_name.2", ft.to_json_data())
        with self.assertRaises(ValueError):
            _ = ToleratedPose.from_json_data({
                "nominal": [
                    [1, 0, 0, -0.5],
                    [0, 1, 0, -1.1],
                    [0, 0, 1, 0.5],
                    [0, 0, 0, 1]
                ],
                "nominalParent": "a-non-registered-frame"
            }, frames=ft)

    def test_frames_with_tolerances(self):
        frames = {
            "1": {
                "transformation": Transformation.from_translation((1, 2, 3)), "parent": "world"
            },
            "2": {
                "transformation": spatial.rotX(np.pi / 2), "parent": "1"
            },
        }
        ft = FrameTree(frames)

        # Check nominal frame (simplest and previous standard) and parallel ones
        for tolerance in (Tolerance.CartesianXYZ((-.1, .2), (-.3, .4), (-.5, .6), frame=ft["nominal"]),):
            # RotX means y = z_world, z = -y_world
            self.assertTrue(tolerance.valid(spatial.rotX(np.pi / 2), Transformation.from_translation((0, 0, 0))))
            self.assertTrue(tolerance.valid(spatial.rotX(np.pi / 2), Transformation.from_translation((0, -.6, 0.))))
            self.assertFalse(tolerance.valid(spatial.rotX(np.pi / 2), Transformation.from_translation((0, .51, 0.))))
            self.assertTrue(tolerance.valid(spatial.rotX(np.pi / 2), Transformation.from_translation((0, 0, .4))))
            self.assertFalse(tolerance.valid(spatial.rotX(np.pi / 2), Transformation.from_translation((0, 0, -.31))))
            # Different desired frame
            self.assertFalse(tolerance.valid(Transformation.neutral(), Transformation.from_translation((0, -.6, 0.))))

        # Check frames parallel to world
        for tolerance in (
                Tolerance.CartesianXYZ((-.1, .2), (-.3, .4), (-.5, .6), frame=ft["world"]),
                Tolerance.CartesianXYZ((-.1, .2), (-.3, .4), (-.5, .6), frame=ft["1"])):
            # This should ignore any rotation
            for _ in range(100):
                R = Transformation.from_rotation(spatial.random_rotation())
                self.assertTrue(tolerance.valid(R, Transformation.from_translation((0, 0, 0))))
                self.assertTrue(tolerance.valid(R, Transformation.from_translation((0, 0, .6))))
                self.assertFalse(tolerance.valid(R, Transformation.from_translation((0, 0, -.51))))
                self.assertTrue(tolerance.valid(R, Transformation.from_translation((0, .4, 0))))
                self.assertFalse(tolerance.valid(R, Transformation.from_translation((0, -.31, 0))))

                # But respect translation (even with some rotation component)
                self.assertTrue(tolerance.valid(Transformation.from_translation((1, 2, 3)) @ R,
                                                Transformation.from_translation((1, 2, 3))))
                self.assertTrue(tolerance.valid(Transformation.from_translation((1, 2, 3)) @ R,
                                                Transformation.from_translation((1, 2, 3.6))))
                self.assertFalse(tolerance.valid(Transformation.from_translation((1, 2, 3)) @ R,
                                                 Transformation.from_translation((1, 2, 3.61))))

        # Check translated + rotated frame "2"
        tolerance = Tolerance.CartesianXYZ((-.1, .2), (-.3, .4), (-.5, .6), frame=ft["2"])
        # Everything from world turned by 90° around x means y = z_world, z = -y_world
        self.assertTrue(tolerance.valid(Transformation.neutral(), Transformation.from_translation((0, 0, 0))))
        self.assertTrue(tolerance.valid(Transformation.neutral(), Transformation.from_translation((0, .5, 0))))
        self.assertFalse(tolerance.valid(Transformation.neutral(), Transformation.from_translation((0, .51, 0))))
        self.assertTrue(tolerance.valid(Transformation.neutral(), Transformation.from_translation((0, 0, -.3))))
        self.assertFalse(tolerance.valid(Transformation.neutral(), Transformation.from_translation((1, 2, -.31))))

    def test_mixed_tolerance(self):
        def tolerated_pose_rotated_with_frame(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max) -> ToleratedPose:
            return ToleratedPose.from_json_data(
                {
                    "nominal": Transformation.from_translation((x, y, z)) @ Transformation(rotX(np.pi / 2)),
                    "toleranceProjection": [
                        "x",
                        "y",
                        "z"
                    ],
                    "tolerance": [
                        [x_min, x_max],
                        [y_min, y_max],
                        [z_min, z_max]
                    ],
                    "toleranceFrame": [WORLD_FRAME, WORLD_FRAME, NOMINAL_FRAME]
                },
                frames=FrameTree.empty())

        pose = tolerated_pose_rotated_with_frame(-1, 2, 1, -.1, .2, -.3, .4, -.5, .6)
        self.assertTrue(pose.valid(Transformation.from_translation((-1, 2, 1))))
        self.assertTrue(pose.valid(Transformation.from_translation((-1.1, 2, 1))))
        self.assertFalse(pose.valid(Transformation.from_translation((-1.11, 2, 1))))
        self.assertTrue(pose.valid(Transformation.from_translation((-1, 2.4, 1))))
        self.assertFalse(pose.valid(Transformation.from_translation((-1, 2.41, 1))))
        for i in range(10):  # z world is not constrained as the z constraint is in nominal frame
            self.assertTrue(pose.valid(Transformation.from_translation((-1, 2.4, 1 + .1 * i))))

        # With same tolerance frame
        pose = ToleratedPose.from_json_data(
            {
                "nominal": Transformation.from_translation((1, 2, 3)) @ Transformation(rotX(np.pi / 2)),
                "toleranceProjection": [
                    "x",
                    "y"
                ],
                "tolerance": [
                    [-.1, .2],
                    [-.3, .4]
                ],
                "toleranceFrame": [WORLD_FRAME, WORLD_FRAME]
            },
            frames=FrameTree.empty())
        self.assertEqual(len(pose.tolerance.tolerances), 1)
        self.assertTrue(tuple(pose.tolerance.tolerances)[0]._frame is WORLD_FRAME)

    def test_axis_angle(self):
        tolerance = Tolerance.RotationAxis([0, 0], [0, 0], [-np.pi, np.pi])  # Any rotation around z
        for _ in range(10):
            self.assertTrue(tolerance.valid(Transformation.neutral(),
                                            Transformation(spatial.rotZ(random.random() * 10))))
            self.assertFalse(tolerance.valid(Transformation.neutral(),
                                             Transformation(spatial.rotX(random.random() * 10))))
            T = Transformation.random()
            self.assertTrue(tolerance.valid(T, T))
            self.assertTrue(tolerance.valid(T, T @ spatial.rotZ(random.random() * 10)))
            self.assertFalse(tolerance.valid(T, T @ spatial.rotX(random.random() * 10)))

        # Any rotation around _world_ z
        tolerance = Tolerance.RotationAxis([0, 0], [0, 0], [-np.pi, np.pi], frame=WORLD_FRAME)
        for _ in range(10):
            self.assertTrue(tolerance.valid(Transformation(spatial.rotZ(random.random() * 10)),
                                            Transformation.neutral()))
            self.assertFalse(tolerance.valid(Transformation.neutral(),
                                             Transformation(spatial.rotX(random.random() * 10))))
            T = Transformation.random()
            self.assertTrue(tolerance.valid(T, T))
            self.assertTrue(tolerance.valid(T, Transformation(spatial.rotZ(random.random() * 10)) @ T))
            self.assertFalse(tolerance.valid(T, Transformation(spatial.rotX(random.random() * 10)) @ T))

    def test_visualization(self):
        task_files = get_test_tasks()["task_files"]
        PTP_1_task_file = [file for file in task_files if "PTP_1" in file.name][0]
        description = json.load(open(PTP_1_task_file))
        description["frames"] = self.data
        task = Task.from_json_data(description)
        task.visualize()
        print("Visualize task")
