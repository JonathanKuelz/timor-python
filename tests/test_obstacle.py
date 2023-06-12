import unittest

import hppfcl
import numpy as np
import pinocchio as pin

from timor.task.Obstacle import Obstacle
from timor.utilities.file_locations import robots


class TestObstacles(unittest.TestCase):
    """Test reading / writing obstacle data"""
    def setUp(self) -> None:
        self.demo_stl = robots['panda'].joinpath('meshes/collision/hand.stl')

    def test_load_stl(self):
        """
        Check that vertices loaded from stl are same as those in the file

        :return:
        """
        mesh_loader = hppfcl.MeshLoader()
        m = mesh_loader.load(str(self.demo_stl))

        geometry_desc = {'type': 'mesh', 'parameters': {"file": self.demo_stl}}
        obs = Obstacle.from_json_data(dict(ID='123', collision=geometry_desc))

        with open(self.demo_stl, "rb") as f:
            header = f.read(80)
            tri_count = np.frombuffer(f.read(4), dtype=np.int32)
            _ = f.read(12)  # firstNormal
            firstVertex = np.frombuffer(f.read(12), dtype=np.float32)
            secondVertex = np.frombuffer(f.read(12), dtype=np.float32)
            thirdVertex = np.frombuffer(f.read(12), dtype=np.float32)

            self.assertEqual(tri_count, m.num_tris)
            np.testing.assert_array_equal(m.vertices()[m.tri_indices(0)[0], :], firstVertex)
            np.testing.assert_array_equal(m.vertices()[m.tri_indices(0)[1], :], secondVertex)
            np.testing.assert_array_equal(m.vertices()[m.tri_indices(0)[2], :], thirdVertex)

            self.assertEqual(tri_count, obs.collision.collision_geometry.num_tris)
            coll_geom = obs.collision.collision_geometry
            np.testing.assert_array_equal(coll_geom.vertices()[coll_geom.tri_indices(0)[0], :], firstVertex)
            np.testing.assert_array_equal(coll_geom.vertices()[coll_geom.tri_indices(0)[1], :], secondVertex)
            np.testing.assert_array_equal(coll_geom.vertices()[coll_geom.tri_indices(0)[2], :], thirdVertex)

    def test_visualize(self):
        geometry_desc = [{'type': 'mesh', 'parameters': {"file": self.demo_stl}},
                         {'type': 'box', 'parameters': {'x': 1, 'y': 2, 'z': 3}},
                         {'type': 'cylinder', 'parameters': {'r': 2, 'z': 3}},
                         {'type': 'sphere', 'parameters': {'r': 3}}]
        for desc in (geometry_desc, *geometry_desc):  # Check composed and each individually
            obs = Obstacle.from_json_data(dict(ID='123', collision=desc))
            vis = pin.visualize.MeshcatVisualizer()
            vis.initViewer()
            obs.visualize(vis)
