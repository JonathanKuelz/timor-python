from copy import copy, deepcopy
import importlib
import itertools
import logging
import pickle
import unittest

from hppfcl import CollisionRequest, CollisionResult, collide
import meshcat
import numpy as np
import pinocchio as pin
import pytest

from timor import Bodies, Geometry, Obstacle, Transformation
from timor.Geometry import Mesh
from timor.parameterized import ParameterizableBody, ParameterizableBoxBody, ParameterizableCylinderBody, \
    ParameterizableMultiBody, \
    ParameterizableSphereBody
import timor.utilities.errors as err
from timor.utilities.file_locations import robots, test_data


class TestBodiesAndGeometries(unittest.TestCase):

    def test_body(self):
        """Tests body instantiation and basic functionality"""

        female = Bodies.Gender.female
        male = Bodies.Gender.male
        c1 = Bodies.Connector('1', Transformation.from_translation([.4, 0, 0]), None, female, 'clamp', size=1)
        c2 = Bodies.Connector('2', Transformation.from_translation([-.4, 0, 0]), None, male, 'clamp', size=1)
        c3 = Bodies.Connector('3', Transformation.from_translation([-.1, .2, 0]), None, male, 'clamp', size=1)

        box = Geometry.Box({'x': 1, 'y': 1, 'z': 1})
        gen_c1 = copy(c1)
        gen_c2 = copy(c2)
        generic_body = Bodies.Body('42', collision=box, connectors=[gen_c1, gen_c2])
        another_body = Bodies.Body('123', collision=box, connectors=[c1, c2, c3])

        possible_connections = {(c1, gen_c2), (c2, gen_c1), (c3, gen_c1)}  # Between box and generic_body
        self.assertEqual(possible_connections, another_body.possible_connections_with(generic_body))
        self.assertEqual(len(another_body.possible_connections_with(
            Bodies.Body('1', collision=box))), 0)  # No-connector body

        same_id_body = Bodies.Body('123', collision=box)
        with self.assertRaises(err.UniqueValueError):
            # Cannot connect bodies with same ID
            another_body.possible_connections_with(same_id_body)

    def test_geometry(self):
        box = Geometry.Box({'x': 1, 'y': 1, 'z': 1})
        cylinder = Geometry.Cylinder({'r': 1, 'z': 5})
        sphere = Geometry.Sphere({'r': 1})
        mesh = Geometry.Mesh({'file': 'sample_tasks/assets/DMU_125P.stl', 'package_dir': test_data,
                              'scale': 0.001})

        # Volumes
        self.assertEqual(box.volume, 1)
        self.assertAlmostEqual(cylinder.volume, 5 * np.pi * 1 ** 2)
        self.assertAlmostEqual(sphere.volume, 4 / 3 * np.pi * 1 ** 3)
        self.assertGreater(mesh.volume, 0)  # Just check that it makes sense

        composed = Geometry.ComposedGeometry([box, cylinder, sphere, mesh])
        empty = Geometry.EmptyGeometry()
        self.assertEqual(composed.volume, box.volume + cylinder.volume + sphere.volume + mesh.volume)
        composed._composing_geometries.append(empty)
        self.assertEqual(composed.volume, box.volume + cylinder.volume + sphere.volume + mesh.volume)

        another_composed = Geometry.ComposedGeometry([mesh, sphere, box, cylinder])
        self.assertEqual(composed, another_composed)

        for g1, g2 in itertools.combinations([box, cylinder, sphere, mesh], 2):
            self.assertNotEqual(g1, g2)

        # Deepcopy
        for g in (box, cylinder, sphere, mesh):
            g_new = deepcopy(g)
            self.assertEqual(g.volume, g_new.volume)
            self.assertDictEqual(g.parameters, g_new.parameters)

    def test_parameterizable_body(self):
        num_params = {Geometry.GeometryType.BOX: 3, Geometry.GeometryType.CYLINDER: 2, Geometry.GeometryType.SPHERE: 1}
        bodies = list()
        parameters = list()
        offset_bodies = list()
        for i, (t, num) in enumerate(num_params.items()):
            parameters.append(np.random.rand(num))
            cls = ParameterizableMultiBody.geometry_type_to_class[t]
            bodies.append(cls(str(i), parameters[-1]))
            offset_bodies.append(cls(str(i), parameters[-1],
                                     geometry_placement=Transformation.from_translation([0, 0, 15])))
            with self.assertRaises(ValueError):
                cls(str(i), ())
            with self.assertRaises(ValueError):
                cls(str(i), np.random.rand(num + 1))

        for body in bodies:
            self.assertTrue(all(body.inertia.lever == np.zeros(3)))

        for b1, b2 in zip(bodies, offset_bodies):
            self.assertEqual(b1.mass, b2.mass)
            o1 = Obstacle.Obstacle('1', b1.collision)
            o2 = Obstacle.Obstacle('2', b2.collision)
            self.assertFalse(o1.collides(o2))
            self.assertTrue(o1.collides(o1))
            self.assertTrue(o2.collides(o2))

        heavy_bodies = [copy(b) for b in bodies]
        for b in heavy_bodies:
            b.mass_density = 10
        for b1, b2 in zip(bodies, heavy_bodies):
            self.assertGreater(b2.mass, b1.mass)
            self.assertEqual(b2.mass, b1.mass * 10)
            self.assertTrue(np.all(b2.inertia.inertia >= b1.inertia.inertia))
            self.assertTrue(all(b2.inertia.lever == b1.inertia.lever))

        for body, parameters in zip(bodies, parameters):
            body.connector_placements_valid = True  # Has to be done for freezing
            self.assertIsInstance(Bodies.Body.from_json_data(body.to_json_data(), None), Bodies.Body)
            self.assertIsInstance(body.freeze(), Bodies.Body)

            pickle_body = pickle.loads(pickle.dumps(body))
            self.assertIsInstance(pickle_body, ParameterizableBody)
            self.assertEqual(pickle_body.mass, body.mass)
            self.assertEqual(pickle_body.inertia, body.inertia)

            okay_limits = np.vstack((np.maximum(parameters - 0.1, np.zeros_like(parameters)), parameters + 0.1)).T
            not_okay_limits = np.vstack((np.maximum(parameters - 0.1, np.zeros_like(parameters)), parameters - 0.1)).T

            body.parameter_limits = okay_limits
            with self.assertRaises(ValueError):
                body.parameter_limits = not_okay_limits

            body._parameter_limits = not_okay_limits
            with self.assertRaises(ValueError):
                body.parameters = parameters

    def test_parameterizable_multi_body(self):
        cylinder = ParameterizableCylinderBody('cylinder', [.2, 1])
        sphere = ParameterizableSphereBody('sphere', [.4])

        p1 = Transformation.from_translation([0, 0, .5])
        p2 = Transformation.from_translation([0, 0, 1])
        dumbbell = ParameterizableMultiBody(
            'dumbbell', [Geometry.GeometryType.SPHERE, Geometry.GeometryType.SPHERE, Geometry.GeometryType.CYLINDER],
            parameters=[.4, .4, .2, 1],
            geometry_placements=[Transformation.neutral(), p2, p1])

        self.assertAlmostEqual(dumbbell.mass, cylinder.mass + 2 * sphere.mass)

        inertia = dumbbell.inertia
        self.assertAlmostEqual(inertia.lever[2], .5)
        for i in (4, 6, 9):
            self.assertLess(cylinder.inertia.toDynamicParameters()[i], inertia.toDynamicParameters()[i])
            self.assertLess(sphere.inertia.toDynamicParameters()[i], inertia.toDynamicParameters()[i])

        sphere._geometry_transformation = Transformation.from_translation([0, 0, 2])
        o_cyl = Obstacle.Obstacle('cylinder', cylinder.collision)
        o_sphere = Obstacle.Obstacle('sphere', sphere.collision)
        o_composed = Obstacle.Obstacle('composed', dumbbell.collision)
        self.assertTrue(o_composed.collides(o_cyl))
        self.assertFalse(o_composed.collides(o_sphere))
        o_sphere.collision.placement = Transformation.from_translation([0, 0, 1.2])
        self.assertTrue(o_composed.collides(o_sphere))

        dumbbell.connector_placements_valid = True  # Validate connector placements for freezing
        self.assertIsInstance(Bodies.Body.from_json_data(dumbbell.to_json_data(), None), Bodies.Body)
        self.assertIsInstance(dumbbell.freeze(), Bodies.Body)
        pickle_body = pickle.loads(pickle.dumps(dumbbell))
        self.assertIsInstance(pickle_body, ParameterizableMultiBody)
        self.assertEqual(pickle_body.mass, dumbbell.mass)
        self.assertEqual(pickle_body.inertia, dumbbell.inertia)

    def test_wrl(self):
        mesh_wrl = Mesh({'file': str(robots['IMPROV'].joinpath('STLfiles').joinpath('convexDecompose/L1.stl.wrl')),
                         'package_dir': str(robots['IMPROV'].joinpath('STLfiles/convexDecompose'))})
        mesh_stl = Mesh({'file': robots['IMPROV'].joinpath('STLfiles').joinpath('L1.stl'),
                         'package_dir': robots['IMPROV'].joinpath('STLfiles')})
        test_cube = ParameterizableBoxBody("tmp_test_box", (1., 1., 1.),
                                           geometry_placement=Transformation.from_translation((1., 0., 0.)))
        request = CollisionRequest()
        result = CollisionResult()

        v = meshcat.visualizer.Visualizer()
        v["test_wrl"].set_object(mesh_wrl.viz_object[0])
        v["test_stl"].set_object(mesh_stl.viz_object[0])
        v["test_cube"].set_object(test_cube.visual.viz_object[0])
        v["test_cube"].set_transform(test_cube.visual.viz_object[1].homogeneous)

        # mesh_wrl should be (slight) over-approximation -> similar collision properties
        for x in np.arange(0., 1., 0.1):
            mesh_wrl.placement = Transformation.from_translation((x, 0, 0))
            mesh_stl.placement = Transformation.from_translation((x, 0, 0))

            v["test_wrl"].set_transform(mesh_wrl.viz_object[1].homogeneous)
            v["test_stl"].set_transform(mesh_stl.viz_object[1].homogeneous)

            if collide(test_cube.collision.as_hppfcl_collision_object[0], mesh_stl.as_hppfcl_collision_object[0],
                       request, result):
                self.assertTrue(collide(test_cube.collision.as_hppfcl_collision_object[0],
                                        mesh_wrl.as_hppfcl_collision_object[0], request, result),
                                f"Overapprox. wrl did not collide @ x = {x}")

    @pytest.mark.full
    def test_mesh_color(self):
        """
        Cannot check that actual intended colors are used as we cannot get data back from the meshcat viewer.

        Just tests API; correct render must be checked manually.
        """
        if not importlib.util.find_spec("trimesh"):
            raise unittest.SkipTest("trimesh not installed")
        mesh_ply = Mesh({'file': test_data.joinpath('sample_tasks').joinpath('assets').
                        joinpath('IWB Hermle UWF 900_only.ply'),
                         'package_dir': test_data.joinpath('sample_tasks').joinpath('assets')})

        self.assertTupleEqual(mesh_ply.viz_object[0].vertices.shape, mesh_ply._colors.shape)
        self.assertTupleEqual(mesh_ply.viz_object[0].vertices.shape, mesh_ply.viz_object[0].color.shape)

        v = pin.visualize.MeshcatVisualizer()
        v.initViewer()
        v.viewer["test_ply"].set_object(mesh_ply.viz_object[0], meshcat.geometry.MeshPhongMaterial(vertexColors=True))

        obstacle = Obstacle.Obstacle("test", mesh_ply)
        obstacle.visualize(v)

        obstacle_red = Obstacle.Obstacle("test_red", mesh_ply)
        obstacle_red.visualize(v, meshcat.geometry.MeshPhongMaterial(color=0xFF0000))

        logging.info("Colors need to be checked manually.")


if __name__ == '__main__':
    unittest.main()
