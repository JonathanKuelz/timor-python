import logging
import time
import unittest

import networkx as nx
import numpy as np
import numpy.testing as np_test

from timor import AtomicModule, Body, Connector, Geometry, ModuleAssembly, ModuleHeader, ModulesDB, RobotBase, \
    Transformation
from timor.parameterized import ParameterizedCylinderLink, ParameterizedOrthogonalJoint, ParameterizedOrthogonalLink, \
    ParameterizedStraightJoint
from timor.utilities import spatial
import timor.utilities.errors as err
from timor.utilities.file_locations import get_module_db_files
from timor.Joints import TimorJointType


class TestModuleAssembly(unittest.TestCase):

    def setUp(self) -> None:
        """Initialize some modules"""
        self.json_file = get_module_db_files('IMPROV')
        self.db = ModulesDB.from_json_file(self.json_file)

    def test_graph_features(self):
        """An assembly of modules is (given some abstraction) just a graph. Check some basic graph attributes:"""
        assembly = ModuleAssembly(self.db)
        for _ in range(12):
            assembly.add_random_from_db(lambda x: x not in self.db.end_effectors)

        adjacency = assembly.adjacency_matrix
        self.assertEqual(2 * len(assembly.connections), adjacency.sum())

        num_nodes = len(assembly.module_instances)
        # Check the networkx graph functionality and use it to evaluate the assembly structure
        nxGraph = assembly.graph_of_modules
        self.assertEqual(nxGraph.number_of_nodes(), num_nodes)
        self.assertTrue(nx.is_connected(nxGraph), "There seem to be unconnected modules in the assembly")
        np_test.assert_array_equal(adjacency, nx.convert_matrix.to_numpy_array(nxGraph, dtype=int))
        np_test.assert_array_equal(adjacency[np.diag_indices(num_nodes)], np.zeros((num_nodes,), dtype=int),
                                   "There are self-loops in the assembly")

        G_a = assembly.assembly_graph

    def test_modules_assembly(self):
        module_ids = ['1', '21', '4', '21', '5', '23', '7', '12']
        connections = [
            (1, '21_proximal_connector', 0, 'base_out'),
            (2, '4_proximal_connector', 1, '21_distal_connector'),
            (3, '21_proximal_connector', 2, '4_distal_connector'),
            (4, '5_proximal_connector', 3, '21_distal_connector'),
            (5, '23_proximal_connector', 4, '5_distal_connector'),
            (6, '7_proximal_connector', 5, '23_distal_connector'),
            (7, '12_proximal_connector', 6, '7_distal_connector'),
        ]
        assembly = ModuleAssembly(self.db, module_ids, connections)
        assembly = ModuleAssembly(self.db, module_ids, set(connections))  # Order invariance

        # Now, add a "loose" module and see if it fails
        with self.assertRaises(ValueError):
            assembly = ModuleAssembly(self.db, module_ids + ['1'], connections)

        # Add an invalid connection
        module_ids = module_ids + ['1']  # base
        connections.append((7, '12_distal_connector', 8, 'base'))
        with self.assertRaises(err.InvalidAssemblyError):
            assembly = ModuleAssembly(self.db, module_ids, connections)

        # Check that an empty assembly does not need to stay empty
        assembly = ModuleAssembly(self.db)
        assembly.add_random_from_db()
        assembly.add_random_from_db()
        self.assertIsNotNone(assembly.base_connector)
        self.assertEqual(len(assembly.module_instances), 2)

        # Check that the assembly can be (de)serialized and we have a functional (in)equality
        new_assembly = assembly.from_json_string(assembly.to_json_string())
        self.assertEqual(assembly, new_assembly)
        try:
            new_assembly.add_random_from_db()
        except LookupError:
            logging.debug("No modules left to add. Removing one module for inequality test")
            new_assembly.connections.pop()
        self.assertNotEqual(assembly, new_assembly)

    def test_parameterizable_assembly(self):
        """Tests assemblies containing at least on parameterizable module."""
        base_header = ModuleHeader('B', 'Base')
        cyl_header = ModuleHeader('cylinder', 'cylinder')
        l_header = ModuleHeader('L', 'orthogonal link')
        straight_header = ModuleHeader('J1', 'Straight Prismatic Joint')
        orthogonal_header = ModuleHeader('J2', 'Orthogonal Revolute Joint')
        box = Geometry.Box({'x': .5, 'y': .5, 'z': .1})
        base_cons = (
            Connector('world', spatial.rotX(np.pi), gender='male', connector_type='base'),
            Connector('out', Transformation.from_translation([0, 0, .1 / 2]), gender='male')
        )
        base_body = Body('base_body', box, connectors=base_cons)
        base = AtomicModule(base_header, bodies=(base_body,))
        cylinder = ParameterizedCylinderLink(cyl_header)
        l_shaped = ParameterizedOrthogonalLink(l_header, lengths=(1, 3), radius=.2)
        straight = ParameterizedStraightJoint(straight_header, radius=.1, joint_type=TimorJointType.prismatic)
        orthogonal = ParameterizedOrthogonalJoint(orthogonal_header, radius=.1, joint_type=TimorJointType.revolute)

        db = ModulesDB((base, cylinder, l_shaped, straight, orthogonal))

        assembly = ModuleAssembly.from_serial_modules(db, ('B', 'cylinder', 'J1', 'L', 'J2'))
        Gmod = assembly.graph_of_modules
        self.assertTrue(nx.is_connected(Gmod))
        Ga = assembly.assembly_graph
        self.assertTrue(nx.is_weakly_connected(Ga))

        rob = assembly.to_pin_robot()
        self.assertAlmostEqual(rob.mass, assembly.mass)

        fk = rob.fk()
        move_up = .3
        rob.update_configuration(np.array((move_up, 0)))
        self.assertAlmostEqual(fk[2, 3], rob.fk()[2, 3] - move_up)
        self.assertTrue(np.allclose(fk[:2, 3], rob.fk()[:2, 3]))
        self.assertTrue(np.allclose(fk[:3, :3], rob.fk()[:3, :3]))

        rob.update_configuration(np.array((0, np.pi)))
        self.assertAlmostEqual(fk[0, 3], rob.fk()[0, 3])
        self.assertAlmostEqual(fk[1, 3], rob.fk()[1, 3])
        self.assertAlmostEqual(fk[2, 3] + orthogonal.l2 * 2, rob.fk()[2, 3])

    def test_robot_property(self):
        assembly = ModuleAssembly(self.db)
        empty_robot = assembly.robot
        for _ in range(3):
            assembly.add_random_from_db(lambda module: module not in self.db.end_effectors)
        t0 = time.time()
        robot = assembly.robot
        self.assertIsNotNone(assembly._robot.value)
        t1 = time.time()
        robot = assembly.robot
        t2 = time.time()
        self.assertIsInstance(robot, RobotBase)
        self.assertLess(t2 - t1, t1 - t0)
        self.assertLess(t2 - t1, 1 / 1000)  # Retrieving an existing value should take way less than 1ms

        assembly.add_random_from_db()
        self.assertIsNone(assembly._robot.value)  # Should be reset
        robot = assembly.robot
        self.assertIsInstance(assembly._robot.value, RobotBase)
        self.assertIs(assembly.robot, robot)
        self.assertIs(assembly.robot, assembly._robot.value)

        # Perform a couple of "random" operations that retrieve information but do not change the assembly
        _ = (assembly.mass, assembly.nModules, assembly.nJoints)
        _ = (assembly.assembly_graph, assembly.graph_of_modules)
        # The assembly should still be cached
        self.assertIs(assembly.robot, robot)


if __name__ == '__main__':
    unittest.main()
