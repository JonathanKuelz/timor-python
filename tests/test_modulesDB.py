import json
from pathlib import Path
import random
import unittest

import networkx as nx
import numpy as np
import numpy.testing as np_test

from timor import Joint, ToleratedPose, Transformation
from timor.Module import AtomicModule, ModuleAssembly, ModuleHeader, ModulesDB
from timor.utilities import logging
import timor.utilities.errors as err
from timor.utilities.file_locations import get_module_db_files, robots


class TestModulesDB(unittest.TestCase):
    """Test if all known good module DBs are loadable, class is working, I/O from/to json, assemblies, ..."""

    def setUp(self) -> None:
        """Defines the filepaths before testing"""
        self.json_file, self.package_dir = get_module_db_files('IMPROV')
        self.db = ModulesDB.from_file(self.json_file, package_dir=self.package_dir)
        random.seed(123)
        np.random.seed(123)

    def test_dump_json(self):
        """Tests loading and dumping from and to json."""
        db = ModulesDB.from_file(self.json_file, package_dir=self.package_dir)
        json_string = db.to_json()
        parsed = json.loads(json_string)
        db_two = ModulesDB.from_json(parsed, self.package_dir)
        json_again = db_two.to_json()

        self.assertEqual(json_string, json_again)
        self.assertEqual(db.all_module_ids, db_two.all_module_ids)
        self.assertTrue(nx.is_isomorphic(db.connectivity_graph, db_two.connectivity_graph))

    def test_empty(self):
        """Test that classes can be instantiated empty"""
        minimum_header = ModuleHeader('id', 'name')
        mod = AtomicModule(minimum_header)  # Header required
        mod = AtomicModule.from_json_data({'header': minimum_header._asdict()}, Path())
        db = ModulesDB()
        assembly = ModuleAssembly(db)
        with self.assertRaises(ValueError):
            robot = assembly.to_pin_robot()  # An empty robot is not possible
        with self.assertRaises(LookupError):
            assembly.add_random_from_db()  # The db is empty

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

    def test_load_all_from_file(self):
        for db_name, db_dir in robots.items():
            if len(tuple(db_dir.rglob('*.json'))) == 0:
                logging.debug('Not trying to load db {} - probably not a database'.format(db_name))
                continue  # Not a json DB file in there
            db = ModulesDB.from_file(*get_module_db_files(db_name))
            self.assertGreater(len(db), 0, msg=f"{db_name} is empty after loading")

    def test_jointless_robot(self):
        """Test the Assembly to Robot procedure for a robot without a joint"""
        db = ModulesDB.from_file(*get_module_db_files('geometric_primitive_modules'))
        assembly = ModuleAssembly.from_serial_modules(db, ['base'] + ['l_15'] * 5)
        robot = assembly.to_pin_robot()
        self.assertEqual(robot.njoints, 0)
        self.assertAlmostEqual(robot.mass, assembly.mass)
        self.assertIsInstance(robot.fk(), Transformation)
        goal = ToleratedPose(nominal=Transformation.from_translation((9, 9, 9)))
        q, success = robot.ik(goal)
        self.assertIsInstance(q, np.ndarray)
        self.assertEqual(q.size, robot.njoints)
        self.assertFalse(success)

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

    def test_possible_connections(self):
        possibilities = self.db.possible_connections

        # Make sure there are no unnecessary duplicates
        def switch_mod(con):
            return con[2], con[1], con[0], con[3]  # When A -> B, B -> A

        def switch_cnctr(con):
            return con[0], con[3], con[1], con[1]  # Could only happen when A == B

        def switch_both(con):
            return switch_cnctr(switch_mod(con))  # Combines the tests above

        for connection in possibilities:
            if connection[0] is not connection[2]:  # Here of course, the assertion would always fail
                self.assertFalse(switch_mod(connection) in possibilities)
            self.assertFalse(switch_cnctr(connection) in possibilities)
            self.assertFalse(switch_both(connection) in possibilities)

        for module in self.db:  # Make sure every module can be connected to at least one other
            to_others = (c for c in possibilities if module in c and (c[0] is not c[2]))
            self.assertTrue(any(True for _ in to_others))  # No need to create the whole generator

        # Build some random robots and make sure the aforementioned connections actually work out
        N = 5  # How many modules will be added to the random robot?
        for _ in range(10):  # Run this test multiple times due to randomness
            base = random.choice(list(self.db.bases))
            assembly = ModuleAssembly(self.db, [base.id])
            for i in range(N):
                assembly.add_random_from_db(db_filter=lambda x: x not in self.db.bases.union(self.db.end_effectors))

    def test_robot_from_assembly(self):

        assembly = ModuleAssembly(self.db)
        for _ in range(12):
            assembly.add_random_from_db(lambda x: x not in self.db.end_effectors)

        robot = assembly.to_pin_robot()  # This is the crucial part

        self.assertAlmostEqual(assembly.mass, robot.mass)

        # Visualize in a random configuration (meshcat visualizer, will be closed when tests are done)
        robot.update_configuration(np.random.random((robot.njoints,)))
        robot.visualize(coordinate_systems='joints')

        # Ensure propagation of joint dynamics
        joints = [j for j in assembly.assembly_graph.nodes if isinstance(j, Joint)]
        for j in joints:
            self.assertTrue(j.gear_ratio in robot.model.rotorGearRatio)
            self.assertTrue(j.motor_inertia in robot.model.rotorInertia)
            self.assertTrue(j.friction_viscous in robot.model.damping)
            self.assertTrue(j.friction_coulomb in robot.model.friction)


if __name__ == '__main__':
    unittest.main()
