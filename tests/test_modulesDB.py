from pathlib import Path
import random
import unittest

import networkx as nx
import numpy as np

from timor import Joint, Transformation
from timor.Module import AtomicModule, ModuleAssembly, ModuleHeader, ModulesDB
from timor.utilities import logging
from timor.utilities.file_locations import get_module_db_files, robots
from timor.utilities.tolerated_pose import ToleratedPose


class TestModulesDB(unittest.TestCase):
    """Test if all known good module DBs are loadable, class is working, I/O from/to json, assemblies, ..."""

    def setUp(self) -> None:
        """Defines the filepaths before testing"""
        self.module_set_name = 'IMPROV'
        self.json_file = get_module_db_files(self.module_set_name)
        self.db = ModulesDB.from_json_file(self.json_file)
        random.seed(123)
        np.random.seed(123)

    def test_dump_json(self):
        """Tests loading and dumping from and to json."""
        db = ModulesDB.from_json_file(self.json_file)
        json_string = db.to_json_string()
        db_two = ModulesDB.from_json_string(json_string)
        json_again = db_two.to_json_string()

        self.assertEqual(json_string, json_again)
        self.assertEqual(db.all_module_ids, db_two.all_module_ids)
        self.assertTrue(nx.is_isomorphic(db.connectivity_graph, db_two.connectivity_graph))

    def test_empty(self):
        """Test that classes can be instantiated empty"""
        minimum_header = ModuleHeader('id', 'name')
        mod = AtomicModule(minimum_header)  # Header required
        mod = AtomicModule.from_json_data({'header': minimum_header.asdict()}, Path())
        assembly = ModuleAssembly.empty()
        robot = assembly.to_pin_robot()
        with self.assertRaises(LookupError):
            assembly.add_random_from_db()  # The db is empty

    def test_equality(self):
        """Test that two DBs with same modules and attributes are equal"""
        db1 = ModulesDB.from_name('IMPROV')
        db2 = ModulesDB.from_name('geometric_primitive_modules')
        self.assertNotEqual(db1, db2)
        self.assertEqual(db1, db1)
        self.assertEqual(db2, db2)
        self.assertFalse(db1 == ModulesDB(tuple(db1)))
        self.assertNotEqual(db1, ModulesDB(tuple(db1)))  # Different names
        self.assertEqual(db1, ModulesDB(tuple(db1), name=db1.name, **db1._model_generation_kwargs))
        self.assertEqual(db2, ModulesDB(tuple(db2), name=db2.name, **db2._model_generation_kwargs))

    def test_load_all_from_file(self):
        for db_name, db_dir in robots.items():
            if len(tuple(db_dir.rglob('*.json'))) == 0:
                logging.debug('Not trying to load db {} - probably not a database'.format(db_name))
                continue  # Not a json DB file in there
            db = ModulesDB.from_name(db_name)
            self.assertGreater(len(db), 0, msg=f"{db_name} is empty after loading")

    def test_jointless_robot(self):
        """Test the Assembly to Robot procedure for a robot without a joint"""
        db = ModulesDB.from_name('geometric_primitive_modules')
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

        q, success = robot.ik(ToleratedPose(robot.fk()))
        self.assertTrue(success)
        self.assertEqual(q.size, robot.njoints)

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
