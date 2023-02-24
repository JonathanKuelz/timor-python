import logging
import random
from time import time
import unittest

import numpy as np
import pinocchio as pin

from timor import ModuleAssembly, ModulesDB, PinRobot
from timor.Geometry import Sphere
from timor.task.Obstacle import Obstacle
from timor.task.Task import Task, TaskHeader
from timor.utilities import spatial
from timor.utilities.file_locations import robots


class TestCollisionUtilities(unittest.TestCase):
    """Test collision checking."""
    def setUp(self) -> None:
        self.package_dir = robots['panda'].parent
        self.urdf = robots['panda'].joinpath('urdf').joinpath('panda.urdf')
        self.robot = PinRobot.from_urdf(self.urdf, self.package_dir)
        self.modules = ModulesDB.from_name('geometric_primitive_modules')
        self.header = TaskHeader('Collision Test')
        random.seed(123)
        np.random.seed(123)

    def test_assembly_to_robot_collisions(self):
        improv_db = ModulesDB.from_name('IMPROV')
        improv_assembly = ModuleAssembly.from_serial_modules(improv_db, ('1', '21', '4', '21', '5', '23', '7', '12'))
        improv_assembly_2 = ModuleAssembly.from_serial_modules(improv_db, ('1', '21', '4', '21', '5', '23', '7', '12'),
                                                               ignore_collisions='rigid_via_joint')

        simple_modules = ('base', 'J2', 'i_30', 'J1', 'l_15', 'eef')
        simple_assembly = ModuleAssembly.from_serial_modules(self.modules, simple_modules)

        robot_colliding = simple_assembly.to_pin_robot(ignore_collisions='rigid')
        robot_functional = simple_assembly.to_pin_robot(ignore_collisions='via_joint')

        self.assertTrue(robot_colliding.has_self_collision())
        self.assertFalse(robot_functional.has_self_collision())
        for _ in range(100):
            q = robot_colliding.random_configuration()
            robot_colliding.update_configuration(q)
            self.assertTrue(robot_colliding.has_self_collision())

        # The IMPROV module set requires the "multi_joint" collision mode to move to collision free configurations
        robot_all = improv_assembly.to_pin_robot(ignore_collisions='rigid')
        robot_neighbor = improv_assembly.to_pin_robot(ignore_collisions='via_joint')
        robot_no_collisions = improv_assembly.to_pin_robot(ignore_collisions='rigid_via_joint')

        known_collision_free = (  # Some poses that should be collision free judging from the plots
            np.zeros(robot_all.njoints),
            np.array((0, 0, np.pi / 2, 0, 0, -np.pi / 2)),
            np.array((0, -np.pi / 4, np.pi / 8, np.pi / 8, -np.pi / 6, 0))
        )
        for conf in known_collision_free:
            # Make sure the "multi_joint" mode is necessary and working
            self.assertTrue(robot_all.has_self_collision(conf))
            self.assertTrue(robot_neighbor.has_self_collision(conf))
            self.assertFalse(robot_no_collisions.has_self_collision(conf))
            self.assertFalse(improv_assembly_2.robot.has_self_collision(conf))

        # Make sure there still are possibilities to collide for the no_collision robot
        for i in range(1000):
            if robot_no_collisions.has_self_collision(robot_no_collisions.random_configuration()):
                logging.debug("Found a self collision after {} iterations.".format(i))
                break
        else:
            self.assertTrue(False, "No self collision found after 1000 iterations.")

    def test_robot_self_collision(self):
        self.assertFalse(self.robot.has_self_collision())

        t0 = time()
        timeout = 10
        while True:
            q = pin.randomConfiguration(self.robot.model)
            self.robot.update_configuration(q)
            if self.robot.has_self_collision():
                break
            if time() - t0 > timeout:
                raise TimeoutError("Took unexpectedly long to find a self collision")

        q0 = pin.neutral(self.robot.model)
        self.assertFalse(self.robot.has_self_collision(q0))
        self.robot.update_configuration(q0)
        self.assertTrue(self.robot.has_self_collision(q))  # The collision detected previously

    def test_task_collisions(self):
        self.robot.update_configuration(pin.neutral(self.robot.model))
        task = Task(self.header)
        self.assertFalse(self.robot.has_collisions(task))

        random_configurations = [pin.randomConfiguration(self.robot.model) for _ in range(100)]
        collides_self = np.zeros((len(random_configurations),), dtype=bool)
        for i, q in enumerate(random_configurations):
            self.robot.update_configuration(q)
            self.assertEqual(self.robot.has_self_collision(), self.robot.has_collisions(task))
            collides_self[i] = self.robot.has_self_collision(q)

        sphere = Sphere({'r': .3}, pose=spatial.homogeneous(np.array([.5, .5, .3])))
        obstacle = Obstacle('sphere', sphere, name='Sphere')
        task.add_obstacle(obstacle)
        for q, self_collision in zip(random_configurations, collides_self):
            self.robot.update_configuration(q)
            if not self_collision:
                if self.robot.has_collisions(task):
                    self.assertEqual(self.robot.collisions(task), [(self.robot, obstacle)])
            else:
                self.assertTrue(len(self.robot.collisions(task)) in (1, 2))


if __name__ == '__main__':
    unittest.main()
