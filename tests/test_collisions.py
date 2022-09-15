import random
from time import time
import unittest

import numpy as np
import pinocchio as pin

from timor.Robot import PinRobot
from timor.Geometry import Sphere
from timor.scenario.Obstacle import Obstacle
from timor.scenario.Scenario import Scenario, ScenarioHeader
from timor.utilities import spatial
from timor.utilities.file_locations import robots


class TestCollisionUtilities(unittest.TestCase):
    """Test collision checking."""
    def setUp(self) -> None:
        self.package_dir = robots
        self.urdf = self.package_dir.joinpath('panda').joinpath('urdf').joinpath('panda.urdf')
        self.robot = PinRobot.from_urdf(self.urdf, self.package_dir)
        self.header = ScenarioHeader('Collision Test')
        random.seed(123)
        np.random.seed(123)

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

    def test_scenario_collisions(self):
        self.robot.update_configuration(pin.neutral(self.robot.model))
        scenario = Scenario(self.header)
        self.assertFalse(self.robot.has_collisions(scenario))

        random_configurations = [pin.randomConfiguration(self.robot.model) for _ in range(100)]
        collides_self = np.zeros((len(random_configurations),), dtype=bool)
        for i, q in enumerate(random_configurations):
            self.robot.update_configuration(q)
            self.assertEqual(self.robot.has_self_collision(), self.robot.has_collisions(scenario))
            collides_self[i] = self.robot.has_self_collision(q)

        sphere = Sphere({'r': .3}, pose=spatial.homogeneous(np.array([.5, .5, .3])))
        obstacle = Obstacle('sphere', sphere, name='Sphere')
        scenario.add_obstacle(obstacle)
        for q, self_collision in zip(random_configurations, collides_self):
            self.robot.update_configuration(q)
            if not self_collision:
                if self.robot.has_collisions(scenario):
                    self.assertEqual(self.robot.collisions(scenario), [(self.robot, obstacle)])
            else:
                self.assertTrue(len(self.robot.collisions(scenario)) in (1, 2))


if __name__ == '__main__':
    unittest.main()
