import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest

import hppfcl
import numpy as np
import numpy.testing as np_test
import pinocchio as pin

from timor import Geometry, Transformation
from timor import Robot
from timor.Module import ModuleAssembly
from timor.task import Constraints, Obstacle, Task, Tolerance
from timor.utilities import file_locations, prebuilt_robots, logging
from timor.utilities.tolerated_pose import ToleratedPose


class DummyTask:
    """A dummy class that implements everything that's needed to instantiate a solution without loading a functioning
    real task it is tailored for first."""

    def __init__(self, task_id: str):
        self.id = task_id


def allclose(x: np.ndarray, y: np.ndarray) -> bool:
    """Uses np.allclose with custom tolerance. 1e-6 is max. allowed abs. tolerance, no relative tolerance"""
    return np.allclose(x, y, atol=1e-6, rtol=0)


def pin_models_functionally_equal(m1: pin.Model, m2: pin.Model) -> bool:
    """
    Evaluates whether two pinocchio models are equal with regard to their kinematic and dynamic parameters,
    allowing different frames and numercial deviations to exist.
    Closely following original equality implementation:
    https://github.com/stack-of-tasks/pinocchio/blob/master/src/multibody/model.hpp (operator==)
    """
    checks = tuple((
        m1.nq == m2.nq,
        m1.nv == m2.nv,
        m1.njoints == m2.njoints,
        # No check for number of bodies! (everything is fine as long as the inertias align)
        # No check for frames!
        tuple(m1.parents) == tuple(m2.parents),  # cpp StdVec needs to be unpacked for a sane comparison
        # No check for names!
        tuple(tuple(st) for st in m1.subtrees) == tuple(tuple(st) for st in m2.subtrees),
        m1.gravity == m2.gravity,

        tuple(m1.idx_qs) == tuple(m2.idx_qs),
        tuple(m1.nqs) == tuple(m2.nqs),
        tuple(m1.idx_vs) == tuple(m2.idx_vs),
        tuple(m1.nvs) == tuple(m2.nvs),

        # No check for reference configurations

        # all(m1.rotorInertia == m2.rotorInertia),  # Not supported by URDF as it seems
        all(m1.friction == m2.friction),
        all(m1.damping == m2.damping),
        # m1.rotorGearRatio == m2.rotorGearRatio,  # Not supported by URDF as it seems
        all(m1.effortLimit == m2.effortLimit),
        all(m1.velocityLimit == m2.velocityLimit),
        all(m1.lowerPositionLimit == m2.lowerPositionLimit),
        all(m1.upperPositionLimit == m2.upperPositionLimit),

        # Allow minor deviations that might occur during (de)serialization of a model
        all(allclose(i1.mass, i2.mass) for i1, i2 in zip(m1.inertias, m2.inertias)),
        all(allclose(i1.lever, i2.lever) for i1, i2 in zip(m1.inertias, m2.inertias)),
        all(allclose(i1.inertia, i2.inertia) for i1, i2 in zip(m1.inertias, m2.inertias)),

        all(allclose(p1.homogeneous, p2.homogeneous) for p1, p2 in zip(m1.jointPlacements, m2.jointPlacements)),

        tuple(m1.joints) == tuple(m2.joints)
    ))
    return all(checks)


def _pin_geometry_objects_functionally_equal(o1: pin.GeometryObject, o2: pin.GeometryObject) -> bool:
    """
    Evaluates whether two collision Geometries are equal with regard to their collision/visual properties.

    Allows different names of collision objects.
    Closely following original equality implementation:
    https://github.com/stack-of-tasks/pinocchio/blob/master/src/multibody/fcl.hxx
    """
    def geometry_equal(geo1: hppfcl.CollisionGeometry, geo2: hppfcl.CollisionGeometry):
        """Provisional geometry equality check, as hppfcl also contains name properties that are not important here"""
        if not type(geo1) is type(geo2):
            return False
        if isinstance(geo1, (hppfcl.BVHModelOBB, hppfcl.BVHModelOBBRSS)):
            return all((
                (o1.geometry.vertices() == o2.geometry.vertices()).all(),
                all(geo1.tri_indices(i) == geo2.tri_indices(i) for i in range(geo1.num_tris))
            ))
        elif isinstance(geo1, hppfcl.Cylinder):
            return all((
                geo1.radius == geo2.radius,
                geo1.halfLength == geo2.halfLength
            ))
        elif isinstance(geo1, hppfcl.Box):
            return all(geo1.halfSide == geo2.halfSide)
        elif isinstance(geo1, hppfcl.Sphere):
            return geo1.radius == geo2.radius
        else:
            raise NotImplementedError()

    checks = (
        # No Name check (would be o1.name == o2.name)
        # No frame check, as num frames can differ between models (o1.parentFrame == o2.parentFrame)
        o1.parentJoint == o2.parentJoint,
        geometry_equal(o1.geometry, o2.geometry),
        allclose(o1.placement.homogeneous, o2.placement.homogeneous),
        # mesh path information is lost when directly transforming to robot, therefore no "o1.meshPath == o2.meshPath"
        # mesh scale information is auto-transformed when reading urdf, check for mesh equality is in geometry_equal
        o1.disableCollision is o2.disableCollision
    )
    return all(checks)


def pin_geometry_models_structurally_equal(m1: pin.GeometryModel, m2: pin.GeometryModel) -> bool:
    """
    Evaluates whether two pinocchio GeometryModels are equal with regard to their collision and visual properties,
    allowing different frames and numercial deviations to exist.
    Closely following original equality implementation:
    https://github.com/stack-of-tasks/pinocchio/blob/master/src/multibody/geometry.hpp (operator==)
    """
    checks = (
        m1.ngeoms == m2.ngeoms,
        all(_pin_geometry_objects_functionally_equal(o1, o2) for o1, o2 in
            zip(m1.geometryObjects, m2.geometryObjects)
            ),
        tuple(m1.collisionPairs) == tuple(m2.collisionPairs)
    )
    return all(checks)


def pin_geometry_models_functionally_equal(r1: Robot.RobotBase, r2: Robot.RobotBase,
                                           iterations: int = 100) -> float:
    """Tests if (collision) geometry models of two robots are functionally equal - find the same self-collisions."""
    difference = 0
    for i in range(iterations):
        q = r1.random_configuration()
        if r1.has_self_collision(q) != r2.has_self_collision(q):
            logging.warn(f"Self-collision differ for {q}")
    return difference / iterations


class JsonSerializationTests(unittest.TestCase):
    """Tests all relevant json (de)serialization methods"""

    def setUp(self) -> None:
        random.seed(99)
        np.random.seed(99)

        # Set up tasks, for which there are already some json files
        self.assets = file_locations.get_test_tasks()["asset_dir"]
        self.task_files = file_locations.get_test_tasks()['task_files']
        panda_loc = file_locations.robots['panda']
        self.robot = Robot.PinRobot.from_urdf(panda_loc.joinpath('urdf').joinpath('panda.urdf'), panda_loc.parent)

        # Set up some Obstacles
        random_homogeneous = Transformation.random()
        box = Obstacle.Obstacle('Box', collision=Geometry.Box({'x': 1, 'y': 1, 'z': -1}, Transformation.random()))
        cylinder = Obstacle.Obstacle('Cylinder',
                                     collision=Geometry.Cylinder({'r': 1, 'z': -1}, Transformation.random()))
        composed = Obstacle.Obstacle('Composed', collision=Geometry.ComposedGeometry((box.collision,
                                                                                      cylinder.collision)))
        self.obstacles = [box, cylinder, composed]

        # Set up some Tolerances
        cylindrical = Tolerance.CartesianCylindrical(r=[.5, 1.5], phi_cyl=[-np.pi / 8, np.pi / 4], z=[2., 2.5])
        axis_angles = Tolerance.RotationAxisAngle(n_x=[.5, .7], n_y=[0, .1], n_z=[-.2, .2], theta_r=[-0.1, 0.1])
        self.tolerances = [cylindrical, axis_angles]

        # Set up some Constraints
        goal_order = Constraints.GoalOrderConstraint(list(map(str, range(100))))
        joint_constraints = Constraints.JointLimits(('q', 'dq', 'ddq', 'tau'))
        base_placement = Constraints.BasePlacement(ToleratedPose(random_homogeneous, tolerance=cylindrical))
        collision = Constraints.CollisionFree()
        self.constraints = [goal_order, joint_constraints, base_placement, collision]

    def test_assembly_to_urdf(self):
        db_loc, db_assets = file_locations.get_module_db_files('IMPROV')
        for _ in range(5):
            assembly = prebuilt_robots.random_assembly(n_joints=6, modules_file=db_loc, package=db_assets)
            logging.info(f"Testing assembly {assembly.internal_module_ids}")

            r1 = assembly.to_pin_robot()
            r1._remove_home_collisions()  # This is done per default when loading from URDF
            with tempfile.NamedTemporaryFile() as tmp_urdf:
                urdf = assembly.to_urdf('test_robot', Path(tmp_urdf.name), replace_wrl=True)
                tmp_urdf.flush()
                r2 = Robot.PinRobot.from_urdf(Path(tmp_urdf.name), db_assets)
                self.assertTrue(pin_models_functionally_equal(r1.model, r2.model))
                self.assertLess(pin_geometry_models_functionally_equal(r1, r2), 0.1,
                                msg=f"Varying self-collisions for assembly {assembly}")
                self.assertTrue(pin_geometry_models_structurally_equal(r1.visual, r2.visual))

    def test_constraints_to_json(self):
        for constraint in self.constraints:
            as_json = json.dumps(constraint.to_json_data())
            from_json = json.loads(as_json)
            new = Constraints.ConstraintBase.from_json_data(from_json)
            from_string = Constraints.ConstraintBase.from_json_string(constraint.to_json_string())
            self.assertIs(type(new), type(constraint))
            self.assertIs(type(from_string), type(constraint))
            for key in constraint.__dict__:
                self.assertEqual(constraint.__dict__[key], new.__dict__[key])
                self.assertEqual(constraint.__dict__[key], from_string.__dict__[key])

    def test_load_then_dump_task(self):
        for task_file in self.task_files:
            task = Task.Task.from_json_file(task_file, self.assets)

            with tempfile.NamedTemporaryFile() as tmp_json_dump:
                task.to_json_file(Path(tmp_json_dump.name))
                tmp_json_dump.flush()  # Necessary to certainly write to tempfile!
                another_copy = Task.Task.from_json_file(Path(tmp_json_dump.name), self.assets)
                # This does not test 100% for equality but should cover the most central sanity checks
                self.assertEqual(task.header, another_copy.header)
                self.assertEqual(set(o.id for o in task.obstacles), set(o.id for o in another_copy.obstacles))
                self.assertEqual(set(g.id for g in task.goals), set(g.id for g in another_copy.goals))
                self.assertEqual(set(type(c) for c in task.constraints),
                                 set(type(c) for c in another_copy.constraints))

    def test_cost_function_to_json(self):
        pass

    def test_solution_to_json(self):
        # TODO Read sample solutions and make sure mostly same if dumped again
        pass

    def test_pickle_robot(self):
        """Tests whether the kinematic model remains when pickling - geometry cannot be preserved at the moment"""
        pickled = pickle.dumps(self.robot)
        unpickled = pickle.loads(pickled)
        self.assertTrue(pin_models_functionally_equal(self.robot.model, unpickled.model))

    def test_obstacles_to_json(self):
        package_dir = Path(__file__)  # Not used in this test, but needed for the method
        for obstacle in self.obstacles:
            as_json = json.dumps(obstacle.to_json_data())
            from_json = json.loads(as_json)
            new = Obstacle.Obstacle.from_json_data(dict(
                collision=from_json['collision'],
                package_dir=package_dir,
                ID=from_json['ID'],
                name=from_json['name']))
            new_from_string = Obstacle.Obstacle.from_json_string(obstacle.to_json_string())
            self.assertIs(type(new), type(obstacle))
            self.assertIs(type(new_from_string), type(obstacle))
            new_dict = new.__dict__.copy()
            newfs_dict = new_from_string.__dict__.copy()
            comp_dict = obstacle.__dict__.copy()
            for key in comp_dict:
                if key in ('_children', 'collision'):
                    continue
                np_test.assert_array_equal(new_dict[key], comp_dict[key])
                np_test.assert_array_equal(newfs_dict[key], comp_dict[key])

    def test_serial_module_assembly(self):
        db_loc, db_assets = file_locations.get_module_db_files('IMPROV')
        for _ in range(5):
            assembly = prebuilt_robots.random_assembly(n_joints=6, modules_file=db_loc, package=db_assets)
            modules = assembly.internal_module_ids
            original_ids = tuple(assembly._module_copies.get(mod_id, mod_id) for mod_id in modules)
            copied = ModuleAssembly.from_serial_modules(assembly.db, original_ids)
            self.assertEqual(assembly.internal_module_ids, copied.internal_module_ids)
            self.assertEqual(assembly.nJoints, copied.nJoints)

    def test_tolerances_to_json(self):
        for tolerance in self.tolerances:
            as_json = json.dumps(tolerance.to_projection())
            from_json = json.loads(as_json)
            new = Tolerance.AlwaysValidTolerance()
            for proj, val in zip(from_json['toleranceProjection'], from_json['tolerance']):
                new = new + Tolerance.ToleranceBase.from_projection(proj, val)
            self.assertIs(type(new), type(tolerance))
            for key in tolerance.__dict__:
                if isinstance(tolerance.__dict__[key], np.ndarray):
                    np_test.assert_array_equal(new.__dict__[key], tolerance.__dict__[key])
                else:
                    self.assertEqual(new.__dict__[key], tolerance.__dict__[key])


if __name__ == '__main__':
    unittest.main()
