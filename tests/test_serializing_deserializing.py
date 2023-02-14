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
from timor.Module import ModuleAssembly, ModulesDB
from timor.task import Constraints, CostFunctions, Obstacle, Solution, Task, Tolerance
from timor.utilities import file_locations, prebuilt_robots, logging
from timor.utilities.file_locations import get_module_db_files, schema_dir
from timor.utilities.prebuilt_robots import random_assembly
from timor.utilities.schema import get_schema_validator
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.trajectory import Trajectory


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


def robot_io_equal(m1: Robot.PinRobot, m2: Robot.PinRobot, n_iter: int = 1000) -> bool:
    """
    Make sure two robots behave the same way in kinematics and dynamics
    """
    for _ in range(n_iter):
        q, dq, ddq = m1.random_configuration(), m1.random_configuration(), m1.random_configuration()
        wrench = np.random.random((6,))
        if not all((allclose(m1.id(q, dq, ddq), m2.id(q, dq, ddq)),
                    allclose(m1.id(q, dq, ddq, eef_wrench=wrench), m2.id(q, dq, ddq, eef_wrench=wrench)),
                    m1.fk(q) == m2.fk(q),
                    allclose(m1.fd(ddq, q, dq), m2.fd(ddq, q, dq)))):
            logging.warn(f"Discrepancy for q = {q}, dq = {dq}, ddq / tau = {ddq}, wrench = {wrench}")
            return False
    return True


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


class SerializationTests(unittest.TestCase):
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
        eef_constraint_pose = Constraints.EndEffector(pose=ToleratedPose(Transformation.random()))
        eef_constraint_v = Constraints.EndEffector(velocity_lim=np.asarray((.1234, 0.2345)),
                                                   rotation_velocity_lim=np.asarray((-.1, .4)))
        eef_constraint_all = Constraints.EndEffector(pose=ToleratedPose(Transformation.random()),
                                                     velocity_lim=np.asarray((random.random(), random.random() + 1)),
                                                     rotation_velocity_lim=np.asarray((-random.random(),
                                                                                       random.random())))
        self.constraints = [goal_order, joint_constraints, base_placement, collision, eef_constraint_pose,
                            eef_constraint_v, eef_constraint_all]

    def test_assembly_to_serialized(self):
        """Tests both, assembly to pickle and to json"""
        for db_name in ('IMPROV', 'PROMODULAR', 'modrob-gen2'):
            logging.debug(f"Testing with {db_name}")
            db = ModulesDB.from_name(db_name)
            for _ in range(3):
                assembly = random_assembly(6, db)

                robot = assembly.to_pin_robot()
                assembly_description = assembly.to_json_data()
                self.assertEqual(set(assembly_description.keys()),
                                 {"moduleSet", "moduleOrder", "moduleConnection", "baseConnection", "basePose"})
                self.assertEqual(assembly_description['moduleSet'], db_name)

                reconstructed_assembly = ModuleAssembly.from_json_data(assembly_description, db)
                reconstructed_pickle = pickle.loads(pickle.dumps(assembly))
                new_jsons = [reconstructed_assembly.to_json_data(), reconstructed_pickle.to_json_data()]
                for key in assembly_description.keys():
                    for new_json in new_jsons:
                        if key == 'basePose':
                            np_test.assert_array_equal(new_json[key][0], assembly_description[key][0])
                        elif key == 'moduleConnection':
                            # JSON doesn't know sets, but the order of connections doesn't matter
                            self.assertEqual(set(new_json[key]), set(assembly_description[key]))
                        else:
                            self.assertEqual(new_json[key], assembly_description[key])

                np_test.assert_array_equal(assembly.adjacency_matrix, reconstructed_assembly.adjacency_matrix)
                self.assertEqual(assembly.base_module.id, reconstructed_assembly.base_module.id)
                self.assertEqual(assembly.internal_module_ids, reconstructed_assembly.internal_module_ids)
                self.assertEqual(assembly.original_module_ids, reconstructed_assembly.original_module_ids)

                reconstructed_robot = reconstructed_assembly.robot
                pin_models_functionally_equal(robot.model, reconstructed_robot.model)
                robot_io_equal(robot, reconstructed_robot)
                pin_geometry_models_structurally_equal(robot.visual, reconstructed_robot.visual)
                pin_geometry_models_structurally_equal(robot.collision, reconstructed_robot.collision)
                pin_geometry_models_functionally_equal(robot, reconstructed_robot)

    def test_assembly_to_urdf(self):
        _, db_assets = file_locations.get_module_db_files('IMPROV')
        db = ModulesDB.from_name('IMPROV')
        for _ in range(5):
            assembly = prebuilt_robots.random_assembly(n_joints=6, module_db=db)
            logging.info(f"Testing assembly {assembly.internal_module_ids}")

            r1 = assembly.to_pin_robot()
            r1._remove_home_collisions()  # This is done per default when loading from URDF
            with tempfile.NamedTemporaryFile() as tmp_urdf:
                urdf = assembly.to_urdf('test_robot', Path(tmp_urdf.name), replace_wrl=True)
                tmp_urdf.flush()
                r2 = Robot.PinRobot.from_urdf(Path(tmp_urdf.name), db_assets)
                self.assertTrue(pin_models_functionally_equal(r1.model, r2.model))
                self.assertLess(pin_geometry_models_functionally_equal(r1, r2), 0.01,
                                msg=f"Varying self-collisions for assembly {assembly}")
                self.assertTrue(pin_geometry_models_structurally_equal(r1.visual, r2.visual))

    def test_constraints_to_json(self):
        with self.assertRaises(KeyError):
            Constraints.ConstraintBase.from_json_data({"no-type": "really no type information"})

        with self.assertRaises(NotImplementedError):
            Constraints.ConstraintBase.from_json_data({"type": "thisIsARatherUnlikelyTypeStringToEverExist"})

        with self.assertRaises(ValueError):
            Constraints.EndEffector.from_json_data({"type": "basePlacement"})

        for constraint in self.constraints:
            as_json = json.dumps(constraint.to_json_data())
            new = Constraints.ConstraintBase.from_json_data(json.loads(as_json))
            # Also ensure 1-time readable Iterators are parsed
            from_string = Constraints.ConstraintBase.from_json_string(constraint.to_json_string())
            for n in (new, from_string):
                self.assertIs(type(n), type(constraint))
                for key in constraint.__dict__:
                    if isinstance(constraint.__dict__[key], np.ndarray):
                        np_test.assert_array_equal(constraint.__dict__[key], n.__dict__[key])
                    else:
                        self.assertEqual(constraint.__dict__[key], n.__dict__[key])

                if isinstance(constraint, Constraints.BasePlacement):  # Test that redirect for from_json_string works
                    from_string_2 = Constraints.BasePlacement.from_json_string(constraint.to_json_string())
                    self.assertIsNotNone(from_string_2)
                    with self.assertRaises(ValueError):
                        Constraints.EndEffector.from_json_string(constraint.to_json_string())

    def test_de_serialize_solution_and_task(self):
        assembly = prebuilt_robots.get_six_axis_assembly()
        q1 = assembly.robot.random_configuration()
        q2 = assembly.robot.random_configuration()

        task = Task.Task(
            Task.TaskHeader(1),
            goals=(
                Task.Goals.At("1", ToleratedPose(assembly.robot.fk(q1))),
                Task.Goals.At("2", ToleratedPose(assembly.robot.fk(q2)))
            ),
            constraints=(Task.Constraints.BasePlacement(ToleratedPose(Transformation.neutral())),
                         Task.Constraints.JointLimits(("q")))
        )
        cost_function = CostFunctions.CycleTime(0.2) + CostFunctions.MechanicalEnergy(0.5)
        solution = Solution.SolutionTrajectory(
            trajectory=Trajectory(t=np.asarray((0., 1.)), q=np.asarray((q1, q2)), goal2time={"1": 0., "2": 1.}),
            header=Solution.SolutionHeader(1),
            task=task,
            assembly=assembly,
            cost_function=cost_function,
            base_pose=assembly.robot.placement
        )

        solution_dict = solution.to_json_data()
        self.assertIsNotNone(json.dumps(solution_dict))
        _, validator = get_schema_validator(schema_dir.joinpath("SolutionSchema.json"))
        # make sure is storable
        with tempfile.NamedTemporaryFile(mode="w+") as t:
            json.dump(solution_dict, t)
            t.flush()
            with open(t.name) as f:
                validator.validate(json.load(f))
            loaded_sol = Solution.SolutionBase.from_json_file(Path(t.name), Path("/"), {task.id: task})

        # Also check the custom task
        with tempfile.NamedTemporaryFile(mode="w+") as t:
            task.to_json_file(t.name)
            t.flush()

        self.assertEqual(solution.valid, loaded_sol.valid)
        # Depends on robot, trajectory, position, cost function -> most important parts of solution tested
        self.assertEqual(solution.cost, loaded_sol.cost)
        self.assertEqual(solution.trajectory, loaded_sol.trajectory)
        self.assertEqual(solution.header, loaded_sol.header)

        with tempfile.NamedTemporaryFile(mode="w+") as t:
            json.dump({"test": "json_validator"}, t)
            t.flush()
            with self.assertRaises(ValueError):
                Solution.SolutionBase.from_json_file(Path(t.name), Path("/"), {task.id: task})

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
        db = ModulesDB.from_name('IMPROV')
        for _ in range(5):
            assembly = prebuilt_robots.random_assembly(n_joints=6, module_db=db)
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

    def test_wrap_monolithic_robot(self):
        panda = self.robot
        as_assembly = ModuleAssembly.from_monolithic_robot(panda)
        self.assertIs(panda, as_assembly.robot)

        fresh_robot = as_assembly.to_pin_robot()
        self.assertTrue(pin_models_functionally_equal(panda.model, fresh_robot.model))
        self.assertTrue(robot_io_equal(panda, fresh_robot))
        self.assertLess(pin_geometry_models_functionally_equal(panda, fresh_robot, iterations=1000), 0.01)
        self.assertTrue(pin_geometry_models_structurally_equal(panda.visual, fresh_robot.visual))

        # For the last check, we need to parse collision pairs as we do for URDFs
        fresh_robot = as_assembly.to_pin_robot(ignore_collisions='rigid')
        fresh_robot._remove_home_collisions()
        self.assertTrue(pin_geometry_models_structurally_equal(panda.collision, fresh_robot.collision))

        modular = prebuilt_robots.get_six_axis_modrob()
        as_assembly = ModuleAssembly.from_monolithic_robot(modular)
        self.assertIs(modular, as_assembly.robot)
        self.assertTrue(pin_models_functionally_equal(modular.model, as_assembly.robot.model))
        self.assertTrue(robot_io_equal(panda, fresh_robot))
        self.assertLess(pin_geometry_models_functionally_equal(modular, as_assembly.robot, iterations=1000), 0.01)
        self.assertTrue(pin_geometry_models_structurally_equal(modular.visual, as_assembly.robot.visual))

        # Test with more complex modular robot that also has joint inertias / friction / ...
        modular_complex = ModuleAssembly.from_serial_modules(
            ModulesDB.from_file(*get_module_db_files('IMPROV')),
            ["1", "21", "14", "21", "14", "23", "13"]
        ).robot
        fresh_robot = ModuleAssembly.from_monolithic_robot(modular_complex).robot
        self.assertTrue(pin_models_functionally_equal(modular_complex.model, fresh_robot.model))
        self.assertTrue(robot_io_equal(modular_complex, fresh_robot))
        self.assertLess(pin_geometry_models_functionally_equal(modular_complex, fresh_robot, iterations=1000), 0.01)
        self.assertTrue(pin_geometry_models_structurally_equal(modular_complex.visual, fresh_robot.visual))


if __name__ == '__main__':
    unittest.main()
