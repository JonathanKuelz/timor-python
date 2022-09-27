from copy import copy
import itertools
import unittest

import pinocchio as pin

from timor import Bodies
from timor import Geometry
from timor import Joints
from timor import Module
from timor.utilities import spatial
import timor.utilities.errors as err
from timor.utilities.module_classification import ModuleClassificationError, ModuleType, get_module_type, divide_db_in_types  # noqa: E501
from timor.utilities.transformation import Transformation


class TestModule(unittest.TestCase):
    """Tests properties and methods of the Module implementation."""

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

    def test_classification(self):
        """Tests the classification utilities for simple modules"""
        bodies = []
        for i, body_id in enumerate(('small', 'medium', 'big', 'xl', 'xxl')):
            box = Geometry.Box({'x': i, 'y': i, 'z': i})
            bodies.append(
                Bodies.Body(body_id, collision=box)
            )
        b1, b2, b3, b4, b5 = bodies
        jnt = Joints.Joint('joint', 'revolute', parent_body=b2, child_body=b3)

        c1 = Bodies.Connector('con_1', parent=b2)
        c2 = Bodies.Connector('con_2', parent=b3)
        c3 = Bodies.Connector('con_3', parent=b4)
        c4 = Bodies.Connector('con_4', parent=b4)

        module_headers = []
        for module_id in ('mod_link', 'mod_joint', 'mod_base', 'mod_eef'):
            module_headers.append(
                Module.ModuleHeader(module_id, name=module_id + '_test')
            )
        link_header, joint_header, base_header, eef_header = module_headers
        link_module = Module.AtomicModule(link_header, bodies=[b1])
        joint_module = Module.AtomicModule(joint_header, bodies=[b2, b3], joints=[jnt])
        base_module = Module.AtomicModule(base_header, bodies=[b4])
        eef_module = Module.AtomicModule(eef_header, bodies=[b5])

        self.assertIs(get_module_type(link_module), ModuleType.LINK)
        with self.assertRaises(ModuleClassificationError):
            # In strict mode, there need to be connectors to certainly assign the joint type
            self.assertIs(get_module_type(joint_module), ModuleType.JOINT)
        self.assertIs(get_module_type(link_module, strict=False), ModuleType.LINK)
        self.assertIs(get_module_type(joint_module, strict=False), ModuleType.JOINT)

        # When adding the connectors, strict mode should work
        b2.connectors.add(c1)
        b3.connectors.add(c2)
        self.assertIs(get_module_type(joint_module), ModuleType.JOINT)

        base_connector = Bodies.Connector('BC', parent=b4, connector_type='base')
        eef_connector = Bodies.Connector('BC', parent=b5, connector_type='eef')
        b4.connectors.add(base_connector)
        b5.connectors.add(c3)
        b5.connectors.add(eef_connector)

        with self.assertRaises(ModuleClassificationError):
            get_module_type(base_module)  # No second, "outgoing" connector
        self.assertIs(get_module_type(base_module, strict=False), ModuleType.BASE)
        self.assertIs(get_module_type(eef_module), ModuleType.END_EFFECTOR)

        b4.connectors.add(c4)

        db = Module.ModulesDB({base_module, link_module, joint_module, eef_module})
        for split in divide_db_in_types(db):
            self.assertEqual(len(split), 1)

    def test_connector(self):
        """Tests connector instantiation and internal methods"""

        f = Bodies.Gender.female
        m = Bodies.Gender.male
        h = Bodies.Gender.hermaphroditic

        first = Bodies.Connector('1', spatial.random_homogeneous(), gender=f, connector_type='test', size=80)
        second = Bodies.Connector('2', spatial.random_homogeneous(), gender=m, connector_type='test', size=80)
        third = Bodies.Connector('3', spatial.random_homogeneous(), gender=h, connector_type='test', size=80)
        fourth = Bodies.Connector('4', spatial.random_homogeneous(), gender=h, connector_type='test', size=80)
        # The last two are designed to match no other, either due to size or type
        fifth = Bodies.Connector('5', spatial.random_homogeneous(), gender=h, connector_type='test', size=120)
        sixth = Bodies.Connector('6', spatial.random_homogeneous(), gender=h, connector_type='test_other', size=80)
        all_connectors = (first, second, third, fourth, fifth, sixth)

        # First, let's do a custom sanity checks
        self.assertTrue(first.connects(second))  # Basic male - female
        self.assertTrue(third.connects(fourth))  # Basic hermaphroditic - hermaphroditic

        # Check incompatibility to others due to type and size
        for other in all_connectors:
            if other is not fifth:
                self.assertFalse(fifth.connects(other))
            if other is not sixth:
                self.assertFalse(sixth.connects(other))

        # Check gender (in) compatibility
        for con in all_connectors:
            if con.gender is h:
                self.assertTrue(con.connects(con))  # Hermaphroditic always should work with an identical twin
            else:
                self.assertFalse(con.connects(con))  # Male/Female never connects to identical twin

        combinations = list(itertools.combinations_with_replacement(all_connectors, r=2))

        # Check Symmetry
        for c1, c2 in combinations:
            self.assertEqual(c1.connects(c2), c2.connects(c1))

    def test_joint(self):
        """Tests joint instantiation and internal methods.

        Quite Vanilla, as the Joint class is not much more than a wrapper for joint data
        """
        c = Bodies.Connector('1', spatial.random_homogeneous())
        some_geometry = Geometry.EmptyGeometry({})
        parent = Bodies.Body('1', collision=some_geometry)
        child = Bodies.Body('2', collision=some_geometry)
        joint = Joints.Joint('1', Joints.TimorJointType.revolute, parent, child,
                             parent2joint=spatial.random_homogeneous(),
                             joint2child=spatial.random_homogeneous())

        self.assertEqual(joint.parent2joint, joint.joint2parent.inv)
        self.assertEqual(joint.child2joint, joint.joint2child.inv)

        pin_type = joint.pin_joint_type
        pin_joint = pin_type()
        self.assertIsInstance(pin_joint, pin.JointModelRZ)

        parent.connectors.add(c)
        child.connectors.add(c)
        with self.assertRaises(err.UniqueValueError):
            joint = Joints.Joint(1, Joints.TimorJointType.revolute, parent, child)

    def test_module(self):
        """Tests a module on instantiation, uniqueness of contained elements, and hand-crafted IDs"""
        # Setup: Some connectors, bodies and a joint - important is, that some ID's overlap
        c1 = Bodies.Connector('1', spatial.homogeneous([.25, 0, 0]), connector_type='flange')
        c1_cpy = Bodies.Connector('1', spatial.homogeneous([.25, 0, 0]), connector_type='flange')
        c2 = Bodies.Connector('2', spatial.homogeneous([.25, 0, 0]), connector_type='flange')
        c2_cpy = Bodies.Connector('2', spatial.homogeneous([.25, 0, 0]), connector_type='flange')
        c3 = Bodies.Connector('3', spatial.homogeneous([.25, 0, 0]), connector_type='flange')
        c4 = Bodies.Connector('4', spatial.homogeneous([-.25, 0, 0]), connector_type='flange')
        c5 = Bodies.Connector('5', spatial.homogeneous([-.5, 0, 0]), connector_type='flange')

        box = Geometry.Box({'x': 1, 'y': 1, 'z': 1})
        generic_body = Bodies.Body('1', collision=box, connectors=(c1, c2))
        another_body = Bodies.Body('10', collision=box)
        box_body = Bodies.Body('2', collision=box, connectors=(c3, c4))
        box_body_same_connectors = Bodies.Body('3', collision=box, connectors=(c2_cpy, c1_cpy))
        same_id_body = Bodies.Body('2', collision=box)
        standalone_body = Bodies.Body('100', collision=box, connectors=[c5])

        jnt = Joints.Joint('1', Joints.TimorJointType.prismatic, parent_body=generic_body, child_body=box_body)
        jnt_fails = Joints.Joint('1', Joints.TimorJointType.revolute,
                                 parent_body=Bodies.Body('11', collision=box),
                                 child_body=Bodies.Body('12', collision=box))
        jnt_works = Joints.Joint('2', Joints.TimorJointType.revolute, parent_body=box_body, child_body=another_body)

        header_one = Module.ModuleHeader('1', 'Test module one', ['Jonathan'], ['jonathan.kuelz@tum.de'], ['TUM'])
        header_two = dict(ID='2', name='Test module two', author=['Jonathan'],
                          email=['jonathan.kuelz@tum.de'], affiliation=['TUM'])
        # ----- Let's go -----
        module = Module.AtomicModule(header_one, [generic_body, box_body, another_body], [jnt, jnt_works])

        with self.assertRaises(ValueError):
            # Joint body missing in module
            mod = Module.AtomicModule(header_one, [generic_body], [jnt])
        with self.assertRaises(err.UniqueValueError):
            # Same connector ID for different bodies
            mod = Module.AtomicModule(header_one, [generic_body, box_body_same_connectors])
        with self.assertRaises(err.UniqueValueError):
            mod = Module.AtomicModule(header_one, [generic_body, box_body, jnt_fails.parent_body, jnt_fails.child_body],
                                      [jnt, jnt_fails])

        new_joint = Joints.Joint('5', Joints.TimorJointType.revolute, parent_body=Bodies.Body('1000', collision=box),
                                 child_body=Bodies.Body('1001', collision=box))
        new = Module.AtomicModule(header_two, [generic_body, box_body, new_joint.parent_body, new_joint.child_body],
                                  [jnt, new_joint])

        self.assertEqual(Bodies.ConnectorSet(new.available_connectors.values()),
                         generic_body.connectors
                         .union(box_body.connectors)
                         .union(another_body.connectors)
                         )

        # ------ Test the DB -----
        new_almost_empty = Module.AtomicModule(header_two, [standalone_body])
        db = Module.ModulesDB([module])
        with self.assertRaises(err.UniqueValueError):
            db.add(new)
        with self.assertRaises(err.UniqueValueError):
            Module.ModulesDB((module, new))
        db.add(new_almost_empty)
        self.assertEqual(len(db), 2)

        db = Module.ModulesDB([module])
        new_only_joint = Module.AtomicModule(header_two, [jnt_fails.parent_body, jnt_fails.child_body], [jnt_fails])
        with self.assertRaises(err.UniqueValueError):
            db.add(new_only_joint)


if __name__ == '__main__':
    unittest.main()
