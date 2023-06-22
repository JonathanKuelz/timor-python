import itertools
import pickle
import unittest

import networkx as nx
import numpy as np
import pinocchio as pin

from timor import Bodies, Body, Geometry, Joint, Transformation
from timor import AtomicModule, ModuleHeader, ModulesDB
from timor.Joints import TimorJointType
from timor.parameterized import ParameterizedCylinderLink, ParameterizedOrthogonalJoint, ParameterizedOrthogonalLink, \
    ParameterizedStraightJoint
from timor.utilities import spatial
import timor.utilities.errors as err
from timor.utilities.module_classification import ModuleClassificationError, ModuleType, get_module_type, \
    divide_db_in_types


class TestModule(unittest.TestCase):
    """Tests properties and methods of the Module implementation."""

    def test_classification(self):
        """Tests the classification utilities for simple modules"""
        bodies = []
        for i, body_id in enumerate(('small', 'medium', 'big', 'xl', 'xxl')):
            box = Geometry.Box({'x': i, 'y': i, 'z': i})
            bodies.append(
                Bodies.Body(body_id, collision=box)
            )
        b1, b2, b3, b4, b5 = bodies
        jnt = Joint('joint', 'revolute', parent_body=b2, child_body=b3)

        c1 = Bodies.Connector('con_1', parent=b2)
        c2 = Bodies.Connector('con_2', parent=b3)
        c3 = Bodies.Connector('con_3', parent=b4)
        c4 = Bodies.Connector('con_4', parent=b4)

        module_headers = []
        for module_id in ('mod_link', 'mod_joint', 'mod_base', 'mod_eef'):
            module_headers.append(
                ModuleHeader(module_id, name=module_id + '_test')
            )
        link_header, joint_header, base_header, eef_header = module_headers
        link_module = AtomicModule(link_header, bodies=[b1])
        joint_module = AtomicModule(joint_header, bodies=[b2, b3], joints=[jnt])
        base_module = AtomicModule(base_header, bodies=[b4])
        eef_module = AtomicModule(eef_header, bodies=[b5])

        self.assertIs(get_module_type(link_module), ModuleType.LINK)
        with self.assertRaises(ModuleClassificationError):
            # In strict mode, there need to be connectors to certainly assign the joint type
            self.assertIs(get_module_type(joint_module), ModuleType.JOINT)
        self.assertIs(get_module_type(link_module, strict=False), ModuleType.LINK)
        self.assertIs(get_module_type(joint_module, strict=False), ModuleType.JOINT)

        # When adding the connectors, strict mode should work
        b2.connectors.add(c1)
        b3.connectors.add(c2)
        joint_module._module_graph = joint_module._build_module_graph()  # Fix of hacking new connectors in module
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

        db = ModulesDB({base_module, link_module, joint_module, eef_module})
        for split in divide_db_in_types(db):
            self.assertEqual(len(split), 1)

    def test_connector(self):
        """Tests connector instantiation and internal methods"""

        f = Bodies.Gender.female
        m = Bodies.Gender.male
        h = Bodies.Gender.hermaphroditic

        first = Bodies.Connector('1', Transformation.random(), gender=f, connector_type='test', size=80)
        first_duplicate = Bodies.Connector('1', first.body2connector, gender=f, connector_type='test', size=80)
        second = Bodies.Connector('2', Transformation.random(), gender=m, connector_type='test', size=80)
        third = Bodies.Connector('3', Transformation.random(), gender=h, connector_type='test', size=80)
        fourth = Bodies.Connector('4', Transformation.random(), gender=h, connector_type='test', size=80)
        # The last two are designed to match no other, either due to size or type
        fifth = Bodies.Connector('5', Transformation.random(), gender=h, connector_type='test', size=120)
        sixth = Bodies.Connector('6', Transformation.random(), gender=h, connector_type='test_other', size=80)
        all_connectors = (first, second, third, fourth, fifth, sixth)

        # First, let's do a custom sanity checks
        self.assertTrue(first.connects(second))  # Basic male - female
        self.assertTrue(third.connects(fourth))  # Basic hermaphroditic - hermaphroditic
        self.assertEqual(first, first_duplicate)  # Equality check
        for other in all_connectors[1:]:
            self.assertNotEqual(first, other)  # Inequality check

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
        c = Bodies.Connector('1', Transformation.random())
        some_geometry = Geometry.EmptyGeometry({})
        parent = Bodies.Body('1', collision=some_geometry)
        child = Bodies.Body('2', collision=some_geometry)
        joint = Joint('1', TimorJointType.revolute, parent, child,
                      parent2joint=Transformation.random(),
                      joint2child=Transformation.random())

        self.assertEqual(joint.parent2joint, joint.joint2parent.inv)
        self.assertEqual(joint.child2joint, joint.joint2child.inv)

        pin_type = joint.pin_joint_type
        pin_joint = pin_type()
        self.assertIsInstance(pin_joint, pin.JointModelRZ)

        parent.connectors.add(c)
        child.connectors.add(c)
        with self.assertRaises(err.UniqueValueError):
            joint = Joint(1, TimorJointType.revolute, parent, child)

        self.assertEqual(joint, joint)

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

        jnt = Joint('1', TimorJointType.prismatic, parent_body=generic_body, child_body=box_body)
        jnt_fails = Joint('1', TimorJointType.revolute,
                          parent_body=Bodies.Body('11', collision=box),
                          child_body=Bodies.Body('12', collision=box))
        jnt_works = Joint('2', TimorJointType.revolute, parent_body=box_body, child_body=another_body)

        header_one = ModuleHeader('1', 'Test module one', author=['Jonathan'], email=['jonathan.kuelz@tum.de'],
                                  affiliation=['TUM'])
        header_two = dict(ID='2', name='Test module two', author=['Jonathan'],
                          email=['jonathan.kuelz@tum.de'], affiliation=['TUM'])

        # ----- Let's go -----
        module = AtomicModule(header_one, [generic_body, box_body, another_body], [jnt, jnt_works])
        same_module = AtomicModule(header_one, [generic_body, box_body, another_body], [jnt, jnt_works])

        self.assertEqual(module, same_module)
        self.assertEqual(module.mass, generic_body.mass + box_body.mass + another_body.mass)

        same_module.bodies.pop()  # Don't do this at home
        self.assertNotEqual(module, same_module)

        with self.assertRaises(ValueError):
            # Joint body missing in module
            mod = AtomicModule(header_one, [generic_body], [jnt])
        with self.assertRaises(err.UniqueValueError):
            # Same connector ID for different bodies
            mod = AtomicModule(header_one, [generic_body, box_body_same_connectors])
        with self.assertRaises(err.UniqueValueError):
            mod = AtomicModule(header_one, [generic_body, box_body, jnt_fails.parent_body, jnt_fails.child_body],
                               [jnt, jnt_fails])

        new_joint = Joint('5', TimorJointType.revolute, parent_body=Bodies.Body('1000', collision=box),
                          child_body=Bodies.Body('1001', collision=box))
        new = AtomicModule(header_two, [generic_body, box_body, new_joint.parent_body, new_joint.child_body],
                           [jnt, new_joint])
        self.assertNotEqual(module, new)

        self.assertEqual(Bodies.ConnectorSet(new.available_connectors.values()),
                         generic_body.connectors
                         .union(box_body.connectors)
                         .union(another_body.connectors)
                         )

        # ------ Test the DB -----
        new_almost_empty = AtomicModule(header_two, [standalone_body])
        db = ModulesDB([module])
        with self.assertRaises(err.UniqueValueError):
            db.add(new)
        with self.assertRaises(err.UniqueValueError):
            ModulesDB((module, new))
        db.add(new_almost_empty)
        self.assertEqual(len(db), 2)

        db = ModulesDB([module])
        new_only_joint = AtomicModule(header_two, [jnt_fails.parent_body, jnt_fails.child_body], [jnt_fails])
        with self.assertRaises(err.UniqueValueError):
            db.add(new_only_joint)

    def test_parameterizable_module(self):
        """Tests a parameterizable module on instantiation and resizing capabilities"""
        cyl_header = ModuleHeader('cylinder', 'cylinder')
        l_header = ModuleHeader('L', 'orthogonal link')
        cylinder = ParameterizedCylinderLink(cyl_header)
        l_shaped = ParameterizedOrthogonalLink(l_header, lengths=(1, 3), radius=.2)
        long_cylinder = ParameterizedCylinderLink(cyl_header, length=2)
        constrained_cylinder = ParameterizedCylinderLink(cyl_header, length=2, radius=1, limits=((.1, 1.1), (0, 4)))

        self.assertEqual(cylinder, cylinder)
        self.assertNotEqual(cylinder, l_shaped)
        self.assertNotEqual(cylinder, long_cylinder)
        self.assertNotEqual(cylinder, constrained_cylinder)

        bottom = cylinder.connectors_by_own_id['proximal']
        top = cylinder.connectors_by_own_id['distal']
        self.assertEqual(abs(bottom.body2connector.translation[2]), abs(top.body2connector.translation[2]))
        dz_initial = (bottom.connector2body @ top.body2connector).translation[2]
        self.assertEqual(abs(dz_initial), cylinder.length)

        initial_mass = cylinder.mass
        self.assertEqual(cylinder.mass, cylinder.link.mass)
        self.assertEqual(cylinder.mass, long_cylinder.mass / 2)
        cylinder.resize((1, 1.5))
        dz_new = (bottom.connector2body @ top.body2connector).translation[2]
        self.assertEqual(cylinder.mass, initial_mass * 1.5)
        self.assertEqual(dz_new, dz_initial * 1.5)
        cylinder.resize((1, 1))
        self.assertEqual(cylinder.mass, initial_mass)
        cylinder.resize((.33, 1.5))
        self.assertEqual(cylinder.radius, .33)

        with self.assertRaises(ValueError):
            cylinder.resize((-1, 1))
        with self.assertRaises(ValueError):
            constrained_cylinder.resize((2, 1))

        prox = l_shaped.connectors_by_own_id['proximal']
        dist = l_shaped.connectors_by_own_id['distal']
        trans_diff = (prox.connector2body @ dist.body2connector).translation
        self.assertEqual(np.linalg.norm(trans_diff), np.sqrt(l_shaped.l1 ** 2 + l_shaped.l2 ** 2))

        self.assertIsInstance(AtomicModule.from_json_data(cylinder.to_json_data(), None), AtomicModule)
        self.assertIsInstance(AtomicModule.from_json_data(l_shaped.to_json_data(), None), AtomicModule)
        self.assertIsInstance(cylinder.freeze(), AtomicModule)
        self.assertIsInstance(l_shaped.freeze(), AtomicModule)
        self.assertIsInstance(pickle.loads(pickle.dumps(cylinder)), ParameterizedCylinderLink)
        self.assertIsInstance(pickle.loads(pickle.dumps(l_shaped)), ParameterizedOrthogonalLink)

        # Check if freezing works even if the original parameterizable module changes
        frozen_cyl = cylinder.freeze()
        frozen_l = l_shaped.freeze()
        inertia_cyl = tuple(b.inertia for b in frozen_cyl.bodies)
        inertia_l = tuple(b.inertia for b in frozen_l.bodies)
        jsons = (frozen_cyl.to_json_data(), frozen_l.to_json_data())
        cylinder.resize((100, 100))
        l_shaped.resize((100, 100, 100))
        self.assertEqual(tuple(b.inertia for b in frozen_cyl.bodies), inertia_cyl)
        self.assertEqual(tuple(b.inertia for b in frozen_l.bodies), inertia_l)
        self.assertEqual(frozen_cyl.to_json_data(), jsons[0])
        self.assertEqual(frozen_l.to_json_data(), jsons[1])

        cylinder.link.parameters = (.2, .7)
        with self.assertRaises(ValueError):
            cylinder.freeze()
        cylinder.resize((.2, .7))
        self.assertIsInstance(cylinder.link.freeze(), Body)

    def test_parameterizable_module_with_joint(self):
        """Check parameterized modules that contain at least one joint"""
        straight_header = ModuleHeader('J1', 'Straight Prismatic Joint')
        straight = ParameterizedStraightJoint(straight_header, radius=.1, joint_type=TimorJointType.prismatic,
                                              joint_parameters={'q_limits': (-.3, .5)})
        orthogonal_header = ModuleHeader('J2', 'Orthogonal Revolute Joint')
        orthogonal = ParameterizedOrthogonalJoint(orthogonal_header, radius=.1, joint_type=TimorJointType.revolute)

        self.assertEqual(straight.joint.type, TimorJointType.prismatic)
        self.assertEqual(orthogonal.joint.type, TimorJointType.revolute)

        Gs = straight.module_graph
        self.assertTrue(nx.is_weakly_connected(Gs))
        nodes = {n._id: n for n in Gs.nodes}
        path = nx.shortest_paths.shortest_path(Gs, nodes['proximal'], nodes['distal'])
        end2end = Transformation.neutral()
        for n1, n2 in zip(path[:-1], path[1:]):
            end2end = end2end @ Gs.edges[n1, n2]['transform']
        self.assertAlmostEqual(abs(end2end.translation[2]), straight.length)

        Go = orthogonal.module_graph
        self.assertTrue(nx.is_weakly_connected(Go))
        nodes = {n._id: n for n in Go.nodes}
        path = nx.shortest_paths.shortest_path(Go, nodes['proximal'], nodes['distal'])
        end2end = Transformation.neutral()
        for n1, n2 in zip(path[:-1], path[1:]):
            end2end = end2end @ Go.edges[n1, n2]['transform']
        self.assertAlmostEqual(abs(end2end.translation[1]), orthogonal.l2)
        self.assertAlmostEqual(abs(end2end.translation[2]), orthogonal.l1)

        self.assertIsInstance(AtomicModule.from_json_data(straight.to_json_data(), None), AtomicModule)
        self.assertIsInstance(AtomicModule.from_json_data(orthogonal.to_json_data(), None), AtomicModule)
        self.assertIsInstance(straight.freeze(), AtomicModule)
        self.assertIsInstance(orthogonal.freeze(), AtomicModule)
        self.assertIsInstance(pickle.loads(pickle.dumps(straight)), ParameterizedStraightJoint)
        self.assertIsInstance(pickle.loads(pickle.dumps(orthogonal)), ParameterizedOrthogonalJoint)

        orthogonal.proximal_link.parameters = (.4, .6)
        with self.assertRaises(ValueError):
            orthogonal.freeze()
        orthogonal.resize((1, 2, 3))
        self.assertIsInstance(orthogonal.proximal_link.freeze(), Body)


if __name__ == '__main__':
    unittest.main()
