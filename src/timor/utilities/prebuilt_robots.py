#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 04.03.22
import numpy as np

from timor import Module, ModuleAssembly
from timor.Robot import PinRobot


def random_assembly(n_joints: int, module_db: Module.ModulesDB) -> Module.ModuleAssembly:
    """
    Creates a random assembly from modules.

    The number of modules will be n_joints * 2 + 1 (base) (possibly one less if eef cannot connect)

    :param n_joints: The number of non-base modules to add
    :param module_db: Where to draw modules from
    """
    assembly = Module.ModuleAssembly(module_db)
    assembly.add_random_from_db()  # First one by default is a base

    def no_dead_end(module):
        return len(module.available_connectors) > 1 and \
            not any(c.type == 'eef' for c in module.available_connectors.values())

    def has_joint(module: Module.ModuleBase) -> bool:
        return len(module.joints) > 0

    def add_joint(module):
        return no_dead_end(module) and has_joint(module)

    def add_link(module):
        return no_dead_end(module) and not has_joint(module)

    for i in range(n_joints):
        # Add joint-body-pair or joint-eef
        assembly.add_random_from_db(db_filter=add_joint)
        if i == n_joints - 1:
            for eef in module_db.end_effectors:  # Try all end effectors
                for connector in eef.available_connectors.values():
                    for free in assembly.free_connectors.values():
                        if connector.connects(free):
                            # As soon as we find the first match, we assemble and return
                            assembly.add(eef, connector.own_id,
                                         assembly.module_instances.index(free.parent.in_module), free.own_id)
                            return assembly
        else:
            assembly.add_random_from_db(db_filter=add_link)

    return assembly


def get_six_axis_modrob() -> PinRobot:
    """
    Creates a six axis robot with ca. 1 m reach
    """
    robot = get_six_axis_assembly().robot
    robot.update_configuration(np.array([np.pi / 2] * 6))  # At q = 0, 0, ..., the robot is in self collision
    return robot


def get_six_axis_assembly() -> ModuleAssembly:
    """
    Creates an assembly of a six axis robot with ca. 1 m reach
    """
    db = Module.ModulesDB.from_name('geometric_primitive_modules')
    assembly = Module.ModuleAssembly(
        database=db,
        assembly_modules=('base', 'J2', 'J2', 'l_45', 'J2', 'l_30', 'J2', 'J2', 'J2', 'eef'),
        connections=(
            (0, 'base2robot', 1, 'J2_proximal'), (1, 'J2_distal', 2, 'J2_proximal'), (2, 'J2_distal', 3, '8-0'),
            (3, '8-1', 4, 'J2_proximal'), (4, 'J2_distal', 5, '8-0'), (5, '8-1', 6, 'J2_proximal'),
            (6, 'J2_distal', 7, 'J2_proximal'), (7, 'J2_distal', 8, 'J2_proximal'), (8, 'J2_distal', 9, 'robot2eef')
        ),
        base_connector=(0, 'base')
    )
    return assembly
