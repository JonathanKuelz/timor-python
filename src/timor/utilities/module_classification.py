#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.05.22
from enum import IntEnum
from typing import Dict, Tuple

from timor.Bodies import Connector
from timor.Module import ModuleBase, ModulesDB


class ModuleClassificationError(Exception):
    """This can be raised when a module should be assigned a unique type but the classification is not possible"""


class ModuleType(IntEnum):
    """
    This class implements a simplified but useful distinction of modules in four types: Base, Joint, Link and Eef.

    Be aware that this classification might be an oversimplification, regarding the current module at hand and use
    with care. There is no such distinction in general - however, in the last decades multiple works have published
    modules that can clearly be assigned to one of these types, and if it is possible, this utility class can be
    used to annotate them.
    """

    BASE = 0
    LINK = 1
    JOINT = 2
    END_EFFECTOR = 3


def divide_db_in_types(db: ModulesDB, strict: bool = True) -> Tuple[ModulesDB, ModulesDB, ModulesDB, ModulesDB]:
    """
    Takes one module db and splits it into subsets of bases, links, joints and end-effectors if possible.

    In general, this distinction is an oversimplification, so in cases a module is not clearly one of the possible
    types, this will raise an Exception.

    :param db: Any input modules db with modules that can be uniquely assigned to a ModuleType
    :param strict: If true, this method will raise an Error if there are any doubts about the uniqueness of the
      classification.
    :return: Four DBs in the following order: (Bases, Links, Joints, End-Effectors)
    :raises: ModuleClassificationError
    """
    splits: Dict[ModuleType, ModulesDB] = {mod_type: ModulesDB() for mod_type in ModuleType}
    for module in db:
        module_type = get_module_type(module, strict=strict)
        splits[module_type].add(module)

    return tuple(splits[mod_type] for mod_type in ModuleType)


def get_module_type(module: ModuleBase, strict: bool = True) -> ModuleType:
    """
    Tries to get the module type from Module data.

    Be aware the distinction is not always unique, so only use this method if you know your module set.

    :param module: The module to be classified.
    :param strict: If true, this method will raise an Error if there are any doubts about the uniqueness of the
        classification. There are no doubts if there is:

        * Exactly one base connector, one body, zero joints and one other connector OR
        * Exactly one eef connector, one body, zero joints and one other connector OR
        * One or more joints with a "serial" setup and no special connectors OR
        * Exactly one body, no special connectors and no joints
    :return: A Module Type
    """
    is_unique = True
    if 'base' in (connector.type for connector in module.available_connectors.values()):
        module_type = ModuleType.BASE
        is_unique &= module.num_bodies == 1 and module.num_joints == 0 and len(module.available_connectors) == 2
        is_unique &= not all(connector.type == 'base' for connector in module.available_connectors.values())
    elif 'eef' in (connector.type for connector in module.available_connectors.values()):
        module_type = ModuleType.END_EFFECTOR
        is_unique &= len(module.bodies) == 1 and module.num_joints == 0 and len(module.available_connectors) == 2
    elif module.num_joints > 0:
        module_type = ModuleType.JOINT
        is_unique &= len(module.bodies) == module.num_joints + 1
        for mod_part, degree in module.module_graph.degree:
            # A connected directed spatial graph is a chain iff:
            if isinstance(mod_part, Connector):
                is_unique &= degree == 2
            else:
                is_unique &= degree == 4
    else:
        if len(module.bodies) == 0:
            raise ValueError("Empty Module")
        module_type = ModuleType.LINK
        is_unique &= len(module.bodies) == 1

    if strict and not is_unique:
        raise ModuleClassificationError(f"Cannot classify module {module.id}, {module.name}.")

    return module_type
