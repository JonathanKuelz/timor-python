#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.05.22
from dataclasses import dataclass
import itertools
from typing import Dict, Tuple

import networkx as nx
import numpy as np

from timor.Bodies import Gender
from timor.Module import AtomicModule, ModuleAssembly, ModulesDB
from timor.configuration_search.AssemblyIterator import AssemblyIterator
from timor.utilities import module_classification


def alphanumeric_sort_db(db: ModulesDB) -> Tuple[AtomicModule, ...]:
    """Helper method to sort a module db alphabetically"""
    return tuple(sorted(db, key=lambda mod: mod.id))


@dataclass
class Indices:
    """Wrapper to collect the current state/module indexed when iterating over a library"""

    base: int
    links: np.ndarray  # Type hint from np 1.22 on would be: npt.NDArray[np.ushort]
    joints: int
    valid_joint_combination_id: int
    eef: int


class Science2019(AssemblyIterator):
    """
    This iterator returns assemblies according to the criteria defined in the paper:

      "Effortless creation of safe robots from timor.Modules through self-programming and self-verification"
      by M. Althoff, A. Giusti, S.B. Liu, A.Pereira,
      https://mediatum.ub.tum.de/doc/1506779/gy8gyea2c5vyn65ely4bnzom6.Althoff-2019-ScienceRobotics.pdf

    Iteration happens in a depth-first manner, iterating over links the fastest, then joint combinations, then
    end-effectors and slowest over bases.

    This class is named after the author who initially came up with the idea of the here implemented search space
    reduction, Stefan Liu.

    The following constraints reduce the search space of possible assemblies:
      - The robot is a serial chain of modules. Every module has a proximal (=female) and a distal (=male) connector.
      - A maximum number of degrees of freedom (default: 6)
      - There are never two joints mounted directly after one another
      - There is a maximum of three static links in between two joints
      - The first module must be a joint module (this is not explicitly mentioned in the paper but can be derived from
        the modules they used)
      - The last module before the end-effector must be a joint. This can theoretically be relaxed in the future.
    """

    min_dof: int  # The minimum number of degrees of freedom for any assembly
    max_dof: int  # The maximum allowed number of degrees of freedom for any assembly
    min_links_between_joints: int  # The minimum number of links between two joints in an assembly
    max_links_between_joints: int  # The maximum number of links between two joints in an assembly
    bases: Tuple[AtomicModule]  # All "base modules" available  in the underlying modules DB
    links: Tuple[AtomicModule]  # All "link modules" available  in the underlying modules DB
    joints: Tuple[AtomicModule]  # All "joint modules" available  in the underlying modules DB
    end_effectors: Tuple[AtomicModule]  # All "end effector modules" available  in the underlying modules DB

    module_connects: Dict[str, Tuple[str, ...]]  # A mapping from timor.Module IDs modules it can be attached to
    link_combinations: Tuple[Tuple[str, ...], ...]  # All allowed combinations of links between two joints
    # All possible combinations of joints from base to eef in one assembly
    joint_combinations: Tuple[Tuple[str, ...], ...]

    # Maps (parent_id, child_id) to a tuple of tuples of link IDs that can be used between them
    links_between_joints: Dict[Tuple[str, str], Tuple[Tuple[int, ...], ...]]

    # Maps (augmented_base, augmented_eef) tuples to indices of valid joint combinations that can go in between
    joint_combinations_for_base_eef: Dict[Tuple[str, str], Tuple[int]]

    _state: Indices  # Holds all information necessary to determine where the iterator is currently
    _current_joint_combination: Tuple[str, ...]  # Unpacked from _state to decrease access time
    _current_num_joint_combinations: int  # The length of the current entry in "joint_combinations_for_base_eef"
    _current_num_joint_modules: int  # Depends on _current_joint_combination. Stored for low access time

    def __init__(self, db: ModulesDB,
                 min_dof: int = 0,
                 max_dof: int = 6,
                 min_links_between_joints: int = 0,
                 max_links_between_joints: int = 3,
                 max_links_before_first_joint: int = 0,
                 max_links_after_last_joint: int = 0
                 ):
        """
        This implementation of the LiuIterator allows for a bit more flexibility than the one mentioned in the paper:

        :param db: The database to build the assembly from. To work properly, every module in the db should clearly
          belong to one of the four categories {base, joint module, link module, end-effector}. Sane behavior for
          databases with multi-joint or other complex modules is not guaranteed
        :param min_dof: The minimum number of joints in the robot.
        :param max_dof: The maximum number of joints in the robot. The end-effector is not counted as a degree of
          freedom. Changing the default from 6 to another value would mean to differ from the paper. Other than the
          method from the referenced paper, min_dof is not implicitly equal to max_dof.
        :param min_links_between_joints: The minimum number of links between two joints in an assembly.
        :param max_links_between_joints: The maximum number of links between two joints. Default 3 is from Liu paper.
        :param max_links_before_first_joint: The maximum number of links between a base and the first joint.
        :param max_links_after_last_joint: The maximum number of links between the last joint and the end-effector.
        """
        if min_dof > max_dof:
            raise ValueError("Minimum number of dof cannot be larger than maximum number of dof.")
        self.min_dof = min_dof
        self.max_dof = max_dof
        self.min_links_between_joints = min_links_between_joints
        self.max_links_between_joints = max_links_between_joints
        super().__init__(db)
        bases, links, joints, end_effectors = module_classification.divide_db_in_types(self.db)
        self.bases = alphanumeric_sort_db(bases)
        self.links = alphanumeric_sort_db(links)
        self.joints = alphanumeric_sort_db(joints)
        self.end_effectors = alphanumeric_sort_db(end_effectors)
        self._check_db_valid()
        extra_links = int(bool(max_links_before_first_joint > 0)) + int(bool(max_links_after_last_joint > 0))
        self.max_links_before_first_joint = max_links_before_first_joint
        self.max_links_after_last_joint = max_links_after_last_joint
        self._state = Indices(base=0, links=np.zeros((max_dof - 1 + extra_links,), np.ushort), joints=0,
                              valid_joint_combination_id=0, eef=0)
        self.module_connects = self._extract_connections()
        self.link_combinations = self._identify_link_combinations()
        self.links_between_joints = self._identify_links_between_joints()
        self.joint_combinations = self._identify_joint_combinations()
        b, e = self._identify_augmented_base_eef()
        self.augmented_bases: Tuple[Tuple[str, ...], ...] = b
        self.augmented_end_effectors: Tuple[Tuple[str, ...], ...] = e
        self.joint_combinations_for_base_eef: Dict = self._identify_base_chain_eef()

        self._current_joint_combination = self.joint_combinations[self.state.joints]
        self._current_num_joint_combinations = len(self.valid_joint_combinations)
        self._current_num_joint_modules = len(self.joint_combinations[self.state.joints])

    def __next__(self) -> ModuleAssembly:
        """
        Iterate over:

          - Bases
          - Joints+Links
          - End-Effectors

        in a depth first manner.
        Really rough estimation: 0.2ms per call on 1 processor

        :raises: StopIteration
        """
        modules = list()
        if self.state.base >= len(self.augmented_bases):
            raise StopIteration
        modules.extend(self.augmented_bases[self.state.base])

        for i, (parent_id, child_id) in enumerate(zip(
                self._current_joint_combination[:-1], self._current_joint_combination[1:])):
            modules.append(parent_id)
            modules.extend(self.links_between_joints[(parent_id, child_id)][self.state.links[i]])

        if len(self._current_joint_combination) > 0:
            modules.append(self._current_joint_combination[-1])  # Last joint is not added in the previous loop

        modules.extend(self.augmented_end_effectors[self.state.eef])

        self._increment_state()
        return ModuleAssembly.from_serial_modules(self.db, modules)

    def reset(self) -> None:
        """Resets the indices used to iterate over modules."""
        self._state = Indices(base=0, links=np.zeros((self.max_dof - 1,), np.ushort), joints=0,
                              valid_joint_combination_id=0, eef=0)

    @property
    def current_base(self) -> Tuple[str, ...]:
        """Utility property to return the IDs of the currently used augmented base"""
        return self.augmented_bases[self.state.base]

    @property
    def current_eef(self) -> Tuple[str, ...]:
        """Utility property to return the IDs of the currently used augmented end-effector"""
        return self.augmented_end_effectors[self.state.eef]

    @property
    def state(self) -> Indices:
        """Implementation as property necessary to define abstractmethod"""
        if self._state.joints == self._state.base == self._state.links.sum() \
                == self._state.valid_joint_combination_id == 0:
            # This re-attribution is necessary to start with a valid combination after reset
            idx = (self.augmented_bases[0], self.augmented_end_effectors[0])
            try:
                self._state.joints = self.joint_combinations_for_base_eef[idx][0]
            except IndexError:
                self._state.joints = ()
        return self._state

    @property
    def valid_joint_combinations(self) -> Tuple[int]:
        """Returns a tuple of integers indexing the self.joint_combinations attribute.

        This tuple contains only these indices that are valid for the current combination of base and end-effector.
        """
        return self.joint_combinations_for_base_eef[self.current_base, self.current_eef]

    def _check_db_valid(self):
        """Performs some sanity checks on the given db to validate made assumptions on its structure.

        Those are:
        - There's at least one base, joint link and end-effector [crucial]
        - All modules are connectable [not necessary, but instantiating two iterators would be preferred if not]
        - Every link and joint module has a distal/male and a proximal/female part [crucial]
        - All bases connect to adjacent modules using the same connector/interface [rp]
        - All end effectors connect to adjacent modules using the same connector/interface [rp]
          rp = "relaxation possible but adaption of class methods necessary"
        """
        if not all(len(sub_db) > 0 for sub_db in (self.bases, self.links, self.joints, self.end_effectors)):
            raise ValueError("The module database must contain at least one base, joint, link and end-effector each.")

        if not nx.is_weakly_connected(self.db.connectivity_graph):
            raise ValueError(f"{self.__class__.__name__} expects all modules in the db can be connected.")

        for module in itertools.chain(self.links, self.joints):
            serial_error = False
            connectors = module.available_connectors.values()
            if len(connectors) != 2:
                serial_error = True
            if Gender.m not in (con.gender for con in connectors):
                serial_error = True
            if Gender.f not in (con.gender for con in connectors):
                serial_error = True
            if serial_error:
                raise ValueError(f"There is no unique way to connect {module} to an existing serial chain.")

    def _extract_connections(self) -> Dict[str, Tuple[str, ...]]:
        """Maps modules to possible connecting modules

        Note: this unidirectional extraction of possible connections is possible because this class works with the
        crucial simplification that we have unidirectional modules with a distal and proximal part!
        :return: For each module in the DB, extracts which modules can follow
        """
        connections = self.db.possible_connections
        connects = {mod.id: [] for mod in self.db}
        for (mod_a, con_a, mod_b, con_b) in connections:
            if con_a.gender is Gender.m and con_b.gender is Gender.f:
                connects[con_a.id[0]].append(con_b.id[0])
            elif con_a.gender is Gender.f and con_b.gender is Gender.m:
                connects[con_b.id[0]].append(con_a.id[0])
            else:
                raise ValueError("Unexpected Connection - hermaphroditic connectors not supported.")

        # Cast to tuple to boost performance and indicate this is actually an immutable property
        return {key: tuple(val) for key, val in connects.items()}

    def _extract_eef_distance(self):
        """
        For each module, calculates the "shortest path" to the end effector,

        :return:
        """
        raise NotImplementedError()

    def _identify_augmented_base_eef(self) -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[Tuple[str, ...], ...]]:
        """Extracts all possible "augmented" bases and end effectors.

        What we do here is defining an extended base and extended end-effector. Every possible combination of base and
        attached links as well as every possible combination of static links and end-effector is extracted and will be
        treated as a different base/eef "module" from here on.

        :return: A tuple of two tuples. The first tuple contains all possible combinations of base and attached links.
            The second tuple contains all possible combinations of static links and end-effector.
        """
        candidates_base = tuple(lc for lc in self.link_combinations if len(lc) <= self.max_links_before_first_joint)
        candidates_eef = tuple(lc for lc in self.link_combinations if len(lc) <= self.max_links_after_last_joint)
        bases = list()
        eefs = list()
        for base in self.bases:
            after_base = [()] + list(lc for lc in candidates_base
                                     if len(lc) > 0
                                     and lc[0] in self.module_connects[base.id])
            bases.extend((base.id,) + lc for lc in after_base)
        for eef in self.end_effectors:
            before_eef = [()] + list(lc for lc in candidates_eef
                                     if len(lc) > 0
                                     and eef.id in self.module_connects[lc[-1]])
            eefs.extend(lc + (eef.id,) for lc in before_eef)
        return tuple(bases), tuple(eefs)

    def _identify_base_chain_eef(self):
        """For all augmented bases, end-effectors, returns a mapping to functional joint combinations in between"""
        base_eef_combinations = dict()
        for base, eef in itertools.product(self.augmented_bases, self.augmented_end_effectors):
            base_eef_combinations[base, eef] = tuple(
                i for i, joints in enumerate(self.joint_combinations) if
                len(joints) > 0
                and joints[0] in self.module_connects[base[-1]]
                and eef[0] in self.module_connects[joints[-1]]
            )
            try:
                i = self.joint_combinations.index(())
                if eef[0] in self.module_connects[base[-1]]:
                    base_eef_combinations[base, eef] += (i,)
            except ValueError:
                pass
        return base_eef_combinations

    def _identify_link_combinations(self) -> Tuple[Tuple[str, ...], ...]:
        """Extracts possible tuples of links as they could be built in an assembly.

        Note: Without any filters, this will generate
          sum_over_0_max_in_between_links [num_different_links^max_links_between_joints]
        combinations (which may become infeasible large in max_links_between_joints is not chosen carefully).
        Even with filters, this can grow rapidly!
        :return: A tuple with all possible (allowed) combinations of links that can be in between two joints
        """
        if len(self.links) ** self.max_links_between_joints > 1e8:
            est_id_size = 100  # 2 char string ~100 bytes
            tuple_size = 24
            combination_size = (self.max_links_between_joints * est_id_size + tuple_size) / 1024 ** 2
            memory = len(self.links) ** self.max_links_between_joints * combination_size
            raise ValueError("You are trying to set up a very large iterator. Even the preparations could consume"
                             f"too much memory. Rough estimate: {memory} MB")
        link_combinations = tuple(itertools.chain.from_iterable(
            itertools.product((link.id for link in self.links), repeat=i)
            for i in range(self.min_links_between_joints, self.max_links_between_joints + 1)
        ))
        # In a chain, we have to check that the distal (male) connector of a link can connect the proximal (female)
        # connector of its successor
        valid_children = {parent.id: [child.id for child in self.links if any(
            parent_connector.connects(child_connector)
            for parent_connector in parent.available_connectors.values() if parent_connector.gender is Gender.m
            for child_connector in child.available_connectors.values() if child_connector.gender is Gender.f
        )] for parent in self.links}

        return tuple(combination for combination in link_combinations
                     if
                     all(child in valid_children[parent] for parent, child in zip(combination[:-1], combination[1:])))

    def _identify_joint_combinations(self) -> Tuple[Tuple[str, ...], ...]:
        """
        :return: A tuple with all possible (allowed) combinations of joints, ordered from base to eef
        """
        joint_dof = {joint_module.id: joint_module.num_joints for joint_module in self.joints}
        joint_ids = tuple(joint_dof.keys())
        combinations = list(itertools.chain.from_iterable(
            itertools.product(joint_ids, repeat=i) for i in range(0, self.max_dof + 1)
        ))

        # More filters, e.g. 'some joint never in last position' can be builtin here
        # But remember: Smart generation is more efficient than brute force generation and then filtering
        def dof_filter(combination) -> bool:
            return self.min_dof <= sum(joint_dof[jid] for jid in combination) <= self.max_dof

        can_attach_eef = tuple(joint.id for joint in self.joints if
                               any(joint.can_connect(eef) for eef in self.end_effectors))

        def eef_filter(combination) -> bool: return combination[-1] in can_attach_eef if len(combination) > 0 else True

        can_attach_base = tuple(joint.id for joint in self.joints if
                                any(joint.can_connect(base) for base in self.bases))

        def base_filter(combination) -> bool: return combination[0] in can_attach_base if len(combination) > 0 else True

        def can_connect_filter(combination) -> bool:
            """Checks whether two joints can in any way be used after each other.

            Returns true if for every two adjacent joints in the module chain, there is at least one combination
            of links that makes connecting them possible
            """
            return all(len(self.links_between_joints.get((combination[i], combination[i + 1]), [])) > 0
                       for i in range(len(combination) - 1)
                       )

        return tuple(c for c in combinations
                     if dof_filter(c)
                     and can_connect_filter(c)
                     and eef_filter(c)
                     and base_filter(c))

    def _identify_links_between_joints(self) -> Dict[Tuple[str, str], Tuple[Tuple[int, ...], ...]]:
        """Enumerates all possible combinations of links in between two joints

         Returns a mapping from every possible combination of parent and child joint to link combinations that can be
         in between.
        :return: A mapping from (parent_joint_id, child_joint_id) to a tuple of tuples, each of which contains between
          one and max_links_between_joints integers that index the classes link modules
        """
        mapping = dict()
        joint_ids = (mod.id for mod in self.joints)
        for key in itertools.product(joint_ids, repeat=2):
            parent, child = key
            mapping[key] = list()
            for combination in self.link_combinations:
                if len(combination) == 0:  # special case: joint modules are directly connected
                    if child in self.module_connects[parent]:
                        mapping[key].append(tuple())
                    continue
                first_link, last_link = combination[0], combination[-1]

                if first_link in self.module_connects[parent] and child in self.module_connects[last_link]:
                    mapping[key].append(combination)

        return {key: tuple(val) for key, val in mapping.items() if len(val) > 0}

    def _increment_state(self):
        """Call after every iteration to return the next element when calling __next__ the next time"""
        # -2 because the last index is 1 smaller than the length of the tuple, but also there are no links behind
        # the last joint (we call that an extended end effector), so we start with the second last joint
        self._increment_link_combination(self._current_num_joint_modules - 2)

    def _increment_link_combination(self, idx: int):
        """Iterates over link combinations available"""
        if self._current_num_joint_modules <= 1:
            self._increment_joint_combination()  # There are no links in between joints here
            return
        parent_joint, child_joint = self._current_joint_combination[idx: idx + 2]
        if self.state.links[idx] < len(self.links_between_joints[(parent_joint, child_joint)]) - 1:
            self.state.links[idx] += 1
        else:
            self.state.links[idx] = 0
            if idx > 0:
                # Iterate over the combinations one joint up the chain
                self._increment_link_combination(idx - 1)
            else:
                # For this joint combination, we iterated over all link combinations
                self._increment_joint_combination()

    def _increment_joint_combination(self):
        """Iterates over joint combinations available"""
        self.state.valid_joint_combination_id += 1
        if self.state.valid_joint_combination_id < self._current_num_joint_combinations:
            self.state.joints = self.valid_joint_combinations[self.state.valid_joint_combination_id]
        else:
            self.state.valid_joint_combination_id = 0
            self.state.joints = self.valid_joint_combinations[self.state.valid_joint_combination_id]
            self._increment_end_effector()

        self._current_joint_combination = self.joint_combinations[self.state.joints]
        self._current_num_joint_modules = len(self._current_joint_combination)

    def _increment_end_effector(self):
        """Iterates over end effectors available"""
        if self.state.eef < len(self.augmented_end_effectors) - 1:
            self.state.eef += 1
        else:
            self.state.eef = 0
            self._increment_base()

    def _increment_base(self):
        """Iterates over bases. Does not raise StopIteration, this needs to be done in the __next__ method!"""
        self.state.base += 1

    def __len__(self):
        """Calculates and returns the total number of possible assemblies created by this iterator."""
        num_assemblies = 0
        for key, joint_combination_ids in self.joint_combinations_for_base_eef.items():
            for idx in joint_combination_ids:
                joint_arrangement = self.joint_combinations[idx]
                assemblies_for_this_arrangement = 1
                for parent, child in zip(joint_arrangement[:-1], joint_arrangement[1:]):
                    assemblies_for_this_arrangement *= len(self.links_between_joints[(parent, child)])
                num_assemblies += assemblies_for_this_arrangement
        return num_assemblies
