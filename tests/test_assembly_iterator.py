import unittest

from timor.Module import ModulesDB
from timor.configuration_search.LiuIterator import Science2019


class AssemblyIteratorTest(unittest.TestCase):
    """Test assembly iterators."""
    def setUp(self) -> None:
        self.modrob_db = ModulesDB.from_name('PROMODULAR')

    def test_liu_iterator(self):
        modrob_iterator = Science2019(self.modrob_db, max_dof=3, max_links_between_joints=1)
        lots_of_assemblies = tuple(modrob_iterator)
        assembly_modules = tuple(assembly.internal_module_ids for assembly in lots_of_assemblies)
        self.assertEqual(len(modrob_iterator), 95418)
        self.assertEqual(len(lots_of_assemblies), len(modrob_iterator))  # custom len is correct
        self.assertEqual(len(lots_of_assemblies), len(set(assembly_modules)))

        # Check validity by converting the assemblies to a robot. Take a subsample only due to time constraints
        for assembly in lots_of_assemblies[::2000]:
            robot = assembly.to_pin_robot()

        j = 0
        stop_at = 20
        modrob_iterator.reset()
        # Another way to check there are valid robots
        for j, assembly in enumerate(modrob_iterator):
            robot = assembly.to_pin_robot()
            if j >= stop_at:
                break
        self.assertEqual(j, stop_at)  # Make sure we actually tested 100 samples

        with self.assertRaises(ValueError):
            # Make sure to prevent instantiating iterators where the preparations alone already take too long
            iterator = Science2019(self.modrob_db, max_dof=10, max_links_between_joints=5)

    def test_science_paper_liu(self):
        """
        Tests that this iterator really yields the same combinations as the ones in the science paper.
        Applies some fixes not really intended to do with this iterator, but necessary to get to the same combinations.
        This is mostly due to the domain knowledge applied by liu et al. which is specific to the schunk robot...
        """
        schunk_db = ModulesDB.from_name('IMPROV')
        not_in_paper = ('L7', 'L10', 'PB22', 'PB24')
        for module_name in not_in_paper:
            schunk_db.remove(schunk_db.by_name[module_name])
        iterator = Science2019(schunk_db, min_dof=6, max_links_after_last_joint=1)

        # The possible combination of powerballs is actually hard coded in the paper
        iterator.joint_combinations = (('21', '21', '23'), ('21', '21', '21'))
        real_links = {'4', '5', '14', '15'}  # No extensions

        def paper_compliant_combo(links: tuple[str]) -> bool:
            is_link = tuple(link for link in links if link in real_links)
            if len(is_link) != 1:
                return False
            if len(links) == 3:
                if links[1] not in real_links:
                    return False
            return True

        link_combinations_in_paper = tuple(link_combination for link_combination in iterator.link_combinations
                                           if paper_compliant_combo(link_combination))
        iterator.link_combinations = link_combinations_in_paper

        # Make the fixes from above work
        iterator.links_between_joints = iterator._identify_links_between_joints()
        iterator.augmented_end_effectors = tuple(aeef for aeef in iterator.augmented_end_effectors if not any(
            real_link in aeef for real_link in real_links))  # Only extensions are allowed before the end effector
        iterator.joint_combinations_for_base_eef = iterator._identify_base_chain_eef()
        iterator._current_joint_combination = iterator.joint_combinations[iterator.state.joints]
        iterator._current_num_joint_combinations = len(iterator.valid_joint_combinations)
        iterator._current_num_joint_modules = len(iterator.joint_combinations[iterator.state.joints])

        expected_num_assemblies = 32768  # Number taken from published MATLAB implementation
        self.assertEqual(len(iterator), expected_num_assemblies)

        i = 0
        for _ in iterator:
            i += 1
        self.assertEqual(i, expected_num_assemblies)  # Iterator really yields __len__ combinations


if __name__ == '__main__':
    unittest.main()
