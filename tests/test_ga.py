from typing import Union
import unittest

import numpy as np
import pygad

from timor import ModuleAssembly, ModulesDB
from timor.configuration_search.AssemblyFilter import default_filters
from timor.task import Task
from timor.configuration_search.GA import GA
from timor.utilities import file_locations
from timor.utilities.dtypes import Lexicographic


class TestGAProperties(unittest.TestCase):
    """Test properties of GA"""

    def setUp(self) -> None:
        self.task_file = file_locations.test_data.joinpath("sample_tasks/simple/PTP_1.json")
        self.db = ModulesDB.from_name('modrob-gen2')
        self.task = Task.Task.from_json_file(self.task_file, self.task_file.parent)
        self.filters = default_filters(self.task)

        # Custom hyperparameters to speed up the test
        self.hp = {'population_size': 4,
                   'num_parents_mating': 2,
                   'keep_parents': 2,
                   'keep_elitism': 2,
                   'num_generations': 20,
                   'mutation_probability': 0.11}

        self.fitness_func_calls = 0
        self.rng = np.random.default_rng(651321)

    def _fitness_func(self, assembly: ModuleAssembly, ga_instance: pygad.GA, solution_idx: int) -> float:
        """Fitness function for the genetic algorithm which is fast to evaluate."""
        self.fitness_func_calls += 1
        return assembly.mass

    def test_lexicographic_fitness(self):
        """Test if the lexicographic fitness function works as expected."""

        def lexi_fitness_func(assembly: ModuleAssembly, ga_instance: pygad.GA, solution_idx: int) -> Lexicographic:
            """
            Fitness function that considers the number of modules first, then the mass

            That means: The more modules an assembly has, the better it is. If two assemblies have the same number of
            modules, the one with the lower mass is better.
            """
            mass = assembly.mass
            num_modules = len(assembly.internal_module_ids)
            return Lexicographic(num_modules, -mass)

        ga = GA(self.db)
        ga_ins = ga.optimize(lexi_fitness_func, hp=self.hp)
        for i in range(len(ga_ins.best_solutions) - 1):
            self.assertLessEqual(ga_ins.best_solutions_fitness[i], ga_ins.best_solutions_fitness[i + 1])

        max_sanity_checks = 100
        cached_assemblies = ga.fitness_cache
        assemblies = sorted(cached_assemblies, key=lambda x: cached_assemblies[x])
        better = None
        for i in range(len(assemblies) - 1):
            if better is None:
                worse = ModuleAssembly.from_serial_modules(self.db, assemblies[i])
            else:
                worse = better
            better = ModuleAssembly.from_serial_modules(self.db, assemblies[i + 1])
            self.assertTrue(len(worse.internal_module_ids) <= len(better.internal_module_ids))
            if len(worse.internal_module_ids) == len(better.internal_module_ids):
                self.assertLessEqual(better.mass, worse.mass)

    def test_ga_setup(self):
        """Test that setting up a GA instance works as expected."""
        ga = GA(self.db, rng=self.rng)
        ga_ins = ga.optimize(self._fitness_func, hp=self.hp)
        fitness_calls = self.fitness_func_calls
        self.assertGreater(self.fitness_func_calls, 0)
        ga2 = GA(self.db.name, rng=self.rng)
        ga_ins2 = ga2.optimize(self._fitness_func, hp=self.hp)
        self.assertGreater(self.fitness_func_calls, fitness_calls)
        self.assertFalse(np.array_equal(ga_ins.initial_population, ga_ins2.initial_population))

    def test_callbacks(self):
        """Test the mutation and crossover callbacks provided by pyGAD work with custom operations."""
        ping_counts = {'mutation': 0, 'crossover': 0}

        def make_ping_callback(d: dict, key: str):
            """Creates a callable that increments d[key] and accepts any number of arguments."""
            def ping(a1, a2):
                d[key] += 1
            return ping

        ga = GA(self.db, rng=self.rng)
        callbacks = {'on_mutation': make_ping_callback(ping_counts, 'mutation'),
                     'on_crossover': make_ping_callback(ping_counts, 'crossover')}
        ga.optimize(self._fitness_func, hp=self.hp, **callbacks)
        self.assertEqual(ping_counts['crossover'], self.hp['num_generations'])
        self.assertEqual(ping_counts['mutation'], self.hp['num_generations'])

    def test_crossover(self):
        """Test if the functionality of single crossover is valid."""
        ga = GA(self.db, rng=self.rng)
        ga_ins = ga.optimize(self._fitness_func, hp=self.hp)
        for _ in range(10):
            test_parents = ga_ins.initial_population
            offspring_size = ga_ins.initial_population.shape
            crossover = ga.single_valid_crossover(test_parents, offspring_size, ga_ins)
            # Check if the shape of the offspring is correct
            self.assertEqual(crossover.shape, tuple(offspring_size))

            # Check if the crossover offspring is valid
            for i in range(len(crossover)):
                self.assertTrue(ga.check_individual(crossover[i]))

    def test_mutation(self):
        """Test if the functionality of mutation is valid."""
        ga = GA(self.db, rng=self.rng)
        ga_ins = ga.optimize(self._fitness_func, hp=self.hp)
        for _ in range(10):
            test_offspring = ga_ins.initial_population
            mutated_offspring = ga.mutation(test_offspring, ga_ins)
            # Check if the shape of the offspring remains the same after mutation
            self.assertEqual(test_offspring.shape, mutated_offspring.shape)

            for i in range(len(mutated_offspring)):
                # Check if the offspring is valid
                self.assertTrue(ga.check_individual(mutated_offspring[i]))
        ga_ins.mutation_probability = 0
        self.assertTrue(np.array_equal(ga.mutation(ga_ins.initial_population, ga_ins), ga_ins.initial_population))
        ga_ins.mutation_probability = 1
        self.assertFalse(np.array_equal(ga.mutation(ga_ins.initial_population, ga_ins), ga_ins.initial_population))

    def test_mutation_weights(self):
        weights = {mod.id: 1 for mod in self.db}
        weights['EMPTY'] = 1000
        ga = GA(self.db, mutation_weights=weights, rng=self.rng)
        no_empty = weights.copy()
        no_empty['EMPTY'] = 0
        ga_without_empty = GA(self.db, mutation_weights=no_empty, rng=self.rng)

        for _ in range(10):
            test_offspring = ga._get_initial_population(population_size=(5, 8))
            pygad_ga = GA(self.db, rng=self.rng).optimize(self._fitness_func, hp={'num_generations': 0})
            mutation = ga.mutation(test_offspring, pygad_ga)
            mutation_unweighted = ga_without_empty.mutation(test_offspring, pygad_ga)
            empty_modules_weighted = np.sum(mutation == ga.id2num['EMPTY'])
            empty_modules_unweighted = np.sum(mutation_unweighted == ga.id2num['EMPTY'])
            self.assertGreater(empty_modules_weighted, empty_modules_unweighted)

    def test_offspring_is_valid(self):
        """Test if the whole offspring and the best solution of each generation step is valid."""
        ga = GA(self.db, rng=self.rng)
        ga_ins = ga.optimize(self._fitness_func, hp=self.hp)
        for i in ga_ins.population:
            self.assertTrue(ga.check_individual(i))
        for i in ga_ins.best_solutions:
            self.assertTrue(ga.check_individual(i))

    def test_population_initialization(self):
        """Test if the initial population is valid."""
        ga = GA(self.db, rng=self.rng)

        choose_no_joint = ga.mutation_weights.copy()
        choose_no_joint.update(dict.fromkeys(ga.joint_ids, 0))
        choose_joints = ga.mutation_weights.copy()
        choose_joints.update(dict.fromkeys(ga.joint_ids, 1000))
        choose_no_empty = ga.mutation_weights.copy()
        choose_no_empty.update({'EMPTY': 0})
        choose_empty = ga.mutation_weights.copy()
        choose_empty.update({'EMPTY': 1000})

        for genome in ga._get_initial_population(population_size=(10, 10)):
            self.assertTrue(ga.check_individual(genome))
        joint_nums = {ga.id2num[jid] for jid in ga.joint_ids}
        for genome in ga._get_initial_population(population_size=(5, 5), module_weights=choose_no_joint):
            self.assertTrue(ga.check_individual(genome))
            self.assertTrue(len(joint_nums.intersection(genome)) == 0)
        for genome in ga._get_initial_population(population_size=(5, 5), module_weights=choose_no_empty):
            self.assertTrue(ga.check_individual(genome))
            self.assertTrue(ga.id2num['EMPTY'] not in genome)
        for genome in ga._get_initial_population(population_size=(5, 5), module_weights=choose_joints):
            self.assertTrue(ga.check_individual(genome))
            self.assertTrue(len(joint_nums.intersection(genome)) > 0)
        for genome in ga._get_initial_population(population_size=(5, 5), module_weights=choose_empty):
            self.assertTrue(ga.check_individual(genome))
            self.assertTrue(ga.id2num['EMPTY'] in genome)

    def test_selection(self):
        """Test custom selection methods"""
        def lexico_fitness_func(assembly: ModuleAssembly, ga_instance: pygad.GA, solution_idx: int) -> Lexicographic:
            """Just a dummy function"""
            return Lexicographic(1, 1)

        def lexico_to_scalar(lexi_object: Union[Lexicographic, np.ndarray]):
            """Converts a lexicographic object to a scalar"""
            return np.array([obj.values[0] for obj in lexi_object])

        ga = GA(self.db)
        ga_instance = ga.optimize(lexico_fitness_func, hp={'num_generations': 1}, selection_type='sss')
        lexicographic_fitness = np.array([Lexicographic(self.rng.integers(0, 10),
                                                        self.rng.random()) for _ in range(20)])

        # Test if the rank based selection with sp=2.0 performs worse than the steady state selection
        _, rank_based_selected_idx_sp_2 = ga.rank_based_selection(lexicographic_fitness, 5, ga_instance)
        _, steady_state_selected_idx = ga_instance.steady_state_selection(lexicographic_fitness, 5)
        rank_based_fitness_sp_2 = lexicographic_fitness[rank_based_selected_idx_sp_2]
        steady_state_fitness = lexicographic_fitness[steady_state_selected_idx]
        self.assertLessEqual(np.mean(lexico_to_scalar(rank_based_fitness_sp_2)),
                             np.mean(lexico_to_scalar(steady_state_fitness)))

        # Test if the rank based selection with sp=1.0 performs better than sp=2.0
        ga.sp = 1.0
        _, rank_based_selected_idx_sp_1 = ga.rank_based_selection(lexicographic_fitness, 5, ga_instance)
        rank_based_fitness_sp_1 = lexicographic_fitness[rank_based_selected_idx_sp_1]
        self.assertGreaterEqual(np.mean(lexico_to_scalar(rank_based_fitness_sp_1)),
                                np.mean(lexico_to_scalar(rank_based_fitness_sp_2)))

        # Test if the steady state selection returns the best individuals
        expected_top_fitness = sorted(lexicographic_fitness, reverse=True)[:5]
        actual_selected_fitness = [lexicographic_fitness[idx] for idx in steady_state_selected_idx]
        self.assertTrue(all(selected in expected_top_fitness for selected in actual_selected_fitness))


if __name__ == '__main__':
    unittest.main()
