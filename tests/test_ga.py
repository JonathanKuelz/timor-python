import unittest

import numpy as np
import pygad

from timor import ModuleAssembly, ModulesDB
from timor.configuration_search.AssemblyFilter import default_filters
from timor.task import Task
from timor.configuration_search.GA import GA
from timor.utilities import file_locations


class TestGAProperties(unittest.TestCase):
    """Test properties of GA"""

    def setUp(self) -> None:
        np.random.seed(2022)
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

    def _fitness_func(self, assembly: ModuleAssembly, ga_instance: pygad.GA, solution_idx: int) -> float:
        """Fitness function for the genetic algorithm which is fast du evaluate."""
        self.fitness_func_calls += 1
        return assembly.mass

    def test_ga_setup(self):
        """Test that setting up a GA instance works as expected."""
        ga = GA(self.db)
        ga_ins = ga.optimize(self._fitness_func, hp=self.hp)
        fitness_calls = self.fitness_func_calls
        self.assertGreater(self.fitness_func_calls, 0)
        ga2 = GA(self.db.name)
        ga_ins2 = ga2.optimize(self._fitness_func, hp=self.hp)
        self.assertGreater(self.fitness_func_calls, fitness_calls)
        self.assertFalse(np.array_equal(ga_ins.initial_population, ga_ins2.initial_population))
        fig = ga.plot_fitness(show_figure=False)

    def test_callbacks(self):
        """Test the mutation and crossover callbacks provided by pyGAD work with custom operations."""
        ping_counts = {'mutation': 0, 'crossover': 0}

        def make_ping_callback(d: dict, key: str):
            """Creates a callable that increments d[key] and accepts any number of arguments."""
            def ping(a1, a2):
                d[key] += 1
            return ping

        ga = GA(self.db)
        callbacks = {'on_mutation': make_ping_callback(ping_counts, 'mutation'),
                     'on_crossover': make_ping_callback(ping_counts, 'crossover')}
        ga.optimize(self._fitness_func, hp=self.hp, **callbacks)
        self.assertEqual(ping_counts['crossover'], self.hp['num_generations'])
        self.assertEqual(ping_counts['mutation'], self.hp['num_generations'])

    def test_crossover(self):
        """Test if the functionality of single crossover is valid."""
        ga = GA(self.db)
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
        ga = GA(self.db)
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

    def test_offspring_is_valid(self):
        """Test if the whole offspring and the best solution of each generation step is valid."""
        ga = GA(self.db)
        ga_ins = ga.optimize(self._fitness_func, hp=self.hp)
        for i in ga_ins.population:
            self.assertTrue(ga.check_individual(i))
        for i in ga_ins.best_solutions:
            self.assertTrue(ga.check_individual(i))

    def test_population_initialization(self):
        """Test if the initial population is valid."""
        ga = GA(self.db)
        for genome in ga._get_initial_population(population_size=(10, 10)):
            self.assertTrue(ga.check_individual(genome))
        joint_nums = {ga.id2num[jid] for jid in ga.joint_ids}
        for genome in ga._get_initial_population(population_size=(5, 5), prob_joint=0):
            self.assertTrue(ga.check_individual(genome))
            self.assertTrue(len(joint_nums.intersection(genome)) == 0)
        for genome in ga._get_initial_population(population_size=(5, 5), prob_joint=1):
            self.assertTrue(ga.check_individual(genome))
            # prob_joint=1 still means there only is a joint if there CAN be a joint, but sometimes a connection
            # between chosen base and end effector can only be achieved with using links
            self.assertTrue(len(joint_nums.intersection(genome)) > 0)


if __name__ == '__main__':
    unittest.main()