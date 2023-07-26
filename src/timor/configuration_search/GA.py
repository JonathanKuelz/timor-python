from copy import deepcopy
from datetime import datetime
import itertools
import json
from pathlib import Path
import pickle
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from networkx import MultiDiGraph
import networkx as nx
import numpy as np
import pygad

from timor import ModuleAssembly, ModulesDB
from timor.utilities import logging, module_classification
from timor.utilities.configurations import TIMOR_CONFIG
from timor.utilities.dtypes import LimitedSizeMap, randomly
import timor.utilities.errors as err
from timor.utilities.visualization import plot_time_series


class GA:
    """
    This class provides an interface to optimize assemblies with genetic algorithms using a custom fitness function.

    The idea of a genetic algorithm is to mimic the process of natural selection and evolution to solve optimization
    problems. It involves creating a population of potential solutions (chromosomes), applying genetic operators
    like crossover and mutation to generate offsprings, evaluating their fitness, and selecting the fittest individuals
    for the next generation. Here, our target is to select candidate modules from a module base and assemble
    them in a sequence aiming to minimize a user-specified fitness function.The iterative process ends after a
    pre-defined number of iterations.

    Currently, this class is limited to modules that have exactly two connectors, so it can only be used for the
    assembly of serial kinematic chains.
    """

    __assembly_cache_size: int = 10e5  # Number of assemblies to keep in the cache.
    __empty_slot = 'EMPTY'  # Placeholder ID for a gene representing an empty slot ("no module here").
    # Define toolbox-level default hyperparameters which can be overwritten by user preferences.
    hp_default: Dict[str, any] = {'population_size': 20,
                                  'num_generations': 100,
                                  'num_genes': 12,
                                  'mutation_probability': 0.2,
                                  'initial_prob_joint': .5,
                                  'num_parents_mating': 5,
                                  'keep_parents': 4,
                                  'keep_elitism': 5,
                                  }

    def __init__(self, db: Union[str, ModulesDB]):
        """
        Initialize the GA optimizer by performing necessary pre-computations once a DB is provided.

        The initialization itself does not perform any optimization and is computationally cheap. To start the actual
        optimization, call the optimize() method.

        :param db: Either the name of a ModulesDB or a ModulesDB instance that will be used for the GA. The operators
            for the genetic optimization expect each module to have exactly two connectors -- if the DB contains any
            modules with more or less connectors, this method will raise an error.
        """
        if isinstance(db, str):
            self.db: ModulesDB = ModulesDB.from_name(db)
        elif isinstance(db, ModulesDB):
            self.db: ModulesDB = db

        if not all(len(m.available_connectors) == 2 for m in self.db):
            raise ValueError('The GA optimizer only works for modules with exactly two connectors!')

        # Divide the database into different types of modules.
        bases, links, joints, eefs = map(
            lambda sub_db: tuple(sorted(sub_db, key=lambda m: m.id)),
            module_classification.divide_db_in_types(self.db)
        )
        self.base_ids: Tuple[str, ...] = tuple(sorted(m.id for m in bases))
        self.link_ids: Tuple[str, ...] = tuple(sorted(m.id for m in links))
        self.joint_ids: Tuple[str, ...] = tuple(sorted(m.id for m in joints))
        self.eef_ids: Tuple[str, ...] = tuple(sorted(m.id for m in eefs))

        # Transform num and id to each other. Notice that 'id' is the module property in the database while 'num' is the
        # index of the module in the database starting from 1.
        self.all_modules: Tuple[str, ...] = tuple(itertools.chain(
            (self.__empty_slot,), self.base_ids, self.joint_ids, self.link_ids, self.eef_ids))
        self.id2num: Dict[str, int] = {ID: i for i, ID in enumerate(self.all_modules)}

        # Store the graph in memory for faster access.
        self.G: MultiDiGraph = self.db.connectivity_graph
        # Cache the assembly results for faster access.
        self.assembly_cache: LimitedSizeMap[Tuple[int], ModuleAssembly] = \
            LimitedSizeMap(maxsize=self.__assembly_cache_size)

        # Cache the modules with the same connectors for faster access.
        self.same_connector_cache: Dict[str, List[str]] = {}
        for module in self.all_modules:
            if module == self.__empty_slot:
                self.same_connector_cache[self.__empty_slot] = list(self.all_modules[1:])
                continue
            self.same_connector_cache[module] = [m.id for m in
                                                 self.db.find_modules_with_same_connectors(self.db.by_id[module])]

        # Hyperparameter priority: user preferences > config file > toolbox default values.
        self.hp: Dict[str, any] = deepcopy(self.hp_default)
        if 'OPTIMIZERS' in TIMOR_CONFIG.sections():
            for option in TIMOR_CONFIG.options('OPTIMIZERS'):
                self.hp[option] = TIMOR_CONFIG.get('OPTIMIZERS', option)
        if 'OPTIMIZERS.GA' in TIMOR_CONFIG.sections():
            for option in TIMOR_CONFIG.options('OPTIMIZERS.GA'):
                self.hp[option] = TIMOR_CONFIG.get('OPTIMIZERS.GA', option)
        logging.debug(f"Class default Hyperparameters: {json.dumps(self.hp)}")

        self._last_ga_instance: Optional[pygad.GA] = None

    def check_individual(self, individual: np.ndarray) -> bool:
        """
        Checks whether an integer-encoded individual (i.e. a module sequence) can be assembled.

        :param individual: The individual/offspring to be checked. Notice it consists of integers, not IDs.
        :return: True of the offspring is valid, False otherwise.
        """
        individual_tuple = tuple(individual)
        if individual_tuple in self.assembly_cache:
            return True

        module_ids = self._map_genes_to_id(individual_tuple)
        try:
            assembly = ModuleAssembly.from_serial_modules(self.db, module_ids)
            self.assembly_cache[individual_tuple] = assembly
            return True
        except (err.InvalidAssemblyError, ValueError):
            return False

    def mutation(self, population: np.ndarray, ga_instance: pygad.GA) -> np.ndarray:
        """
        Performs mutation on each chromosome of the offspring by randomly selecting a gene and replacing it.

        The replacement is based on finding modules with the same connectors as the initial module, and the
        replacement candidate is chosen randomly. Notice offspring contains integers not IDs.

        :param population: The offspring to be mutated.
        :param ga_instance: The instance of pygad.GA class.
        """
        mutate = np.where(np.random.random(population.shape) < ga_instance.mutation_probability)
        offspring = population.copy()
        for (i, j) in zip(*mutate):  # i is the individual, j is the gene
            gene_initial_id = self.all_modules[offspring[i, j]]
            replacement_candidates = self.same_connector_cache[gene_initial_id].copy()
            # Check if gene is at the start or end of chromosome. We will never replace base and eef with empty module.
            if j != 0 and j != offspring.shape[1] - 1:
                replacement_candidates.append(self.__empty_slot)
            valid_gene_found = False
            while not valid_gene_found:
                gene_new_id = np.random.choice(replacement_candidates)
                offspring[i, j] = self.id2num[gene_new_id]
                if self.check_individual(offspring[i]):
                    valid_gene_found = True
        return offspring

    def optimize(self,
                 fitness_function: Callable[[ModuleAssembly, pygad.GA, int], float],
                 hp: Optional[dict] = None,
                 sane_keyboard_interrupt: bool = True,
                 timeout: Optional[float] = None,
                 **ga_kwargs
                 ) -> pygad.GA:
        """
        Perform optimization utilizing a GA by trying to optimize the fitness function over module assemblies.

        Users can specify the hyperparameters for the GA. If not specified, the default hyperparameters will be used.

        :ref: https://pygad.readthedocs.io/en/latest/pygad.html?highlight=steady%20state%20parent#supported-parent-selection-operations  # noqa: E501

        :param fitness_function: the fitness function for the GA. It should take three arguments: a ModuleAssembly
            which is to be evaluated as well as the GA instance and the index of the solution in the population. It is
            up to the user which of these arguments are used in the fitness function.
        :param hp: Optional hyperparameters overriding the user and default hyperparameters.
        :param sane_keyboard_interrupt: If True, the GA will be stopped when a keyboard interrupt is received, but the
            exception will not be raised.
        :param timeout: If specified, the GA will be stopped after the specified number of seconds.
        :param ga_kwargs: Additional keyword arguments to be passed to the pygad.GA class that do not qualify as
            hyperparameters (so anything that should not be logged as a hyperparameter, e.g., callbacks).
        """
        if hp is None:
            hp = {}
        run_hp = deepcopy(self.hp)
        run_hp.update(hp)

        ga_kwargs.setdefault('save_best_solutions', True)
        ga_kwargs.setdefault('save_solutions', False)

        if set(run_hp.keys()).intersection(ga_kwargs.keys()):
            raise ValueError('Hyperparameters and additional keyword arguments must not overlap!')

        if run_hp['num_genes'] < 3:
            raise ValueError('The number of genes must be at least 3!')

        logging.info(f"Hyperparameters used: {json.dumps(run_hp)}")

        gene_space: List[List[int]] = [[self.id2num[_id] for _id in self.base_ids]]
        for _ in range(run_hp['num_genes'] - 2):
            gene_space.append([self.id2num[_id] for _id in self.joint_ids + self.link_ids])
        gene_space.append([self.id2num[_id] for _id in self.eef_ids])

        def fitness(_ga: pygad.GA, individual: List[int], individual_idx: int) -> float:
            """Fitness function for the genetic algorithm."""
            assembly = self._pre_computation_fitness(individual)
            return fitness_function(assembly, _ga, individual_idx)

        initial_population = self._get_initial_population((run_hp.pop('population_size'), run_hp['num_genes']),
                                                          run_hp.pop('initial_prob_joint'))

        if 'logger' not in ga_kwargs:
            ga_kwargs['logger'] = logging.getLogger()

        save_fp = run_hp.pop('save_solutions_dir', None)
        if not ga_kwargs['save_best_solutions']:
            raise ValueError("If you provide a filepath to store solutions at, save_best_solutions must be true.")
        if save_fp is not None:
            save_fp = Path(save_fp).resolve()
            save_fp.mkdir(exist_ok=True)
            save_fp = save_fp.joinpath('GA_started_at_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + 'best.pkl')

        if timeout is not None:
            def no_callback(_: pygad.GA):
                return None
            current_callback: Callable[[pygad.GA], Any] = ga_kwargs.get('on_generation', no_callback)
            t_now = time.time()

            def timeout_callback(ga: pygad.GA):
                val = current_callback(ga)
                if time.time() - t_now > timeout:
                    logging.info("Terminating GA optimization due to timeout.")
                    return "stop"
                return val

            ga_kwargs['on_generation'] = timeout_callback

        ga_instance = pygad.GA(
            fitness_func=fitness,
            gene_space=gene_space,
            gene_type=int,
            initial_population=initial_population,
            crossover_type=self.single_valid_crossover,
            mutation_type=self.mutation,
            **run_hp,
            **ga_kwargs
        )
        try:
            ga_instance.run()
        except KeyboardInterrupt:
            if sane_keyboard_interrupt:
                logging.warning("Keyboard interrupt received, stopping optimization.")
            else:
                raise

        if save_fp is not None:
            with save_fp.open('wb') as f:
                pickle.dump(ga_instance.best_solutions, f)
        if ga_kwargs['save_best_solutions']:
            best_solution_modules = ga_instance.best_solutions[-1]
            best_solution = ga_instance.best_solutions_fitness[-1]
            logging.info(f"Best solution modules: {best_solution_modules}")
            logging.info(f"Best solution fitness: {best_solution}")
        self._last_ga_instance = ga_instance
        return ga_instance

    def single_valid_crossover(self, parents: np.ndarray, offspring_size: Tuple, ga_instance: pygad.GA) -> np.ndarray:
        """
        Performs single-point crossover between parents to create new offspring.

        Generally speaking, the genetic section after the split point in 'parent1' is replaced with the genetic section
        after the same split point in 'parent2'. Notice parents consists of integers not IDs.

        :params parents: Indicates the whole parental generation of genetic sections.
        :params offspring_size: Indicates the size of the offspring, which is a tuple of (num_offsprings, num_genes).
        :params ga_instance: The instance of pygad.GA class.
        """
        offsprings = []
        split_positions = np.array(range(offspring_size[1]))

        for p1, p2, s in randomly(itertools.product(parents, parents, split_positions)):
            child = np.concatenate((p1[:s], p2[s:]))
            if self.check_individual(child):
                offsprings.append(child)
            if len(offsprings) >= offspring_size[0]:
                break

        if len(offsprings) == 0:
            raise ValueError("No valid offspring found after checking all possible crossovers.")
        if len(offsprings) != offspring_size[0]:
            raise ValueError("All crossovers searched, but the desired number of offspring was not found.")

        offsprings = np.array(offsprings)
        return offsprings

    def plot_fitness(self, show_figure: bool = True):
        """
        Visualize the fitness values over generations.

        :param show_figure: If True, the figure will be shown.
        """
        if self._last_ga_instance is None:
            logging.warning('No GA optimization has been run yet.')
            return
        elif not hasattr(self._last_ga_instance, 'best_solutions_fitness'):
            logging.warning('No fitness values were recorded during the last GA optimization.'
                            'Set save_best_solutions=True in the GA kwargs to visualize results.')
            return
        num_data_points = len(self._last_ga_instance.best_solutions_fitness)
        times = [i for i in range(num_data_points)]
        data = [(self._last_ga_instance.best_solutions_fitness, "Fitness")]
        f = plot_time_series(times, data, show_figure=show_figure)
        return f

    def _map_genes_to_id(self, genome: Sequence[int]) -> Tuple[str]:
        """
        Maps a list of gene encodings to the corresponding tuple of module IDs.

        :param genome: A genome, encoding genes as integers.
        """
        return tuple(gene_id for gene_id in map(self.all_modules.__getitem__, genome) if gene_id != self.__empty_slot)

    def _get_initial_population(self,
                                population_size: Tuple[int, int],
                                prob_joint: float = None
                                ) -> np.ndarray:
        """
        Generates the initial population for the genetic algorithm.

        It returns a list of chromosomes, where each chromosome represents a potential solution. It extends the shortest
        paths to a desired length by successively adding detours of length 1. Each chromosome is a list of genes, where
        each gene represents a module. The first gene is the base, the last gene is the end effector, and the genes in
        between are links and joints. The length of the chromosome equals to the gene_len.

        :param population_size: The size of the initial population as a tuple of (num_offsprings, num_genes).
        :param prob_joint: The probability of a joint being selected as a gene.
        """
        if prob_joint is None:
            prob_joint = self.hp_default['initial_prob_joint']
        initial_population = []
        for _ in range(population_size[0]):
            base = self.db.by_id[random.choice(self.base_ids)]
            eef = self.db.by_id[random.choice(self.eef_ids)]
            path = nx.shortest_path(self.G, base, eef)
            while len(path) < population_size[1]:
                # return edge connecting the two modules in the randomly selected location
                loc = random.randint(0, len(path) - 2)
                if path[loc] is path[loc + 1]:
                    num_edges = self.G.number_of_edges(path[loc], path[loc + 1])
                    rand_edge = random.randint(0, num_edges - 1)
                    shortest_path = [[(path[loc], path[loc + 1], rand_edge)]]
                else:
                    shortest_path = list(nx.all_simple_edge_paths(self.G, path[loc], path[loc + 1], cutoff=1))
                connector_parent = self.G.edges[shortest_path[0][0]]['connectors'][0]
                connector_child = self.G.edges[shortest_path[0][0]]['connectors'][1]
                neighbors_parent = set(self.G.neighbors(path[0]))
                neighbors_child = set(self.G.neighbors(path[1]))
                shared = [m for m in neighbors_parent.intersection(neighbors_child)
                          if not any(con.type in {'base', 'eef'} for con in m.available_connectors.values())]
                random.shuffle(shared)
                try_choose_joint = random.random() < prob_joint
                if try_choose_joint:
                    _shared = {m for m in shared if m.num_joints > 0}
                    shared = _shared if len(_shared) > 0 else shared
                else:
                    _shared = {m for m in shared if m.num_joints == 0}
                    shared = _shared if len(_shared) > 0 else shared
                for m in shared:
                    c_x1 = list(m.available_connectors.values())[0]
                    c_x2 = list(m.available_connectors.values())[1]
                    # check if the two connectors are compatible
                    if (connector_parent.connects(c_x2) and connector_child.connects(c_x1)) \
                            or (connector_parent.connects(c_x1) and connector_child.connects(c_x2)):
                        path.insert(1, m)
                        break
            for i in range(len(path)):
                path[i] = self.id2num[path[i].id]
            initial_population.append(path)

        return np.array(initial_population)

    def _pre_computation_fitness(self, solution: Sequence[int]) -> ModuleAssembly:
        """
        Perform pre-computation for the fitness function.

        It converts the solution from integer encodings to IDs and assembles the modules in a chain.

        :param solution: The solution to be evaluated, encoded in integer values.
        """
        solution_tuple = tuple(solution)
        if solution_tuple in self.assembly_cache:
            assembly = self.assembly_cache[solution_tuple]
        else:
            module_ids = self._map_genes_to_id(solution_tuple)
            assembly = ModuleAssembly.from_serial_modules(self.db, module_ids)
            self.assembly_cache[solution_tuple] = assembly
        return assembly
