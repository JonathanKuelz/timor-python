from copy import deepcopy
from datetime import datetime
from enum import Enum
import itertools
import json
from pathlib import Path
import pickle
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from networkx import MultiDiGraph
import networkx as nx
import numpy as np
import pygad
from tqdm import tqdm

from timor import ModuleAssembly, ModulesDB, Task
from timor.task import Goals
from timor.utilities import logging, module_classification
from timor.utilities.configurations import TIMOR_CONFIG
from timor.utilities.dtypes import Lexicographic, LimitedSizeMap, randomly
import timor.utilities.errors as err
from timor.utilities.file_locations import map2path
from timor.utilities.visualization import MeshcatVisualizer, clear_visualizer, color_visualization, \
    place_billboard, save_visualizer_scene, save_visualizer_screenshot
import zmq.error


class ProgressUnit(Enum):
    """Classifies the different methods to count the progress in a GA optimization."""

    GENERATIONS = 0
    INDIVIDUALS = 1
    SECONDS = 2


def get_ga_progress(unit: ProgressUnit,
                    ga_inst: Optional[pygad.GA] = None,
                    t0: Optional[float] = 0,
                    generation_bias: int = 0) -> int:
    """
    Returns the progress of a genetic algorithm in [ProgressUnit] units.

    :param unit: The unit to count the progress in.
    :param ga_inst: The pygad.GA instance to get the progress from. Necessary for GENERATIONS and INDIVIDUALS.
    :param t0: The start time of the optimization. Necessary for SECONDS.
    :param generation_bias: The "generation" to start counting from
    """
    if unit is ProgressUnit.SECONDS and t0 is None:
        raise ValueError('t0 must be specified for ProgressUnit.SECONDS')
    elif unit in (ProgressUnit.GENERATIONS, ProgressUnit.INDIVIDUALS) and ga_inst is None:
        raise ValueError('If the progress unit is GENERATIONS or INDIVIDUALS, ga_inst must be specified')

    if unit is ProgressUnit.GENERATIONS:
        return ga_inst.generations_completed + generation_bias
    elif unit is ProgressUnit.INDIVIDUALS:
        return ga_inst.initial_population.shape[0] * (ga_inst.generations_completed + generation_bias)
    elif unit is ProgressUnit.SECONDS:
        return int(time.time() - t0)


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

    __assembly_cache_size: int = 1000  # Number of assemblies to keep in the cache.
    __empty_slot = 'EMPTY'  # Placeholder ID for a gene representing an empty slot ("no module here").
    __fitness_cache_size: int = 1000000  # Number of fitness values to keep in the cache.
    # Define toolbox-level default hyperparameters which can be overwritten by user preferences.
    hp_default: Dict[str, any] = {'population_size': 20,
                                  'num_generations': 100,
                                  'num_genes': 12,
                                  'mutation_probability': 0.2,
                                  'num_parents_mating': 5,
                                  'keep_parents': 4,
                                  'keep_elitism': 5,
                                  'selection_pressure': 2.0,  # It's used by rank_based_selection which is within [1, 2]
                                  }

    def __init__(self,
                 db: Union[str, ModulesDB],
                 custom_hp: Optional[Dict[str, any]] = None,
                 mutation_weights: Optional[Dict[str, float]] = None,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize the GA optimizer by performing necessary pre-computations once a DB is provided.

        The initialization itself does not perform any optimization and is computationally cheap. To start the actual
        optimization, call the optimize() method.

        :param db: Either the name of a ModulesDB or a ModulesDB instance that will be used for the GA. The operators
            for the genetic optimization expect each module to have exactly two connectors -- if the DB contains any
            modules with more or less connectors, this method will raise an error.

        :param custom_hp: A dictionary of custom hyperparameters that will be used instead of the default ones.
            For more hyperparameters see reference: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class.
        :param mutation_weights: Maps a module ID or the `EMPTY` ID to a weight that determines how likely it is to be
            chosen as a replacement during mutation. If None, all modules have the same weight.
        :param rng: A numpy random number generator. If None, a new one will be created.
        """
        if isinstance(db, str):
            self.db: ModulesDB = ModulesDB.from_name(db)
        elif isinstance(db, ModulesDB):
            self.db: ModulesDB = db

        if rng is None:
            rng = np.random.default_rng()
        self.rng: np.random.Generator = rng

        if not all(len(m.available_connectors) == 2 for m in self.db):
            raise ValueError('The GA optimizer only works for modules with exactly two connectors!')

        # Divide the database into different types of modules.
        bases, links, joints, eefs = map(
            lambda sub_db: tuple(sorted(sub_db, key=lambda m: m.id)),
            module_classification.divide_db_in_types(self.db, strict=False)
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
        self.assembly_cache: LimitedSizeMap[Tuple[str], ModuleAssembly] = \
            LimitedSizeMap(maxsize=self.__assembly_cache_size)
        self.fitness_cache: LimitedSizeMap[Tuple[str], float] = LimitedSizeMap(maxsize=self.__fitness_cache_size)

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
        if custom_hp is not None:
            self.hp.update(custom_hp)
        logging.debug(f"Class default Hyperparameters: {json.dumps(self.hp)}")

        if mutation_weights is None:
            mutation_weights = dict.fromkeys(self.all_modules, 1.)
        if not set(self.all_modules).issubset(set(mutation_weights.keys())):
            missing = set(self.all_modules).difference(set(mutation_weights.keys()))
            raise ValueError(f"Modules {missing} are not covered by the mutation weights!")
        self.mutation_weights: Dict[str, float] = mutation_weights

        self._last_ga_instance: Optional[pygad.GA] = None

        # Select the parent selection function based on the selection type.
        self.sp = self.hp['selection_pressure']

    def check_individual(self, individual: np.ndarray) -> bool:
        """
        Checks whether an integer-encoded individual (i.e. a module sequence) can be assembled.

        :param individual: The individual/offspring to be checked. Notice it consists of integers, not IDs.
        :return: True of the offspring is valid, False otherwise.
        """
        try:
            individual = tuple(individual)
            module_ids = self._map_genes_to_id(individual)
            self._get_candidate_assembly(module_ids)
            return True
        except (err.InvalidAssemblyError, ValueError):
            return False

    def make_visualization_callback(self,
                                    viz: Optional[MeshcatVisualizer] = None,
                                    task: Optional[Task.Task] = None,
                                    save_at: Optional[Union[str, Path]] = None,
                                    file_format: str = 'html',
                                    every_n_generations: int = 10,
                                    checkmark_successful: bool = False) -> Callable[[pygad.GA], bool]:
        """
        Creates a callback function that can be passed to the GA optimizer to visualize solution candidates.

        The callback function will be called every n generations and will visualize the best solution candidate of that
        generation. If multiple candidates achieve the best fitness, one will be selected at random.

        :param viz: A meshcat visualizer instance. If None, a new visualizer will be created.
        :param task: If given, a candidate assembly will be visualized in the context of the given task.
        :param save_at: Per default, the visualization is only shown in the browser. If a path to a directory is given,
            all visualizations will be saved in there as static HTML or PNG.
        :param file_format: If save_at is given, this parameter determines the format of the saved visualizations. Can
            be either 'html' or 'png'. While html works "headless" without opening a browser, png enforces opening a
            viewer and cannot be used on a remote server. However, html files are much larger than png files.
        :param every_n_generations: Every nth generation, a visualization will be created.
        :param checkmark_successful: If True, a checkmark will be added to every goal that was achieved in the task.
        """
        file_format = file_format.lower()
        if file_format not in ('html', 'png'):
            raise ValueError(f"Unknown file format '{file_format}'! Must be either 'html' or 'png'.")

        if viz is None:
            viz = MeshcatVisualizer()
            viz.initViewer()

        if save_at is not None:
            save_at = map2path(save_at).expanduser().resolve()
            save_at.mkdir(exist_ok=True)
            if file_format == 'png':
                logging.info("Enforcing open visualizer for PNG output.")
                viz.viewer.open()

        def viz_callback(ga_instance: pygad.GA):
            generation = ga_instance.generations_completed
            if generation % every_n_generations != 0 and generation != ga_instance.num_generations:
                return True
            candidate = ga_instance.best_solution(ga_instance.last_generation_fitness)[0]
            assembly = self._get_candidate_assembly(self._map_genes_to_id(candidate))

            try:
                clear_visualizer(viz)
            except zmq.error.ZMQError:
                viz.viewer.window.connect_zmq()
            if task is None:
                assembly.robot.visualize(viz)
            else:
                task.visualize(viz, robots=[assembly.robot], center_view=False)

            if any(g.goal_pose.valid(assembly.robot.fk()) for g in task.goals if isinstance(g, Goals.At)):
                color_visualization(viz, assembly)

            if checkmark_successful:
                for goal in task.goals:
                    if not isinstance(goal, (Goals.At, Goals.Follow)):
                        continue
                    poses = [goal.goal_pose] if isinstance(goal, Goals.At) else goal.trajectory.pose
                    suc = True
                    for pose in poses:
                        _, suc_goal = assembly.robot.ik(pose, task=task, max_iter=1000)
                        suc = suc and suc_goal
                    if suc:
                        place_billboard(viz, text='🗸', name=goal.id, placement=poses[0].nominal,
                                        text_color='green', background_color='transparent', scale=.3)

            if save_at is not None:
                task_id = task.id if task is not None else ''
                if file_format.lower() == 'html':
                    save_visualizer_scene(viz, filename=save_at / f'{task_id}_gen_{generation}.html', overwrite=True)
                else:
                    save_visualizer_screenshot(viz, filename=save_at / f'_{task_id}_gen_{generation}.png',
                                               overwrite=True)
            return True

        return viz_callback

    def mutation(self, population: np.ndarray, ga_instance: pygad.GA) -> np.ndarray:
        """
        Performs mutation on each chromosome of the offspring by randomly selecting genes and replacing them.

        The replacement is based on finding modules with the same connectors as the initial module, and the
        replacement candidate is chosen randomly. Notice offspring contains integers not IDs.

        :param population: The offspring to be mutated.
        :param ga_instance: The instance of pygad.GA class.
        """
        mutate = np.where(self.rng.random(population.shape) < ga_instance.mutation_probability)
        offspring = population.copy()
        for (individual, gene) in zip(*mutate):
            gene_initial_id = self.all_modules[offspring[individual, gene]]
            replacement_candidates = self.same_connector_cache[gene_initial_id].copy()
            # Check if gene is at the start or end of chromosome. We will never replace base and eef with empty module.
            if gene not in {0, offspring.shape[1] - 1}:
                replacement_candidates.append(self.__empty_slot)
            weights = np.array([self.mutation_weights[c] for c in replacement_candidates])
            p_mutation = weights / weights.sum()
            while True:
                gene_new_id = self.rng.choice(replacement_candidates, p=p_mutation)
                offspring[individual, gene] = self.id2num[gene_new_id]
                if self.check_individual(offspring[individual]):
                    break
        return offspring

    def optimize(self,
                 fitness_function: Callable[[ModuleAssembly, pygad.GA, int], float],
                 hp: Optional[dict] = None,
                 progress_bar: bool = True,
                 debug_logging_frequency: Optional[int] = 10,
                 sane_keyboard_interrupt: bool = True,
                 timeout: Optional[float] = None,
                 wandb_run=None,
                 progress_unit: ProgressUnit = ProgressUnit.GENERATIONS,
                 steps_at_start: int = 0,
                 selection_type: str = 'sss',
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
        :param progress_bar: If True, a tqdm progress bar over the numbers of generations completed will be shown.
        :param debug_logging_frequency: If specified, internal debug metrics will be logged every full n progress units.
        :param sane_keyboard_interrupt: If True, the GA will be stopped when a keyboard interrupt is received, but the
            exception will not be raised.
        :param timeout: If specified, the GA will be stopped after the specified number of seconds.
        :param wandb_run: You can optionally pass a Weights and Biases run instance to log the results. A run instance
            can be obtained by calling wandb.init() before calling this function.
        :param progress_unit: The metric to be used for tracking performance metrics, e.g. for logging or wandb.
        :param steps_at_start: The value of the progress unit at the first generation. This is useful if you want
            to continue a run that was interrupted or if multiple runs should be tracked "as one".
        :param selection_type: The type of parent selection to be used. The options defined in pyGAD include 'sss',
            'rws', 'sus', 'rank', 'tournament', 'random'. Notice that use a customized rank-based selection function can
            be beneficial for diversity in the population and avoiding premature convergence to suboptimal solutions.
        :param ga_kwargs: Additional keyword arguments to be passed to the pygad.GA class that do not qualify as
            hyperparameters (so anything that should not be logged as a hyperparameter, e.g., callbacks).
        """
        self.fitness_cache.reset()
        if hp is None:
            hp = {}
        run_hp = deepcopy(self.hp)
        run_hp.update(hp)
        self.sp = run_hp.pop('selection_pressure')

        ga_kwargs.setdefault('save_best_solutions', True)
        ga_kwargs.setdefault('save_solutions', False)

        if set(run_hp.keys()).intersection(ga_kwargs.keys()):
            raise ValueError('Hyperparameters and additional keyword arguments must not overlap! ({})'.format(
                set(run_hp.keys()).intersection(ga_kwargs.keys())
            ))

        if run_hp['num_genes'] < 3:
            raise ValueError('The number of genes must be at least 3!')

        logging.info(f"Hyperparameters used: {json.dumps(run_hp)}")
        logging.info("Progress unit: {}".format(progress_unit))

        gene_space: List[List[int]] = [[self.id2num[_id] for _id in self.base_ids]]
        for _ in range(run_hp['num_genes'] - 2):
            gene_space.append([self.id2num[_id] for _id in self.joint_ids + self.link_ids])
        gene_space.append([self.id2num[_id] for _id in self.eef_ids])

        def ensure_callback_return_type(fitness_func):
            """
            A decorator function that checks and validates the return type of the provided fitness function.

            This wrapper is designed to ensure that the return type of the fitness function is compatible with the
            GA framework. It works by intercepting calls to the fitness function. On the first call, it executes the
            fitness function and checks the type of its return value. If the return type is Lexicographic and this type
            is not already in the list of supported types by pygad.GA, it adds Lexicographic to this list. The process
            is done only once, as indicated by the has_checked flag. On subsequent calls, the wrapper directly returns
            the fitness function's result without performing the type check again.

            :param fitness_func: The fitness function to be wrapped.
            :return: The wrapper function which adds compatibility for Lexicographic type if necessary.
            """
            has_checked = False

            def wrapper(_ga, individual, individual_idx):
                nonlocal has_checked
                result = fitness_func(_ga, individual, individual_idx)
                if not has_checked:
                    if isinstance(result, Lexicographic):
                        if Lexicographic not in pygad.GA.supported_int_float_types:
                            pygad.GA.supported_int_float_types.append(Lexicographic)
                    has_checked = True
                return result
            return wrapper

        @ensure_callback_return_type
        def fitness(_ga: pygad.GA, individual: List[int], individual_idx: int) -> float:
            """Fitness function for the genetic algorithm."""
            individual = tuple(individual)
            module_ids = self._map_genes_to_id(individual)
            if module_ids in self.fitness_cache:
                return self.fitness_cache[module_ids]
            assembly = self._get_candidate_assembly(module_ids)
            fitness_value = fitness_function(assembly, _ga, individual_idx)
            self.fitness_cache[module_ids] = fitness_value
            return fitness_value

        initial_population = self._get_initial_population(
            population_size=(int(run_hp.pop('population_size')), int(run_hp['num_genes'])),
            module_weights=run_hp.pop('initial_module_weights', None))

        if 'logger' not in ga_kwargs:
            ga_kwargs['logger'] = logging.getLogger()

        save_fp = run_hp.pop('save_solutions_dir', None)
        if not ga_kwargs['save_best_solutions'] and save_fp is not None:
            raise ValueError("If you provide a filepath to store solutions at, save_best_solutions must be true.")
        if save_fp is not None:
            save_fp = Path(save_fp).expanduser().resolve()
            save_fp.mkdir(exist_ok=True)
            save_fp = save_fp.joinpath('GA_started_at_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_best.pkl')

        on_generation_cbs: List[Callable[[pygad.GA], any]] = []
        if progress_bar:
            progress_bar = tqdm(total=hp['num_generations'] + 1 + steps_at_start, desc='Generations')
            progress_bar.update(steps_at_start)

            def progress_callback(_: pygad.GA) -> bool:
                progress_bar.update(1)
                return True

            on_generation_cbs.append(progress_callback)

        t0 = time.time()
        if timeout is not None:
            def timeout_callback(ga_inst: pygad.GA) -> bool:
                if time.time() - t0 > timeout:
                    logging.info("Terminating GA optimization due to timeout after {} generations.".format(
                        ga_inst.generations_completed))
                    return False
                return True

            on_generation_cbs.append(timeout_callback)

        if wandb_run is not None:
            t_last_generation = time.time()

            def log_wandb_callback(_ga: pygad.GA) -> bool:
                nonlocal t_last_generation
                idx_best = _ga.last_generation_fitness.argmax()
                modules_best = self._map_genes_to_id(_ga.population[idx_best])
                wandb_data = {
                    'mean fitness': _ga.last_generation_fitness.mean(),
                    'best fitness': _ga.last_generation_fitness.max(),
                    'num modules for best fitness': len(modules_best),
                    'num joints for best fitness': len([m for m in modules_best if m in self.joint_ids]),
                    'average number of modules': _ga.pop_size[1] - np.sum(_ga.population == 0) / _ga.pop_size[0],
                    'individuals per second': initial_population.shape[0] / (time.time() - t_last_generation),
                    'num assemblies cached': len(self.assembly_cache),
                    'num fitness values cached': len(self.fitness_cache)
                }
                t_last_generation = time.time()
                wandb_run.log(wandb_data, step=get_ga_progress(progress_unit, _ga, t0, steps_at_start))
                return True

            on_generation_cbs.append(log_wandb_callback)

        if debug_logging_frequency is not None:
            def debug_log_callback(_ga: pygad.GA) -> bool:
                if get_ga_progress(progress_unit, _ga, t0, steps_at_start) % debug_logging_frequency == 0:
                    logging.debug("Generation: {}".format(_ga.num_generations))
                    logging.debug(f"Progress: {get_ga_progress(progress_unit, _ga, t0, steps_at_start)}{progress_unit}")
                    logging.debug("Assemblies cached: {}".format(len(self.assembly_cache)))
                return True

            on_generation_cbs.append(debug_log_callback)

        on_generation = self._chain_callbacks(ga_kwargs.pop('on_generation', None), on_generation_cbs)

        if selection_type == 'rank':
            selection_type = self.rank_based_selection

        ga_instance = pygad.GA(
            fitness_func=fitness,
            gene_space=gene_space,
            gene_type=int,
            initial_population=initial_population,
            crossover_type=self.single_valid_crossover,
            mutation_type=self.mutation,
            parent_selection_type=selection_type,
            on_generation=on_generation,
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
        finally:
            if progress_bar:
                progress_bar.close()

        if save_fp is not None:
            with save_fp.open('wb') as f:
                best = ga_instance.best_solutions[ga_instance.best_solution_generation]
                data = {
                    'best_solutions': ga_instance.best_solutions,
                    'best_solutions_fitness': ga_instance.best_solutions_fitness,
                    'best_solution_generation': ga_instance.best_solution_generation,
                    'best_solution_IDs': self._map_genes_to_id(best),
                    'db': self.db.name,
                    'hyperparameters': run_hp,
                }
                pickle.dump(data, f)
        if ga_kwargs['save_best_solutions']:
            logging.info("The best solution was found at generation {} with a fitness value of {}.".format(
                ga_instance.best_solution_generation,
                ga_instance.best_solutions_fitness[ga_instance.best_solution_generation]
            ))
            best = ga_instance.best_solutions[ga_instance.best_solution_generation]
            logging.info("Best solution: {}".format('-'.join(self._map_genes_to_id(best))))
        logging.info("Total optimization time: {:.2f} seconds.".format(time.time() - t0))
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
        parents = parents.astype(np.int64)

        for p1, p2, s in randomly(itertools.product(parents, parents, split_positions), rng=self.rng):
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

    def rank_based_selection(self, fitness: np.ndarray, num_parents: int, ga_instance: pygad.GA) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects the parents using a rank-based selection process.

        :source: https://en.wikipedia.org/wiki/Selection_(genetic_algorithm). The original paper for linear ranking:
            - Baker, James E. (1985), Grefenstette, John J. (ed.), "Adaptive Selection Methods for Genetic Algorithms",
            Conf. Proc. of the 1st Int. Conf. on Genetic Algorithms and Their Applications (ICGA),
            Hillsdale, New. Jersey: L. Erlbaum Associates, pp. 101–111, ISBN 0-8058-0426-9
            - Baker, James E. (1987), Grefenstette, John J. (ed.), "Reducing Bias and Inefficiency in the Selection
            Algorithm", Conf. Proc. of the 2nd Int. Conf. on Genetic Algorithms and Their Applications (ICGA),
            Hillsdale, New. Jersey: L. Erlbaum Associates, pp. 14–21, ISBN 0-8058-0158-8
        :param fitness: A list of fitness values of the current population. Its shape is 1D array with length equal to
        the number of individuals, while each element is a 'Lexicographic' object encompassing multiple values.
        :param num_parents: The number of parents to select.
        :param ga_instance: An instance of the pygad.GA class.
        :return: 1. The selected parents as a numpy array. Its shape is (the number of selected parents, num_genes).
        Note that the number of selected parents is equal to num_parents. 2. The indices of the selected parents inside
        the population. It is a 1D list with length equal to the number of selected parents.
        """
        assert all(isinstance(f, Lexicographic) for f in fitness), "All fitness values must be Lexicographic instances."

        n = len(fitness)
        fitness_ranks = np.argsort(np.argsort(fitness)) + 1

        selection_prob = [(1 / n) * (self.sp - ((2 * self.sp - 2) * (rank - 1) / (n - 1))) for rank in fitness_ranks]
        selected_parents_indices = np.random.choice(range(n), size=num_parents, replace=False, p=selection_prob)

        parents = np.array([ga_instance.population[idx, :].copy() for idx in selected_parents_indices])
        return parents, selected_parents_indices

    @staticmethod
    def _chain_callbacks(original_callback: Optional[Callable[[pygad.GA], any]] = None,
                         optional_callbacks: Sequence[Callable[[pygad.GA], any]] = ()
                         ) -> Callable[[pygad.GA], Optional[str]]:
        """
        This is a helper function to create a single callback function from a sequence of callable objects.

        The order of the callbacks is preserved, with the original callback being called first. If any of the optional
        callbacks returns False, the callback chain is terminated and "stop" is returned which will stop the GA.

        :param original_callback: The original callback function - this function can return any value, but any value
            other than "stop" will be ignored by pygad.
        :param optional_callbacks: A sequence of optional callbacks. Each callback should return True to continue the
            callback chain, or either False or 'stop' to stop the callback chain.
        """
        if original_callback is None:
            def original_callback(_: pygad.GA): return True

        callbacks = tuple(itertools.chain([original_callback], optional_callbacks))

        def _callback(ga_instance: pygad.GA):
            for cb in callbacks:
                ret = cb(ga_instance)
                if ret is False or ret == 'stop':
                    return 'stop'

        return _callback

    def _get_candidate_assembly(self, module_ids: Tuple[str]) -> ModuleAssembly:
        """
        Perform pre-computation for the fitness function.

        It converts the solution from integer encodings to IDs and assembles the modules in a chain.

        :param module_ids: The solution to be evaluated, encoded as module ids.
        """
        if module_ids in self.assembly_cache:
            assembly = self.assembly_cache[module_ids]
        else:
            assembly = ModuleAssembly.from_serial_modules(self.db, module_ids)
            self.assembly_cache[module_ids] = assembly
        return assembly

    def _map_genes_to_id(self, genome: Sequence[int]) -> Tuple[str]:
        """
        Maps a list of gene encodings to the corresponding tuple of module IDs.

        :param genome: A genome, encoding genes as integers.
        """
        return tuple(gene_id for gene_id in map(self.all_modules.__getitem__, genome) if gene_id != self.__empty_slot)

    def _get_initial_population(self,
                                population_size: Tuple[int, int],
                                module_weights: Optional[Dict[str, float]] = None
                                ) -> np.ndarray:
        """
        Generates the initial population for the genetic algorithm.

        It returns a list of chromosomes, where each chromosome represents a potential solution. It extends the shortest
        paths to a desired length by successively adding detours of length 1. Each chromosome is a list of genes, where
        each gene represents a module. The first gene is the base, the last gene is the end effector, and the genes in
        between are links and joints. The length of the chromosome equals to the gene_len.

        :param population_size: The size of the initial population as a tuple of (num_offsprings, num_genes).
        :param module_weights: A dictionary, mapping each module ID in the database to be optimized to a
            weight that determines how likely it is to be chosen as a replacement during mutation.
        """
        module_weights = module_weights if module_weights is not None else dict.fromkeys(self.all_modules, 1.)
        if not set(self.all_modules).issubset(set(module_weights.keys())):
            missing = set(self.all_modules).difference(set(module_weights.keys()))
            raise ValueError(f"Modules {missing} are not covered by the mutation weights!")

        p_bases = np.array([module_weights[_id] for _id in self.base_ids])
        p_eef = np.array([module_weights[_id] for _id in self.eef_ids])
        p_bases = p_bases / p_bases.sum()
        p_eef = p_eef / p_eef.sum()

        initial_population = []
        for _ in range(population_size[0]):
            base = self.db.by_id[self.rng.choice(self.base_ids, p=p_bases)]
            eef = self.db.by_id[self.rng.choice(self.eef_ids, p=p_eef)]
            path = nx.shortest_path(self.G, base, eef)
            while len(path) < population_size[1]:
                # return edge connecting the two modules in the randomly selected location
                if len(path) == 2:
                    loc = 0  # Numpy rng.integers() does not support high=0
                else:
                    loc = self.rng.integers(low=0, high=len(path) - 2)
                if path[loc] == self.__empty_slot:
                    continue
                loc_next = loc + 1
                while path[loc_next] == self.__empty_slot:
                    loc_next = loc_next + 1
                if path[loc] is path[loc_next]:
                    num_edges = self.G.number_of_edges(path[loc], path[loc_next])
                    rand_edge = self.rng.integers(0, num_edges - 1)
                    shortest_path = [[(path[loc], path[loc_next], rand_edge)]]
                else:
                    shortest_path = list(nx.all_simple_edge_paths(self.G, path[loc], path[loc_next], cutoff=1))

                connector_parent = self.G.edges[shortest_path[0][0]]['connectors'][0]
                connector_child = self.G.edges[shortest_path[0][0]]['connectors'][1]
                neighbors_parent = set(self.G.neighbors(path[loc]))
                neighbors_child = set(self.G.neighbors(path[loc_next]))
                shared = [m for m in neighbors_parent.intersection(neighbors_child)
                          if not any(con.type in {'base', 'eef'} for con in m.available_connectors.values())]
                if connector_child.connects(connector_parent):
                    shared.append(self.__empty_slot)
                weights = np.array([module_weights[m if m == self.__empty_slot else m.id] for m in shared])
                p_draw = weights / weights.sum()
                shared = np.array(shared)[p_draw > 0]
                draw = self.rng.choice(shared, size=len(shared), replace=False, p=p_draw[p_draw > 0], shuffle=False)
                for m in draw:
                    if m == self.__empty_slot:
                        path.insert(loc_next, self.__empty_slot)
                        break
                    c_x1 = list(m.available_connectors.values())[0]
                    c_x2 = list(m.available_connectors.values())[1]
                    # check if the two connectors are compatible
                    if (connector_parent.connects(c_x2) and connector_child.connects(c_x1)) \
                            or (connector_parent.connects(c_x1) and connector_child.connects(c_x2)):
                        path.insert(loc_next, m)
                        break
            for i in range(len(path)):
                path[i] = self.id2num[path[i].id if path[i] != self.__empty_slot else self.__empty_slot]
            initial_population.append(path)

        return np.array(initial_population)
