#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 23.01.23
from datetime import datetime
import itertools
import json
import pickle

import matplotlib.pyplot as plt
import optuna
import random
import pygad
from timor.utilities import logging, module_classification

from common import *
from ga_optimize import optimize

OPTUNA_NUM_GENERATIONS = 100
GA_FILE = Path(__file__).parent.joinpath('GA_results.log')
HISTORY = Path(__file__).parent.joinpath('fitness_history')
STUDY_FILE = Path(__file__).parent / 'hp_optim.pkl'


def local_log(*msg, console=True):
    """Log to file and console"""
    msg = ' '.join(map(str, msg))
    dt = datetime.today().strftime('%m-%d-%H-%M-%S')
    with GA_FILE.open('a') as f:
        f.write(f"{dt}: {msg}")
        f.write('\n')
    if console:
        print(msg)


def main(trial=None) -> float:
    """
    Iterate over all assemblies with at most 6 DOF

    :param trial: An optuna trial object if this main function is used for HP optimization
    """
    local_log("---------- Starting new run ----------", console=False)
    logging.setLevel('WARN')
    db = prepare_for_user_study(ModulesDB.from_name(DB))
    task = Task.Task.from_json_file(TASK_FILE, None)
    filters = default_filters(task)
    bases, links, joints, eefs = map(lambda DB: sorted(DB, key=lambda m: m.id), module_classification.divide_db_in_types(db))
    num2id = {i: m.id for i, m in zip(range(1, len(db) + 1), itertools.chain(bases, eefs, joints, links))}
    num_iter = 0

    def fitness(solution: List[int], solution_idx: int) -> float:
        """The fitness function."""
        nonlocal num_iter
        num_iter += 1
        solution = [1] + [i for i in solution if i != 0] + [2]  # add B and E
        robot = map(num2id.get, solution)
        assembly = ModuleAssembly.from_serial_modules(db, robot)
        results, suc = eval_with_filters(assembly, task, filters)
        if suc:
            f = 1 / assembly.mass
        else:
            f = 1 / float('inf')
        if trial is not None:
            trial.report(f, num_iter)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return f

    if trial is not None:
        hp = {
            'population_size': trial.suggest_int('population_size', 5, 50),
            'num_generations': OPTUNA_NUM_GENERATIONS,
            'mutation_probability': trial.suggest_float('mutation_probability', 0.005, 0.3),
            'num_parents_mating': trial.suggest_int('num_parents_mating', 1, 5),
            'crossover_type': trial.suggest_categorical('crossover_type', ['single_point', 'two_points']),
            'keep_elitism': trial.suggest_int('keep_elitism', 1, 5),
            'save_best_solutions': False,
            'save_solutions': False,
        }
        hp['keep_parents'] = trial.suggest_int('keep_parents', 1, hp['num_parents_mating'])
        local_log(json.dumps(hp), console=False)
    else:
        hp = None

    ga_instance = optimize(db, fitness_func=fitness, hyperparameters=hp)

    to_console = True if trial is None else False  # Only log to console if we are not in optuna
    names_for_best_solution = tuple(db.by_id[num2id[i]].name for i in ga_instance.best_solution()[0] if i != 0)
    local_log("Best solution:", ', '.join(names_for_best_solution), console=to_console)
    local_log("Best solution cost:", 1 / ga_instance.best_solution()[1], console=to_console)
    local_log("Best solution found in generation:", ga_instance.best_solution_generation, console=to_console)

    fitness_hist = ga_instance.best_solutions_fitness
    savefile = HISTORY.joinpath('best_solutions_fitness_' + datetime.now().strftime('%m-%d-%H-%M-%S') + '.pkl')
    with savefile.open('wb') as f:
        pickle.dump(fitness_hist, f)

    # if to_console:
        # Let's see some visuals
        # plt.close('all')
        # ga_instance.plot_fitness()
        # ga_instance.plot_genes()
        # ga_instance.plot_new_solution_rate()

    return 1 / ga_instance.best_solution()[1]  # Return a "reward" for optuna


def hp_tuning():
    """Optimize hyperparameters with optuna"""
    hyperband_pruner = optuna.pruners.HyperbandPruner(min_resource=500)
    study = optuna.create_study(direction='minimize', pruner=hyperband_pruner)
    try:
        # Parallelization can be done, for instructions, see:
        # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
        study.optimize(main, n_trials=200, n_jobs=1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Best trial:")
        print(study.best_trial)
        print("Saving study at {}".format(str(STUDY_FILE)))
        with STUDY_FILE.open('wb') as f:
            pickle.dump(study, f)

    from optuna.visualization import plot_contour
    fig = plot_contour(study)
    fig.write_html(str(STUDY_FILE) + '.html')


if __name__ == '__main__':
    """Run the evolutionary algorithm."""
    # hp_tuning()  # Runs the hyperparameter tuning, but be aware that it takes a long time
    HISTORY.mkdir(exist_ok=True)
    for _ in range(50):
        main()  # Runs the main function with default hyperparameters
