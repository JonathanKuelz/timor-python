#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 15.02.23
import itertools
from typing import Callable, List

import random
import pygad
from timor import ModulesDB
from timor.utilities import module_classification


hp_defaults = {  # These are the result of a previous optimization
    'population_size': 50,
    'num_generations': 500,
    'mutation_probability': 0.237,
    'num_parents_mating': 5,
    'keep_parents': 4,
    'crossover_type': 'single_point',
    'keep_elitism': 5,
    'save_best_solutions': True,
    'save_solutions': True
}


def optimize(db: ModulesDB,
             fitness_func: Callable[[List[int], int], float],
             genome_structure: str = 'B-L-J-L-J-L-J-L-J-L-J-L-J-L-E',
             hyperparameters: dict = None) -> pygad.GA:
    """
    Takes a modules db and optimizes assemblies given a fitness function.

    :param db: The modules db
    :param fitness_func: Takes a solution (list of module ids) and a solution_index (int) and returns a fitness value
    :param genome_structure: A string of the form 'X-X-...X' where X can be B (base), L (link), J (joint), E (end effector)
    :param hyperparameters: A dictionary of hyperparameters to override the defaults
    """
    bases, links, joints, eefs = map(lambda DB: sorted(DB, key=lambda m: m.id), module_classification.divide_db_in_types(db))
    link_ids = tuple(m.id for m in links)
    joint_ids = tuple(m.id for m in joints)

    genes = genome_structure.split('-')[1:-1]  # remove B and E which are not part of the optimization
    num2id = {i: m.id for i, m in zip(range(1, len(db) + 1), itertools.chain(bases, eefs, joints, links))}
    id2num = {ID: i for i, ID in num2id.items()}
    id2num['000'] = 0  # this encodes the "no module" option

    def get_gene_space(genome: str) -> List[int]:
        """Maps L: link or nothing and J: joint or nothing"""
        if genome == 'L':
            return [0] + list(map(id2num.get, link_ids))
        elif genome == 'J':
            return [0] + list(map(id2num.get, joint_ids))

    J = id2num[joint_ids[0]]
    L = lambda: id2num[random.choice(link_ids + ('000',))]  # Link ID

    hp = hp_defaults.copy()
    if hyperparameters is not None:
        hp.update(hyperparameters)

    initial_population = [  # Create a couple random solutions that do have modules at all positions
        [L(), J, L(), J, L(), J, L(), J, L(), J, L(), J, L()] for _ in range(hp.pop('population_size'))
    ]
    ga_instance = pygad.GA(
        num_genes=len(genes),
        fitness_func=fitness_func,
        gene_space=[get_gene_space(g) for g in genes],
        gene_type=int,
        initial_population=initial_population,
        **hp
    )

    ga_instance.run()

    return ga_instance
