from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from timor import Goals, ModulesDB, ModuleAssembly, Task
from timor.configuration_search import AssemblyFilter

BASE_MODULE = '103'
DB = 'modrob-gen2'
EEF_MODULE = 'GEP2010IL'
TASK_FILE = Path(__file__).parent.joinpath("user_study_task.json")


def prepare_for_user_study(db: ModulesDB) -> ModulesDB:
    """Prepares the database for the user study by adding a base module."""
    keep_connectors = {c.id for c in db.all_connectors if c.size in ([80], [])}
    removed_bases = {'104', '105', '105'}
    removed_links = {'21', '22', '10', '17', '8', '15', '14', '8'}  # There are a lot of links that are quite similar
    discard = removed_bases | removed_links | {'61'}  # 61 is a test module
    modules = {m for m in db if all(c in keep_connectors for c in m.available_connectors) and m.id not in discard}
    return ModulesDB(modules, name='User Study Modules', package_dir=db._package_dir)


def default_filters(task: Task.Task) -> Tuple[AssemblyFilter.AssemblyFilter, ...]:
    """Returns the default filters used in the user study."""
    create_robot = AssemblyFilter.RobotCreationFilter()
    ik_simple = AssemblyFilter.InverseKinematicsSolvable(ignore_self_collision=True, max_iter=150)
    ik_complex = AssemblyFilter.InverseKinematicsSolvable(task=task, max_iter=5000, check_static_torques=True)
    return create_robot, ik_simple, ik_complex


def evaluate(assembly: ModuleAssembly, task: Task.Task, info: bool = True) -> Tuple[List[np.ndarray], int, float]:
    """Evaluates a user guess and prints the result."""
    q_goals = []
    successes = 0
    cost = sum(m.mass for m in assembly.module_instances)
    for goal in task.goals:
        assert isinstance(goal, Goals.At), "Error in the user study, unexpected goal."
        q, success = assembly.robot.ik(goal.goal_pose, max_iter=1000, ignore_self_collision=True)
        if success:  # We don't even try if we do not find a solution without collisions
            q, success = assembly.robot.ik(goal.goal_pose, task=task, q_init=q, max_iter=5000)
        if success:
            successes += 1
            if info:
                print("You found a valid configuration for goal {}!".format(goal.id))
        else:
            if info:
                print("You did not find a valid configuration for goal {}.".format(goal.id))
        q_goals.append(q)
    if info:
        print("The total cost of your configuration is {}.".format(cost))
        print("You can see your configuration in the meshcat viewer.")
    return q_goals, successes, cost


def eval_with_filters(assembly: ModuleAssembly, task: Task.Task,
                      filters: Tuple[AssemblyFilter.AssemblyFilter, ...]
                      ) -> Tuple[AssemblyFilter.IntermediateFilterResults, int]:
    """Evaluates whether the assembly is a valid solution for the task.

    :param assembly: The assembly to evaluate
    :param task: The task to evaluate the assembly for
    :param filters: The filters to apply to the assembly in evaluation
    """
    success, results = run_filters(filters, assembly, task)
    return results, success


def run_filters(filters: Iterable[AssemblyFilter.AssemblyFilter],
                assembly: ModuleAssembly,
                task: Task.Task) -> Tuple[bool, AssemblyFilter.IntermediateFilterResults]:
    """
    Runs a sequence of filters on an assembly.

    :param filters: A sequence of filter types
    :param assembly: The assembly to be checked
    :param task: The task to be evaluated
    :return: True, if the assembly passes all filters + all results generated by the filters
    """
    AssemblyFilter.assert_filters_compatible((type(f) for f in filters))
    results = AssemblyFilter.IntermediateFilterResults()
    for fltr in filters:
        if not fltr.check(assembly, task, results):
            return False, results
    return True, results
