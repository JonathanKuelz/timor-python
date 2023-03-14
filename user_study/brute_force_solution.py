#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 23.01.23
import tqdm
from timor.utilities import logging
from timor.configuration_search.LiuIterator import Science2019

from common import *


def main():
    """Iterate over all assemblies with at most 6 DOF"""
    logging.setLevel('WARN')
    db = prepare_for_user_study(ModulesDB.from_name(DB))
    task = Task.Task.from_json_file(TASK_FILE, None)
    iterator = Science2019(db, min_dof=1, max_dof=5, max_links_between_joints=1,
                           max_links_after_last_joint=1, max_links_before_first_joint=0)
    best = float('inf')
    filters = default_filters(task)
    for assembly in tqdm.tqdm(iterator, total=len(iterator)):
        cost = assembly.mass
        if cost >= best:
            continue  # We will not improve
        results, suc = eval_with_filters(assembly, task, filters)
        logging.info(f"Cost: {cost}, Success: {suc}")
        if suc:
            logging.info(f"New best: {cost}")
            logging.info(f"Assembly: {', '.join(assembly.original_module_ids)}")
            print("\nFound a valid configuration for all goals!")
            best = cost
            print(', '.join(assembly.original_module_ids))
            print("New best cost: {}".format(best))
            print(results)
            print("-" * 80)


if __name__ == '__main__':
    """Run the brute force algorithm"""
    main()
