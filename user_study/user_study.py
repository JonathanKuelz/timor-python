#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 23.01.23
import datetime
import logging
import re
import string
import time
from typing import Optional

import meshcat
import pinocchio as pin
from timor.utilities.visualization import animation, MeshcatVisualizerWithAnimation
from timor.utilities import logging as timor_logging

from common import *


TIME_IN_MINUTES = 30

viz = MeshcatVisualizerWithAnimation()
viz.initViewer()


def clear(visualizer: pin.visualize.MeshcatVisualizer) -> None:
    """Clears the meshcat visualizer."""
    visualizer.clean()
    visualizer.viewer.window.send(meshcat.commands.Delete('visuals'))
    visualizer.viewer.window.send(meshcat.commands.Delete('collisions'))


def fake_a_trajectory(assembly: ModuleAssembly, q_goals: List[np.ndarray]) -> None:
    """Shows a "trajectory" (an alternation of joint angles close to the goal) in the meshcat viewer."""
    repeats = 100  # prevent meshcat interpolating geometries
    q_extended = np.tile(np.repeat(q_goals, repeats, axis=0), [10, 1])
    dt = 2 / repeats  # show the solutions for 2s, alternating
    animation(assembly.robot, q_extended, dt=dt, visualizer=viz)


def parse_user_guess(db: ModulesDB) -> Optional[Tuple[str]]:
    """Lets the user select a chain of modules"""
    guess = input("Enter your guess: ")
    print('\n' * 5)
    if guess == 'q':
        return tuple('q',)
    values = re.split(r'[\s,]+', guess.strip())
    for id_provided in values:
        if id_provided not in db.all_module_ids:
            print("You provided an invalid module ID: {}".format(id_provided))
            print("Please try again.")
            return None
        if id_provided == BASE_MODULE:
            print("You do not need to specify the base module.")
            return None
        if id_provided == EEF_MODULE:
            print("You do not need to specify the gripper.")
            return None
    return tuple([BASE_MODULE] + values + [EEF_MODULE])


def print_overview(db: ModulesDB) -> None:
    """Shown before every guess of the user."""
    print("-" * 80)
    print("To guess a configuration, enter the IDs of the modules you want to use, separated by spaces or commas.")
    print("Available IDs are: {}".format(', '.join(sorted((mid for mid in db.all_module_ids
                                                   if mid not in {BASE_MODULE, EEF_MODULE}), key=lambda x: int(x)))))
    print("The base module and the gripper are added automatically and do not need to be specified.")
    print("To quit the study, enter 'q'.")


def welcome_user() -> None:
    """Prints a welcome message to the user."""
    url = viz.viewer.window.web_url
    print('\n' * 20)
    print("Welcome to the Timor user study!")
    print("- You will be asked to guess the best configuration of a robot.")
    print("- You will have {} minutes to guess as many configurations as possible.".format(TIME_IN_MINUTES))
    print("- Your main goal should be assembling modules s.t. the final robot can reach the displayed goals.")
    print("- You can further optimize for module costs")
    print("- You should have a printout with an overview on available modules, their IDs and their costs.")
    print("- To guess a configuration, enter the IDs of the modules you want to use, separated by spaces or commas.")
    print("- For example, to use modules 1, 2 and 3, enter '1 2 3' or '1, 2, 3'.")
    print("- As soon as you start the study, the task will be displayed in the meshcat viewer.")
    print("- You can use the meshcat viewer at the following URL: {}".format(url))
    print()
    input("Press enter to start the study.")

    print("\n" * 5)
    print("Good luck!")


def main():
    """Main function of the user study."""
    db = prepare_for_user_study(ModulesDB.from_name(DB))
    timor_logging.setLevel(logging.ERROR)

    username = None
    while username is None:
        username = input("Please enter your name: ")
        letters = string.ascii_lowercase + string.ascii_uppercase + ' -_,'
        if any(letter not in letters for letter in username):
            print("Only ascii letters are allowed for a username.")
            username = None
        if username == '':
            username = None
    handler = logging.FileHandler(filename='user_study_{}_{}.log'.format(
        username, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().handlers = [handler]  # remove the timor handler
    logging.getLogger().setLevel(logging.INFO)

    welcome_user()
    task = Task.Task.from_json_file(TASK_FILE, None)
    assembly = ModuleAssembly(db, (BASE_MODULE,))
    task.visualize(viz, robots=(assembly.robot,))

    best = float('inf')
    n_guesses = 0
    t0 = time.time()
    guess = None
    try:
        while time.time() - t0 < TIME_IN_MINUTES * 60:
            print_overview(db)
            if guess is not None:
                print("Your last guess was: {}".format(', '.join(guess)))
            guess = parse_user_guess(db)
            if guess is None or len(guess) == 0:
                logging.info("Invalid guess by user.")
                continue
            if guess[0] == 'q':
                logging.warning("User quit the study by entering q.")
                break
            n_guesses += 1
            assembly = ModuleAssembly.from_serial_modules(db, guess)
            q, suc, cost = evaluate(assembly, task)
            clear(viz)
            task.visualize(viz, robots=(assembly.robot,))
            fake_a_trajectory(assembly, q)

            logging.info("Guess number {}: {}".format(n_guesses, guess))
            logging.info("Time: {:.2f}s".format(time.time() - t0))
            logging.info("Successes: {}".format(suc))
            logging.info("Costs of robot: {}".format(cost))
            logging.info("Configurations: {}".format(q))
            logging.info("-" * 80)

            if suc == len(task.goals):
                if cost < best:
                    best = cost
                print("You found a valid configuration for all goals!")
                print("You can now proceed trying to improve cost. Your current best cost is: {}".format(best))
    except KeyboardInterrupt:
        logging.warning("User interrupted the study with Keyboard Interrupt.")
    finally:
        total_time = time.time() - t0
        logging.info("Ending study after {:.2f} seconds".format(total_time))
        print("=========================================")
        print("Thank you for participating in the study!")
        print("You guessed {} configurations in {:.1f} minutes.".format(n_guesses, total_time / 60))


if __name__ == '__main__':
    main()
