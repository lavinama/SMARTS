from typing import Any, Dict

import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from train.action import Action as DiscreteAction
from train.info import Info
from train.observation import FilterObs
from train.reward import Reward

from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.single_agent import SingleAgent


def make(config: Dict[str, Any]) -> gym.Env:
    # Create environment
    env = gym.make(
        "smarts.env:multi-scenario-v0",
        scenario=config["scenario"],
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
        wrappers=config["wrappers"],
        action_space=config["action_space"],
        headless=not config["head"],  # If False, enables Envision display.
        visdom=config["visdom"],  # If True, enables Visdom display.
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
    )

    # Wrap env
    env = FormatObs(env=env)
    env = FormatAction(env=env, space=ActionSpaceType[config["action_space"]])
    env = Info(env=env)
    env = Reward(env=env)
    env = DiscreteAction(env=env, space=config["action_wrapper"])
    env = FilterObs(env=env)
    env = SingleAgent(env=env)

    # Check custom environment
    check_env(env)

    # Wrap env with SB3 wrappers
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(venv=env, n_stack=config["n_stack"], channels_order="first")
    env = VecMonitor(
        venv=env,
        filename=str(config["logdir"]),
        info_keywords=("is_success",),
    )

    return env


def make_all(config: Dict[str, Any]) -> gym.Env:
    # Create environment
    env = gym.make(
        "smarts.env:multi-all-scenario-v0",
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
        action_space=config["action_space"],
        headless=not config["head"],  # If False, enables Envision display.
        visdom=config["visdom"],  # If True, enables Visdom display.
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
    )

    # Wrap env
    env = FormatObs(env=env)
    env = FormatAction(env=env, space=ActionSpaceType[config["action_space"]])
    env = Info(env=env)
    env = Reward(env=env)
    env = DiscreteAction(env=env, space=config["action_wrapper"])
    env = FilterObs(env=env)
    env = SingleAgent(env=env)

    # Check custom environment
    check_env(env)

    # Wrap env with SB3 wrappers
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(venv=env, n_stack=config["n_stack"], channels_order="first")
    env = VecMonitor(
        venv=env,
        filename=str(config["logdir"]),
        info_keywords=("is_success",),
    )

    return env