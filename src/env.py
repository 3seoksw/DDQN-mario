import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrapper import SkipFrame, CustomReward


def create_train_env(world, stage, action_type):
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v1")

    if action_type == "SIMPLE":
        actions = SIMPLE_MOVEMENT
    elif action_type == "RIGHT":
        actions = RIGHT_ONLY
    elif action_type == "COMPLEX":
        actions = COMPLEX_MOVEMENT
    else:
        actions = None
        raise ValueError

    env = JoypadSpace(env, actions)
    env = CustomReward(env)
    env = SkipFrame(env, skip=4)

    return env, env.observation_space.shape, len(actions)