import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from recorder import Monitor
from wrapper import CustomReward, SkipFrame


def create_train_env(world, stage, action_type, record_path=None):
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v1")

    if record_path:
        monitor = Monitor(256, 240, record_path)
    else:
        monitor = None

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
    env = CustomReward(env, monitor=None)
    env = SkipFrame(env, skip=4)

    return env, env.observation_space.shape, len(actions)
