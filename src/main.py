import argparse
import datetime
from pathlib import Path

import torch

from agent import Mario
from env import create_train_env
from logger import Logger


def parse_args():
    parser = argparse.ArgumentParser("""DDQN Implementation""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="COMPLEX")
    parser.add_argument("--num_episodes", type=int, default=16000)
    parser.add_argument("--lr", type=float, default=0.00025)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    env, state_dim, num_actions = create_train_env(
        args.world, args.stage, args.action_type
    )

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True)
    checkpoint = None

    agent = Mario(
        state_dim=state_dim,
        action_dim=num_actions,
        lr=args.lr,
        save_dir=save_dir,
        checkpoint=checkpoint,
    )
    logger = Logger(save_dir)

    num_episodes = 0
    flag_count = 0
    for e in range(args.num_episodes):
        state, info = env.reset()
        state = torch.from_numpy(state).float()

        while True:
            env.render()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(next_state).float()

            agent.save_into_replay_buffer(state, next_state, action, reward, done)

            q, loss = agent.learn()

            logger.log_step(reward, loss, q)

            state = next_state

            if done or info["flag_get"]:
                if info["flag_get"]:
                    flag_count += 1
                    print(f"Flag reached at episode {e}")
                    logger.record(
                        episode=e, epsilon=agent.epsilon, step=agent.curr_step
                    )
                    agent.save()
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(episode=e, epsilon=agent.epsilon, step=agent.curr_step)

        num_episodes += 1
    print(f"Number of episodes reached to the flag: {flag_count}")
