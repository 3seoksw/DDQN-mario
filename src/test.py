import argparse
import os
from pathlib import Path

import torch

from agent import Mario
from env import create_train_env


def parse_args():
    parser = argparse.ArgumentParser("""DDQN Implementation: Test""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="SIMPLE")
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument(
        "--checkpoint", type=str, default="2024-01-26T15-00-00/mario_net_1"
    )
    parser.add_argument("--test_episodes", type=int, default=10)

    args = parser.parse_args()
    return args


def test(args):
    record_path = f"../videos/mario_{args.world}_{args.stage}_1.mp4"
    checkpoint = Path(f"checkpoints/{args.checkpoint}.chkpt")

    env, state_dim, num_actions = create_train_env(
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        record_path=record_path,
    )

    agent = Mario(
        state_dim=state_dim,
        action_dim=num_actions,
        lr=args.lr,
        save_dir=None,
        checkpoint=checkpoint,
    )

    for e in range(args.test_episodes):
        state, info = env.reset()
        state = torch.from_numpy(state).float()

        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(next_state).float()

            agent.save_into_replay_buffer(state, next_state, action, reward, done)

            state = next_state

            if done or info["flag_get"]:
                break

        print(f"{e} episode: {args.world}-{args.stage} Completed")


if __name__ == "__main__":
    args = parse_args()
    test(args)
