import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from model import DDQN
from optimizer import GlobalAdam


class Mario:
    def __init__(self, state_dim, action_dim, lr, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = deque(maxlen=100000)
        self.batch_size = 32

        self.epsilon = 1
        self.epsilon_decay = 0.99999975
        self.epsilon_min = 0.1
        self.gamma = 0.9
        self.lr = lr

        self.curr_step = 0
        self.burnin = 1e5
        self.learn_every = 3
        self.sync_every = 1e4

        self.save_every = 5e5
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.net = DDQN(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = GlobalAdam(self.net.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        # Exploration
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)

        # Exploitation
        else:
            state = (
                torch.FloatTensor(state).cuda()
                if self.use_cuda
                else torch.FloatTensor(state)
            )
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        # increment step
        self.curr_step += 1
        return action_idx

    def save_into_replay_buffer(self, state, next_state, action, reward, done):
        state = (
            torch.FloatTensor(state).cuda()
            if self.use_cuda
            else torch.FloatTensor(state)
        )
        next_state = (
            torch.FloatTensor(next_state).cuda()
            if self.use_cuda
            else torch.FloatTensor(next_state)
        )
        action = (
            torch.LongTensor([action]).cuda()
            if self.use_cuda
            else torch.LongTensor([action])
        )
        reward = (
            torch.DoubleTensor([reward]).cuda()
            if self.use_cuda
            else torch.DoubleTensor([reward])
        )
        done = (
            torch.BoolTensor([done]).cuda()
            if self.use_cuda
            else torch.BoolTensor([done])
        )

        self.buffer.append(
            (
                state,
                next_state,
                action,
                reward,
                done,
            )
        )

    def sample_from_replay_buffer(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample experience from buffer
        state, next_state, action, reward, done = self.sample_from_replay_buffer()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(dict(model=self.net.state_dict(), epsilon=self.epsilon), save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=("cuda" if self.use_cuda else "cpu"))
        epsilon = ckp.get("epsilon")
        state_dict = ckp.get("model")

        print(f"Loading model at {load_path} with exploration rate {epsilon}")
        self.net.load_state_dict(state_dict)
        self.epsilon = epsilon
