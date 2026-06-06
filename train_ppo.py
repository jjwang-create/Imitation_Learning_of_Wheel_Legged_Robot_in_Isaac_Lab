# Copyright (c) 2022-2025.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import argparse
import os
import time
from typing import Tuple

from jetbot_env import JetbotPyBulletEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        value = self.critic(obs)
        return dist, value


class PPOTrainer:
    def __init__(
        self,
        env: JetbotPyBulletEnv,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps: float = 1e-5,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        num_steps: int = 2048,
        batch_size: int = 64,
        epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.eps = eps
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.policy = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=eps)
        self.memory = deque(maxlen=num_steps)

    def select_action(self, obs: np.ndarray, training: bool = True) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(obs_t)
        if training:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            action = dist.mean
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), log_prob, value

    def compute_returns(self, rewards: np.ndarray, dones: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        last_value = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.eps * (1 - dones[t]) * advantages[t + 1] if t < len(advantages) - 1 else delta
            returns[t] = advantages[t] + values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self):
        observations, actions, old_log_probs, values, returns, advantages = self._prepare_batch()

        for _ in range(self.epochs):
            for idx in range(0, len(observations), self.batch_size):
                obs_batch = observations[idx:idx+self.batch_size]
                act_batch = actions[idx:idx+self.batch_size]
                old_log_prob_batch = old_log_probs[idx:idx+self.batch_size]
                returns_batch = returns[idx:idx+self.batch_size]
                advantages_batch = advantages[idx:idx+self.batch_size]

                dist, value = self.policy(obs_batch)
                log_probs = dist.log_prob(act_batch).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(log_probs - old_log_prob_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = ((value.squeeze(-1) - returns_batch) ** 2).mean()
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def _prepare_batch(self):
        observations = torch.stack([m["obs"] for m in self.memory]).squeeze(1).to(self.device)
        actions = torch.stack([m["action"] for m in self.memory]).squeeze(1).to(self.device)
        old_log_probs = torch.stack([m["log_prob"] for m in self.memory]).squeeze(1).to(self.device)
        values = torch.stack([m["value"] for m in self.memory]).squeeze(1).to(self.device)
        rewards = torch.FloatTensor([m["reward"] for m in self.memory]).to(self.device)
        dones = torch.FloatTensor([m["done"] for m in self.memory]).to(self.device)

        returns, advantages = self.compute_returns(
            rewards.cpu().numpy(),
            dones.cpu().numpy(),
            values.cpu().numpy()
        )
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        return observations, actions, old_log_probs, values, returns, advantages

    def collect_rollout(self):
        obs_dict = self.env.reset()
        obs = obs_dict["policy"]

        for _ in range(self.num_steps):
            action, log_prob, value = self.select_action(obs)
            next_obs, reward, done, info = self.env.step(action)

            self.memory.append({
                "obs": torch.FloatTensor(obs),
                "action": torch.FloatTensor(action),
                "log_prob": log_prob,
                "value": value.squeeze(-1),
                "reward": reward,
                "done": done,
            })

            obs = next_obs

            if done.all():
                obs_dict = self.env.reset()
                obs = obs_dict["policy"]

    def train(self, num_iterations: int, save_interval: int = 100, log_interval: int = 10):
        print(f"Starting training on {self.device}")
        print(f"Number of environments: {self.env.num_envs}")
        print(f"Number of iterations: {num_iterations}")

        for iteration in range(num_iterations):
            start_time = time.time()

            self.collect_rollout()
            self.update()

            elapsed = time.time() - start_time

            if (iteration + 1) % log_interval == 0:
                recent_rewards = [m["reward"] for m in list(self.memory)[-self.env.num_envs:]]
                mean_reward = np.mean(recent_rewards)
                print(f"Iter {iteration+1}/{num_iterations} | "
                      f"Mean reward: {mean_reward:.4f} | "
                      f"Time: {elapsed:.2f}s")

            if (iteration + 1) % save_interval == 0:
                self.save(f"policy_iter_{iteration+1}.pt")
                print(f"Saved policy to policy_iter_{iteration+1}.pt")

        self.save("policy_final.pt")
        print("Training complete! Final policy saved to policy_final.pt")

    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, os.path.join(os.path.dirname(__file__), path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded policy from {path}")


def evaluate(env: JetbotPyBulletEnv, policy: ActorCritic, num_episodes: int = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    episode_rewards = []

    for ep in range(num_episodes):
        obs_dict = env.reset()
        obs = obs_dict["policy"]
        total_reward = 0
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                dist, _ = policy(obs_t)
                action = dist.mean.cpu().numpy()

            obs, reward, done, _ = env.step(action)
            total_reward += reward.mean()

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: reward = {total_reward:.4f}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nMean reward over {num_episodes} episodes: {mean_reward:.4f} +/- {std_reward:.4f}")
    return mean_reward


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Jetbot with PyBullet")
    parser.add_argument("--num_envs", type=int, default=100, help="Number of parallel environments")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--save_interval", type=int, default=100, help="Save interval")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained policy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    env = JetbotPyBulletEnv(
        num_envs=args.num_envs,
        render_mode="gui" if args.render else None
    )

    obs_dim = 3  # [dot, cross, forward_speed]
    action_dim = 2  # [left_wheel, right_wheel] velocity targets

    trainer = PPOTrainer(
        env=env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_steps=2048,
        batch_size=64,
        epochs=10,
    )

    if args.evaluate and args.checkpoint:
        trainer.load(args.checkpoint)
        evaluate(env, trainer.policy, num_episodes=10)
    else:
        trainer.train(
            num_iterations=args.num_iterations,
            save_interval=args.save_interval,
            log_interval=args.log_interval
        )

    env.close()


if __name__ == "__main__":
    main()
