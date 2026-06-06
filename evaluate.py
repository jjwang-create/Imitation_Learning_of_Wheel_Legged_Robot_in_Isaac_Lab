# Copyright (c) 2022-2025.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import argparse
import os
import cv2
from typing import Optional

from jetbot_env import JetbotPyBulletEnv


class VideoRecorder:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, output_path: str = "output.mp4"):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def add_frame(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


class Evaluator:
    def __init__(self, env: JetbotPyBulletEnv, policy_path: str, device: str = "cuda"):
        self.env = env
        self.device = device

        from train_ppo import ActorCritic
        self.policy = ActorCritic(obs_dim=3, action_dim=2).to(device)

        checkpoint = torch.load(policy_path, map_location=device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.eval()
        print(f"Loaded policy from {policy_path}")

    def evaluate(self, num_episodes: int = 5, render: bool = True, record_video: bool = False, video_path: Optional[str] = None):
        episode_rewards = []
        episode_lengths = []

        if record_video:
            video_recorder = VideoRecorder(width=640, height=480, fps=30, output_path=video_path)
            print(f"Recording video to {video_path}")

        for ep in range(num_episodes):
            obs_dict = self.env.reset()
            obs = obs_dict["policy"]
            total_reward = 0
            step_count = 0
            done = False
            env_done = np.zeros(self.env.num_envs, dtype=bool)

            print(f"\nEpisode {ep + 1}/{num_episodes}")

            while not done:
                obs_t = torch.FloatTensor(obs).to(self.device)
                with torch.no_grad():
                    dist, _ = self.policy(obs_t)
                    action = dist.mean.cpu().numpy()

                obs, reward, dones, _ = self.env.step(action)
                env_done = env_done | dones
                total_reward += reward[0]
                step_count += 1

                done = env_done[0] or (step_count >= self.env.max_episode_length)

                if render:
                    self.env.render()

                if record_video and ep == 0:
                    frame = self._create_frame(step_count, total_reward)
                    video_recorder.add_frame(frame)

                if step_count >= self.env.max_episode_length:
                    done = True

            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            print(f"  Steps: {step_count}, Total reward: {total_reward:.4f}")

        if record_video:
            video_recorder.release()
            print(f"Video saved to {video_path}")

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)

        print(f"\n{'='*50}")
        print(f"Evaluation Results:")
        print(f"  Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")
        print(f"  Mean episode length: {mean_length:.1f}")
        print(f"{'='*50}")

        return mean_reward, episode_rewards

    def _create_frame(self, step: int, reward: float) -> np.ndarray:
        import pybullet as p

        width, height = 640, 480
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=3.0,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height, nearVal=0.1, farVal=100.0
        )

        img_arr = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(img_arr[2], dtype=np.uint8)
        rgb_array = rgb_array.reshape((height, width, 4))[:, :, :3]
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        cv2.putText(rgb_array, f"Step: {step}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(rgb_array, f"Reward: {reward:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return rgb_array

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy on Jetbot")
    parser.add_argument("--checkpoint", type=str, default="policy_final.pt",
                        help="Path to trained policy checkpoint")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Enable GUI rendering")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable GUI rendering")
    parser.add_argument("--record-video", action="store_true",
                        help="Record video of first episode")
    parser.add_argument("--video-path", type=str, default="evaluation.mp4",
                        help="Output video path")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Number of parallel environments (use low number for evaluation)")
    args = parser.parse_args()

    env = JetbotPyBulletEnv(
        num_envs=args.num_envs,
        render_mode="gui" if (args.render and not args.no_render) else None
    )

    evaluator = Evaluator(env, args.checkpoint)
    evaluator.evaluate(
        num_episodes=args.num_episodes,
        render=args.render and not args.no_render,
        record_video=args.record_video,
        video_path=args.video_path
    )

    evaluator.close()


if __name__ == "__main__":
    main()
