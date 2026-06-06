# Copyright (c) 2022-2025.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pybullet as p
import pybullet_data
from typing import Tuple, Dict, Optional


class JetbotPyBulletEnv:
    def __init__(
        self,
        num_envs: int = 100,
        episode_length_s: float = 5.0,
        sim_dt: float = 1 / 240,
        render_mode: Optional[str] = None,
    ):
        self.num_envs = num_envs
        self.episode_length_s = episode_length_s
        self.sim_dt = sim_dt
        self.render_mode = render_mode
        self.max_episode_length = int(episode_length_s / sim_dt)

        self._init_pybullet()
        self._load_robot()
        self._init_buffers()

    def _init_pybullet(self):
        if self.render_mode == "gui":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.sim_dt)

        self.plane_id = p.loadURDF("plane.urdf")

    def _load_robot(self):
        self.robot_ids = []
        for i in range(self.num_envs):
            base_pos = [(i % 10) * 2, (i // 10) * 2, 0.05]
            robot_id = p.loadURDF("racecar/racecar.urdf", base_pos)
            self.robot_ids.append(robot_id)

        num_joints = p.getNumJoints(self.robot_ids[0])
        self.joint_indices = []
        self.joint_names = []
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_ids[0], i)
            joint_name = info[1].decode("utf-8")
            if "wheel" in joint_name.lower():
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)

        if len(self.joint_indices) == 0:
            self.joint_indices = [2, 3]
            self.joint_names = ["left_wheel", "right_wheel"]

        print(f"Loaded robot with {len(self.joint_indices)} wheel joints: {self.joint_names}")

    def _init_buffers(self):
        self.commands = np.random.randn(self.num_envs, 3).astype(np.float32)
        self.commands[:, 2] = 0.0
        self.commands = self.commands / (np.linalg.norm(self.commands, axis=1, keepdims=True) + 1e-8)

        self.episode_length_buf = np.zeros(self.num_envs, dtype=np.int32)
        self.actions = np.zeros((self.num_envs, 2), dtype=np.float32)

        self.velocity = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.forwards = np.zeros((self.num_envs, 3), dtype=np.float32)

        default_root_state = np.zeros((self.num_envs, 13), dtype=np.float32)
        default_root_state[:, 2] = 0.05
        self.default_root_state = default_root_state

    def _compute_yaw_from_command(self, command: np.ndarray) -> float:
        ratio = command[1] / (command[0] + 1e-8)
        yaw = np.arctan(ratio)
        if command[0] < 0:
            if command[1] >= 0:
                yaw += np.pi
            else:
                yaw -= np.pi
        return yaw

    def reset(self, env_ids: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        for idx in env_ids:
            env_id = int(idx)
            root_pos = [(env_id % 10) * 2, (env_id // 10) * 2, 0.05]
            root_orn = p.getQuaternionFromEuler([0, 0, 0])

            p.resetBasePositionAndOrientation(
                self.robot_ids[env_id], root_pos, root_orn
            )
            p.resetBaseVelocity(self.robot_ids[env_id], [0, 0, 0], [0, 0, 0])

            for joint_idx in self.joint_indices:
                p.resetJointState(self.robot_ids[env_id], joint_idx, 0, 0)

            self.commands[env_id] = np.random.randn(3).astype(np.float32)
            self.commands[env_id, 2] = 0.0
            self.commands[env_id] = self.commands[env_id] / (
                np.linalg.norm(self.commands[env_id]) + 1e-8
            )

        self.episode_length_buf[env_ids] = 0

        observations = self._get_observations()
        return {"policy": observations}

    def _reset_idx(self, env_ids: np.ndarray):
        for idx in env_ids:
            env_id = int(idx)
            root_pos = [(env_id % 10) * 2, (env_id // 10) * 2, 0.05]
            root_orn = p.getQuaternionFromEuler([0, 0, 0])

            p.resetBasePositionAndOrientation(
                self.robot_ids[env_id], root_pos, root_orn
            )
            p.resetBaseVelocity(self.robot_ids[env_id], [0, 0, 0], [0, 0, 0])

            for joint_idx in self.joint_indices:
                p.resetJointState(self.robot_ids[env_id], joint_idx, 0, 0)

            self.commands[env_id] = np.random.randn(3).astype(np.float32)
            self.commands[env_id, 2] = 0.0
            self.commands[env_id] = self.commands[env_id] / (
                np.linalg.norm(self.commands[env_id]) + 1e-8
            )

        self.episode_length_buf[env_ids] = 0

    def _pre_physics_step(self, actions: np.ndarray):
        self.actions = actions.copy()

        for i in range(self.num_envs):
            joint_velocities = [
                actions[i, 0] * 10.0,
                actions[i, 1] * 10.0,
                actions[i, 0] * 10.0,
                actions[i, 1] * 10.0,
            ]
            p.setJointMotorControlArray(
                self.robot_ids[i],
                self.joint_indices,
                p.VELOCITY_CONTROL,
                targetVelocities=joint_velocities,
                forces=[50.0, 50.0, 50.0, 50.0]
            )

    def _apply_action(self):
        pass

    def _get_observations(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, 3), dtype=np.float32)

        for i in range(self.num_envs):
            robot_id = self.robot_ids[i]
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(robot_id)

            forward_vec = np.array([1.0, 0.0, 0.0])
            orn_matrix = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
            forward = orn_matrix @ forward_vec
            forward[2] = 0
            forward = forward / (np.linalg.norm(forward) + 1e-8)

            command = self.commands[i]
            command[2] = 0
            command_norm = command / (np.linalg.norm(command) + 1e-8)

            dot = np.dot(forward, command_norm)
            cross = np.cross(forward, command_norm)[2]
            forward_speed = lin_vel[0]

            obs[i] = [dot, cross, forward_speed]

        return obs

    def _get_rewards(self) -> np.ndarray:
        rewards = np.zeros(self.num_envs, dtype=np.float32)

        for i in range(self.num_envs):
            robot_id = self.robot_ids[i]
            lin_vel, _ = p.getBaseVelocity(robot_id)
            _, base_orn = p.getBasePositionAndOrientation(robot_id)

            orn_matrix = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
            forward = orn_matrix @ np.array([1.0, 0.0, 0.0])
            forward[2] = 0
            forward = forward / (np.linalg.norm(forward) + 1e-8)

            command = self.commands[i]
            command[2] = 0
            command_norm = command / (np.linalg.norm(command) + 1e-8)

            forward_reward = lin_vel[0]
            alignment_reward = np.dot(forward, command_norm)

            rewards[i] = forward_reward * np.exp(alignment_reward)

        return rewards

    def _get_dones(self) -> Tuple[np.ndarray, np.ndarray]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = np.zeros(self.num_envs, dtype=bool)
        return out_of_bounds, time_out

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        self._pre_physics_step(actions)

        for _ in range(2):
            p.stepSimulation()

        observations = self._get_observations()
        rewards = self._get_rewards()
        dones, time_outs = self._get_dones()

        self.episode_length_buf += 1

        if time_outs.any():
            env_ids = np.where(time_outs)[0]
            self._reset_idx(env_ids)

        infos = {"time_outs": time_outs}

        return observations, rewards, dones, infos

    def close(self):
        p.disconnect(self.client)


def main():
    env = JetbotPyBulletEnv(num_envs=4, render_mode=None)

    print("Resetting environment...")
    obs_dict = env.reset()
    print(f"Initial observations shape: {obs_dict['policy'].shape}")

    print("\nRunning 100 steps...")
    for i in range(100):
        actions = np.random.randn(env.num_envs, 2) * 0.1
        obs, rewards, dones, infos = env.step(actions)
        if i % 20 == 0:
            print(f"Step {i}: reward={rewards.mean():.4f}")

    print("\nResetting environment...")
    obs_dict = env.reset()

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
