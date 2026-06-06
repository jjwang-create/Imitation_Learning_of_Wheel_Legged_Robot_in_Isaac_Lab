import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Optional, Tuple


class HuskyPyBulletEnv:
    """Differential-drive Husky robot tracking random direction commands."""

    def __init__(self, num_envs: int = 100, render_mode: Optional[str] = None):
        self.num_envs = num_envs
        self.render_mode = render_mode
        self.sim_dt = 1.0 / 240.0
        self.decimation = 4
        self.max_episode_length = 500
        self.command_resample_interval = 150

        if render_mode == "gui":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.sim_dt)
        p.loadURDF("plane.urdf")

        self._load_robots()
        self._init_buffers()

    def _load_robots(self):
        self.robot_ids = []
        self.wheel_joints = []
        cols = 10
        for i in range(self.num_envs):
            x = (i % cols) * 3.0
            y = (i // cols) * 3.0
            rid = p.loadURDF("husky/husky.urdf", [x, y, 0.15])
            self.robot_ids.append(rid)

        num_joints = p.getNumJoints(self.robot_ids[0])
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_ids[0], i)
            name = info[1].decode("utf-8")
            if "wheel" in name.lower() and info[2] == 0:
                self.wheel_joints.append(i)

        print(f"Husky loaded: {len(self.wheel_joints)} wheel joints -> {[p.getJointInfo(self.robot_ids[0], j)[1].decode() for j in self.wheel_joints]}")

    def _init_buffers(self):
        self.commands = np.random.randn(self.num_envs, 3).astype(np.float32)
        self.commands[:, 2] = 0.0
        self.commands = self.commands / (np.linalg.norm(self.commands, axis=1, keepdims=True) + 1e-8)
        self.episode_length_buf = np.zeros(self.num_envs, dtype=np.int32)
        self.command_step_buf = np.zeros(self.num_envs, dtype=np.int32)
        self.actions = np.zeros((self.num_envs, 2), dtype=np.float32)

    def reset(self, env_ids: Optional[np.ndarray] = None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        cols = 10
        for idx in env_ids:
            i = int(idx)
            x = (i % cols) * 3.0
            y = (i // cols) * 3.0
            p.resetBasePositionAndOrientation(self.robot_ids[i], [x, y, 0.15], [0, 0, 0, 1])
            p.resetBaseVelocity(self.robot_ids[i], [0, 0, 0], [0, 0, 0])
            for j in self.wheel_joints:
                p.resetJointState(self.robot_ids[i], j, 0, 0)
            self.commands[i] = np.random.randn(3).astype(np.float32)
            self.commands[i, 2] = 0.0
            self.commands[i] = self.commands[i] / (np.linalg.norm(self.commands[i]) + 1e-8)
        self.episode_length_buf[env_ids] = 0
        self.command_step_buf[env_ids] = 0
        return {"policy": self._get_observations()}

    def step(self, actions: np.ndarray):
        self.actions = np.clip(actions, -1.0, 1.0)

        for i in range(self.num_envs):
            left_speed = self.actions[i, 0] * 15.0
            right_speed = self.actions[i, 1] * 15.0
            p.setJointMotorControl2(self.robot_ids[i], self.wheel_joints[0], p.VELOCITY_CONTROL, targetVelocity=left_speed, force=100)
            p.setJointMotorControl2(self.robot_ids[i], self.wheel_joints[1], p.VELOCITY_CONTROL, targetVelocity=right_speed, force=100)
            p.setJointMotorControl2(self.robot_ids[i], self.wheel_joints[2], p.VELOCITY_CONTROL, targetVelocity=left_speed, force=100)
            p.setJointMotorControl2(self.robot_ids[i], self.wheel_joints[3], p.VELOCITY_CONTROL, targetVelocity=right_speed, force=100)

        for _ in range(self.decimation):
            p.stepSimulation()

        obs = self._get_observations()
        rewards = self._get_rewards()
        dones, time_outs = self._get_dones()
        self.episode_length_buf += 1
        self.command_step_buf += 1

        # Resample commands periodically to force continuous tracking
        resample = self.command_step_buf >= self.command_resample_interval
        if resample.any():
            self._resample_commands(np.where(resample)[0])

        if time_outs.any():
            self._reset_idx(np.where(time_outs)[0])

        return obs, rewards, dones, {"time_outs": time_outs}

    def _reset_idx(self, env_ids: np.ndarray):
        cols = 10
        for idx in env_ids:
            i = int(idx)
            x = (i % cols) * 3.0
            y = (i // cols) * 3.0
            p.resetBasePositionAndOrientation(self.robot_ids[i], [x, y, 0.15], [0, 0, 0, 1])
            p.resetBaseVelocity(self.robot_ids[i], [0, 0, 0], [0, 0, 0])
            for j in self.wheel_joints:
                p.resetJointState(self.robot_ids[i], j, 0, 0)
            self.commands[i] = np.random.randn(3).astype(np.float32)
            self.commands[i, 2] = 0.0
            self.commands[i] = self.commands[i] / (np.linalg.norm(self.commands[i]) + 1e-8)
        self.episode_length_buf[env_ids] = 0
        self.command_step_buf[env_ids] = 0

    def _resample_commands(self, env_ids: np.ndarray):
        for idx in env_ids:
            i = int(idx)
            self.commands[i] = np.random.randn(3).astype(np.float32)
            self.commands[i, 2] = 0.0
            self.commands[i] = self.commands[i] / (np.linalg.norm(self.commands[i]) + 1e-8)
        self.command_step_buf[env_ids] = 0

    def _get_observations(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, 4), dtype=np.float32)
        for i in range(self.num_envs):
            _, base_orn = p.getBasePositionAndOrientation(self.robot_ids[i])
            lin_vel, ang_vel = p.getBaseVelocity(self.robot_ids[i])
            orn_mat = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
            forward = orn_mat[:, 0].copy()
            forward[2] = 0
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            cmd = self.commands[i].copy()
            cmd[2] = 0
            cmd = cmd / (np.linalg.norm(cmd) + 1e-8)
            dot = np.dot(forward, cmd)
            cross = np.cross(forward, cmd)[2]
            forward_speed = lin_vel[0]
            yaw_rate = ang_vel[2]
            obs[i] = [dot, cross, forward_speed, yaw_rate]
        return obs

    def _get_rewards(self) -> np.ndarray:
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        for i in range(self.num_envs):
            _, base_orn = p.getBasePositionAndOrientation(self.robot_ids[i])
            lin_vel, _ = p.getBaseVelocity(self.robot_ids[i])
            orn_mat = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
            forward = orn_mat[:, 0].copy()
            forward[2] = 0
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            cmd = self.commands[i].copy()
            cmd[2] = 0
            cmd = cmd / (np.linalg.norm(cmd) + 1e-8)
            alignment = np.dot(forward, cmd)
            forward_speed = lin_vel[0]
            rewards[i] = forward_speed * np.exp(2.0 * (alignment - 1.0)) + 0.1 * alignment
        return rewards

    def _get_dones(self) -> Tuple[np.ndarray, np.ndarray]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = np.zeros(self.num_envs, dtype=bool)
        return out_of_bounds, time_out

    def close(self):
        p.disconnect(self.client)


if __name__ == "__main__":
    env = HuskyPyBulletEnv(num_envs=4, render_mode=None)
    obs_dict = env.reset()
    print(f"Obs shape: {obs_dict['policy'].shape}")
    for step in range(200):
        actions = np.random.randn(env.num_envs, 2) * 0.3
        obs, rewards, dones, _ = env.step(actions)
        if step % 40 == 0:
            print(f"Step {step}: reward={rewards.mean():.3f}")
    env.close()
    print("Done!")
