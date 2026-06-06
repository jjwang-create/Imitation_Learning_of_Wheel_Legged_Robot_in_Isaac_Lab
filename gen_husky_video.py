import numpy as np
import torch
import cv2
import argparse
import os
import math
import pybullet as p

from husky_env import HuskyPyBulletEnv
from train_husky import ActorCritic


def draw_info_panel(frame, lines, position="top_left"):
    h, w = frame.shape[:2]
    if position == "top_left":
        x, y = 10, 30
    elif position == "top_right":
        x = w - 250
        y = 30
    else:
        x, y = 10, h - 100
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="policy_final.pt")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--output", type=str, default="husky_demo.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    env = HuskyPyBulletEnv(num_envs=1, render_mode=None)
    policy = ActorCritic(obs_dim=4, action_dim=2, hidden_dim=128).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"Loaded {args.checkpoint}")

    width, height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))

    obs_dict = env.reset()
    obs = obs_dict["policy"]
    robot_id = env.robot_ids[0]

    for step in range(args.steps):
        obs_t = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            dist, _ = policy(obs_t)
            action = dist.mean.cpu().numpy()
        obs, reward, done, _ = env.step(action)

        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(robot_id)
        euler = p.getEulerFromQuaternion(base_orn)
        robot_yaw = euler[2]

        view_mat = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[base_pos[0], base_pos[1], 0.3],
            distance=4.0, yaw=0, pitch=-60, roll=0, upAxisIndex=2,
        )
        proj_mat = p.computeProjectionMatrixFOV(fov=60, aspect=width/height, nearVal=0.1, farVal=50)
        img = p.getCameraImage(width, height, viewMatrix=view_mat, projectionMatrix=proj_mat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(img[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        cx, cy = width // 2, height // 2 + 60
        scale = 50
        fwd_angle = -robot_yaw
        fwd_x = cx + int(scale * math.cos(fwd_angle))
        fwd_y = cy - int(scale * math.sin(fwd_angle))
        cv2.arrowedLine(frame, (cx, cy), (fwd_x, fwd_y), (0, 255, 255), 3)

        cmd = env.commands[0]
        cmd_yaw = math.atan2(cmd[1], cmd[0])
        cmd_x = cx + int(scale * math.cos(cmd_yaw))
        cmd_y = cy - int(scale * math.sin(cmd_yaw))
        cv2.arrowedLine(frame, (cx, cy), (cmd_x, cmd_y), (0, 0, 255), 3)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)

        dot = obs[0, 0]
        cross = obs[0, 1]
        draw_info_panel(frame, [
            f"Step: {step}/{args.steps}",
            f"Reward: {reward[0]:.3f}",
            f"Speed: {lin_vel[0]:.2f} m/s",
            f"Align(dot): {dot:.3f}",
        ], "top_left")
        draw_info_panel(frame, [
            "Yellow: Forward",
            "Red: Target",
            f"Yaw: {math.degrees(robot_yaw):.0f} deg",
            f"Cross: {cross:.3f}",
        ], "top_right")

        writer.write(frame)
        if step % 60 == 0:
            print(f"Progress: {step}/{args.steps}")

    writer.release()
    env.close()
    print(f"Video saved: {args.output}")


if __name__ == "__main__":
    main()
