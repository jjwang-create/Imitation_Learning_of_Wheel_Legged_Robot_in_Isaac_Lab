# Copyright (c) 2022-2025.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import argparse
import cv2
import math

from jetbot_env import JetbotPyBulletEnv


def compute_camera_matrix(robot_pos, robot_yaw, distance=4.0, pitch=-45):
    yaw = math.degrees(robot_yaw) + 180
    camera_target = robot_pos

    cam_eye_x = robot_pos[0] + distance * math.cos(math.radians(yaw))
    cam_eye_y = robot_pos[1] + distance * math.sin(math.radians(yaw))
    cam_eye_z = robot_pos[2] + distance * math.sin(math.radians(-pitch))

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[cam_eye_x, cam_eye_y, cam_eye_z],
        cameraTargetPosition=camera_target,
        cameraUpVector=[0, 0, 1]
    )
    return view_matrix


def draw_arrow(frame, center, angle, length, color, thickness=2):
    end_x = int(center[0] + length * math.cos(angle))
    end_y = int(center[1] - length * math.sin(angle))
    cv2.arrowedLine(frame, (int(center[0]), int(center[1])), (end_x, end_y), color, thickness)


def draw_info_panel(frame, text_lines, position="top_left"):
    y0, dy = 30, 35
    for i, line in enumerate(text_lines):
        y = y0 + i * dy
        if position == "top_left":
            x = 15
        else:
            x = frame.shape[1] - 250
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    import pybullet as p

    parser = argparse.ArgumentParser(description="Generate visualization video")
    parser.add_argument("--checkpoint", type=str, default="policy_final.pt")
    parser.add_argument("--output", type=str, default="jetbot_demo.mp4")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    env = JetbotPyBulletEnv(num_envs=1, render_mode=None)
    client_id = env.client

    from train_ppo import ActorCritic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ActorCritic(obs_dim=3, action_dim=2).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    print(f"Loaded policy from {args.checkpoint}")

    obs_dict = env.reset()
    obs = obs_dict["policy"]

    width, height = 800, 600
    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Recording to {args.output}...")

    for step in range(args.steps):
        obs_t = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            dist, _ = policy(obs_t)
            action = dist.mean.cpu().numpy()

        obs, reward, done, _ = env.step(action)

        robot_id = env.robot_ids[0]
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(robot_id)

        euler = p.getEulerFromQuaternion(base_orn)
        robot_yaw = euler[2]

        forward_vec = np.array([1.0, 0.0, 0.0])
        orn_matrix = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        forward = orn_matrix @ forward_vec

        command = env.commands[0]
        command[2] = 0
        command_norm = command / (np.linalg.norm(command) + 1e-8)

        dot = np.dot(forward, command_norm)
        cross = np.cross(forward, command_norm)[2]

        camera_distance = 5.0
        cam_x = base_pos[0] - camera_distance * math.cos(robot_yaw + math.pi)
        cam_y = base_pos[1] - camera_distance * math.sin(robot_yaw + math.pi)
        cam_z = base_pos[2] + 4.0

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[cam_x, cam_y, cam_z],
            cameraTargetPosition=[base_pos[0], base_pos[1], base_pos[2]],
            cameraUpVector=[0, 0, 1]
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
        frame = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        frame_center_x = width // 2 + 50
        frame_center_y = height // 2 + 50
        arrow_scale = 60

        forward_angle = -robot_yaw
        forward_end_x = frame_center_x + int(arrow_scale * math.cos(forward_angle))
        forward_end_y = frame_center_y - int(arrow_scale * math.sin(forward_angle))
        cv2.arrowedLine(frame, (frame_center_x, frame_center_y), (forward_end_x, forward_end_y), (0, 255, 255), 3)

        command_yaw = math.atan2(command[1], command[0])
        command_end_x = frame_center_x + int(arrow_scale * math.cos(command_yaw))
        command_end_y = frame_center_y - int(arrow_scale * math.sin(command_yaw))
        cv2.arrowedLine(frame, (frame_center_x, frame_center_y), (command_end_x, command_end_y), (0, 0, 255), 3)

        cv2.circle(frame, (frame_center_x, frame_center_y), 8, (255, 255, 255), -1)

        info_lines = [
            f"Step: {step}/{args.steps}",
            f"Reward: {reward[0]:.2f}",
            f"Speed: {lin_vel[0]:.2f} m/s",
            f"Align: {dot:.3f}",
        ]
        draw_info_panel(frame, info_lines, position="top_left")

        cmd_lines = [
            "Legend:",
            "Yellow: Forward",
            "Red: Target",
            "",
            f"Yaw: {math.degrees(robot_yaw):.1f}°",
            f"Cross: {cross:.3f}",
        ]
        draw_info_panel(frame, cmd_lines, position="top_right")

        writer.write(frame)

        if step % 60 == 0:
            print(f"Progress: {step}/{args.steps} ({100*step/args.steps:.1f}%)")

    writer.release()
    env.close()
    print(f"\nVideo saved to: {args.output}")


if __name__ == "__main__":
    main()
