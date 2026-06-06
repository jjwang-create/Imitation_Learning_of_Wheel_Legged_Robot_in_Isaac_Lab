# Imitation Learning of Wheel-Legged Robot in Isaac Lab

基于 Isaac Lab 的轮足机器人模仿学习项目，包含 PyBullet 仿真环境和 Isaac Lab 原生环境。

## 项目结构

```
Imitation_Learning_of_Wheel_Legged_Robot_in_Isaac_Lab/
├── IsaacLab_PyBullet/          # PyBullet 轻量级仿真环境（无需 GPU）
│   ├── jetbot_env.py          # Jetbot 机器人环境
│   ├── husky_env.py           # Husky 机器人环境
│   ├── train_ppo.py           # PPO 训练脚本
│   ├── evaluate.py             # 策略评估与视频录制
│   ├── gen_video.py           # 仿真视频生成
│   └── train_husky.py         # Husky 训练脚本
│
├── source/                     # Isaac Lab 源代码
│   ├── direct_rl/             # 直接 RL 任务
│   └── isaaclab_tasks/        # 各类机器人任务
│
└── scripts/                   # 运行脚本
```

## 快速开始

### PyBullet 仿真（推荐，无需 GPU）

```bash
cd IsaacLab_PyBullet
pip install pybullet torch numpy opencv-python-headless

# 训练
python train_ppo.py --num_envs 100 --num_iterations 500

# 评估
python evaluate.py --checkpoint policy_final.pt

# 生成视频
python gen_video.py --steps 2400 --output demo.mp4
```

### Isaac Lab（需要 GPU 和 Isaac Sim）

```bash
# 安装依赖
./scripts/install_deps.sh

# 训练
python scripts/train.py --agent rsl_rl --task IsaacLab_DirectRL_Jetbot-v0

# 评估
python scripts/eval.py --agent rsl_rl --task IsaacLab_DirectRL_Jetbot-v0
```

## 机器人模型

### Jetbot
- 4 轮差速驱动
- 前方相机
- 用于导航和跟踪任务

### Husky
- 4 轮差速驱动
- 更重的负载能力
- 适合室外环境

## 训练算法

- **PPO** (Proximal Policy Optimization)
- 支持多环境并行
- GPU 加速训练

## 环境说明

| 环境 | 观测空间 | 动作空间 | 说明 |
|------|----------|----------|------|
| Jetbot | 3D (目标方向) | 2D (线速度, 角速度) | 目标跟踪 |
| Husky | 3D (目标方向) | 2D (线速度, 角速度) | 目标跟踪 |

## 依赖

- Python 3.8+
- PyTorch 2.0+
- PyBullet
- NumPy
- OpenCV

## License

BSD-3-Clause
