# End2Joint — 基于神经网络的机器人逆运动学求解

从末端执行器 (EE) 位姿映射到关节空间的多方案 IK 框架，支持单步预测、自回归推理、FK 引导修正。

## 项目结构

```
End2Joint/
├── actibot_sdk/          # 机器人 SDK（ROS2 控制、URDF 模型、配置文件）
├── data/                 # 数据集
│   ├── 0525_workflow_120/           # 120 个 episode 训练数据（LeRobot 格式）
│   ├── 0525_workflow_120_fk_results/ # FK 计算结果
│   ├── 0602_test_for_net/           # 测试数据（5 episodes）
│   ├── 0602_test_for_net_fk_results/ # state-based FK 结果
│   ├── 0602_test_for_net_action_fk/  # action-based FK 结果
│   └── my_dataset_groot_fk_results/  # 外部数据集 FK 结果
├── example/              # 工具脚本
│   ├── actibot_fk.py                 # FK 核心引擎（Pinocchio）
│   ├── compute_fk_batch.py           # 批量计算 state 的 FK
│   ├── compute_fk_action.py          # 批量计算 action 的 FK
│   ├── analyze_fk_batch.py           # FK 结果分析
│   └── visualize_fk.py               # Meshcat 可视化
├── docs/                 # 文档
├── ik_ee2joint/          # 简化版 IK：12D ee_pose → 14D joints（无时序信息）
├── ik_net/               # 标准版 IK：26D ee_pose+prev_joints → 14D joints
│   ├── config.py                     # 超参数配置（唯一需修改的文件）
│   ├── model.py                      # ResidualMLP 模型
│   ├── train.py                      # 训练循环
│   ├── evaluate.py                   # 测试评估 + 可视化
│   ├── predict.py                    # 推理测试（单步预测）
│   ├── predict_ar.py                 # 自回归推理（模拟部署）
│   ├── playback_predictions.py       # ROS2 回放预测结果到机器人
│   └── visualize_predictions.py      # 仿真可视化预测结果
├── ik_net_robust/        # 改进版：噪声注入 + FK 修正抗自回归累积
│   ├── config.py                     # 超参数配置
│   ├── dataloader.py                 # 数据加载 + 噪声注入
│   ├── model.py                      # ResidualMLP 模型
│   ├── train.py                      # 训练循环
│   ├── evaluate.py                   # 测试评估
│   ├── fk_utils.py                   # FK 修正函数
│   ├── predict.py                    # 推理测试（含 FK 修正选项）
│   ├── test_ar.py                    # 自回归精度测试
│   └── README.md                     # 改进方法说明
```

## 模型对比

| 模型 | 输入 | 输出 | 核心特点 | 关节角精度 | FK 位置精度 |
|------|------|------|---------|-----------|-----------|
| **ik_ee2joint** | 12D ee_pose | 14D joints | 无时序信息，直接映射 | ~1.0° | ~14mm |
| **ik_net** | 26D ee_pose + prev_joints | 14D joints | **残差连接**，利用帧间连续性 | **0.16°** | **2.6mm** |
| **ik_net_robust** | 26D ee_pose + prev_joints | 14D joints | 噪声注入 + FK 修正抗自回归漂移 | 0.16°（FK修正后 0.36°） | 1.4mm |

## 工作原理

### 正向运动学 (FK)

使用 Pinocchio 库加载 URDF 模型，计算给定关节角的末端位姿（4×4 齐次变换矩阵）：

```python
q → pin.forwardKinematics(model, data, q) → 4×4 矩阵 → 位置(xyz) + 姿态(rpy)
```

### 逆运动学 (IK) 网络

**残差连接**是核心——网络不直接预测关节角，而是预测相邻帧之间的增量：

```
q_t = q_{t-1} + MLP(ee_t, q_{t-1})
```

30fps 采样下相邻帧关节角变化通常 < 0.5°，学习小量比学习完整关节角（±180°）容易数百倍。

### FK 引导修正

在推理时用 Pinocchio 物理模型修正 MLP 的预测：

```
MLP → q_pred → FK(q_pred) → ee_pred
         ↓                       ↓
    Jacobian 阻尼伪逆 ←── 与 ee_target 比较
         ↓
    q_corrected = q_pred + Δq
```

修正仅需 ~110μs/帧，可将自回归误差从 5.88° 降到 0.36°。

## 数据格式

### 输入数据 (LeRobot 格式)

```
observation.state  → float32[16]  左臂7 + 右臂7 + 夹爪2
action             → float32[16]  控制信号
timestamp          → float32      时间戳
episode_index      → int64        episode 编号
frame_index        → int64        帧编号
```

### FK 结果格式

```
joints_14:  L_sh_pitch ~ R_wr_pitch  (rad)
ee_12:      eeL_xyzrpy + eeR_xyzrpy  (m / rad)
gripper_2:  gripper_L, gripper_R
```

## 环境配置

```bash
conda create -n actibot_sdk python=3.12 -y
conda activate actibot_sdk
conda install pinocchio=3.6.0 -c conda-forge -y
pip install torch scikit-learn matplotlib pandas pyarrow
```

## 快速开始

### 1. 计算 FK 结果

```bash
# 用 observation.state 计算
python example/compute_fk_batch.py data/0602_test_for_net

# 用 action 控制信号计算
python example/compute_fk_action.py data/0602_test_for_net
```

### 2. 训练 IK 网络

```bash
# 训练标准版（残差连接）
python ik_net/train.py

# 训练改进版（噪声注入）
python ik_net_robust/train.py
```

### 3. 推理测试

```bash
# 单步预测
python ik_net/predict.py --data /path/to/fk_results

# 自回归推理（模拟实际部署）
python ik_net/predict_ar.py --data /path/to/data --fkfix --save ./results

# 自回归精度测试
python ik_net_robust/test_ar.py --fkfix
```

### 4. 回放到机器人

```bash
python ik_net/playback_predictions.py path/to/predictions.parquet
```

## 精度结果

| 模式 | 关节角 MAE | FK 位置误差 | FK 姿态误差 | 推理速度 |
|------|-----------|-----------|-----------|---------|
| ik_net 单步预测 | 0.22° | 2.6 mm | 8.9 mrad | 64 μs |
| ik_net 自回归 | 5.88° | 49.8 mm | — | 64 μs |
| ik_net_robust 自回归(+噪声) | 3.74° | 24.0 mm | — | 64 μs |
| ik_net_robust +FK修正 | **0.36°** | **0.04 mm** | **0.03 mrad** | **170 μs** |
