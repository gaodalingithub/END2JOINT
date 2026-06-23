# End2Joint — 基于神经网络的机器人逆运动学求解

从末端位姿 + 实际关节角映射到 control 信号（action），左右臂联合训练。

## 项目结构

```
End2Joint/
├── actibot_sdk/          # 机器人 SDK（ROS2 控制、URDF 模型、配置文件）
├── data/                 # 数据集
│   ├── 0525_workflow_120/           # 120 episodes 训练数据（LeRobot 格式）
│   ├── 0525_workflow_120_action_fk/ # 训练集 FK 结果（45列，含 ee + state + action）
│   ├── 0602_test_for_net/           # 测试数据（5 episodes）
│   └── 0602_test_for_net_action_fk/ # 测试集 FK 结果
├── example/              # FK 计算工具
│   ├── actibot_fk.py                 # FK 核心引擎（Pinocchio）
│   ├── compute_fk_batch.py           # 基于 observation.state 的 FK
│   ├── compute_fk_action.py          # 基于 action 的 FK（推荐，输出 45 列）
│   └── README.md
├── docs/                 # 文档
├── ik_net/               # 标准 IK 网络
│   ├── config.py                     # 超参数配置
│   ├── dataloader.py                 # 数据加载（单文件 ee + state + action）
│   ├── model.py                      # ResidualMLP 模型
│   ├── train.py                      # 训练循环
│   ├── evaluate.py                   # 测试评估
│   ├── predict.py                    # 推理 + 保存结果
│   ├── predict_ar.py                 # 自回归推理
│   ├── analyze_real_error.py         # 真机验证数据分析
│   ├── playback_predictions.py       # ROS2 回放
│   └── save_real/                    # 真机验证日志
├── ik_net_robust/        # 改进版：噪声注入 + FK 修正
│   ├── config.py                     # 超参数配置
│   ├── dataloader.py                 # 数据加载 + 噪声注入
│   ├── model.py                      # ResidualMLP 模型
│   ├── train.py                      # 训练循环
│   ├── evaluate.py                   # 测试评估
│   ├── fk_utils.py                   # FK 修正函数
│   ├── predict.py                    # 自回归推理 + FK 修正
│   └── test_ar.py                    # 自回归精度测试
├── server/               # 真机验证服务端
│   ├── ik_validation_server.py       # IK-Net 真机闭环验证服务
│   ├── ik_validation_fk_server.py    # 带 FK 修正的验证服务
│   ├── replay_client.py              # 回放客户端
│   ├── replay_server.py              # 回放服务端
│   └── replay_parquet.py             # parquet 回放工具
```

## 模型

| 模型 | 输入 | 输出 | 核心特点 |
|------|------|------|---------|
| **ik_net** | 26D ee_pose + state_prev | **14D action** | 单源数据，直接预测 |
| **ik_net_robust** | 26D ee_pose + state_prev | 14D action | 噪声注入 + FK 修正 |

## 数据格式

`compute_fk_action.py` 输出（45 列，单文件全部数据）：

| 类别 | 列数 | 列名 | 用途 |
|------|------|------|------|
| 元信息 | 3 | episode_index, frame_index, timestamp | 帧标识 |
| **action 关节** | 14 | L_sh_pitch ~ R_wr_pitch | 训练目标 y |
| **state 关节** | 14 | state_L_sh_pitch ~ state_R_wr_pitch | 输入 prev |
| 夹爪 | 2 | gripper_L, gripper_R | 保存用 |
| **末端位姿** | 12 | eeL_xyzrpy + eeR_xyzrpy | 输入 ee_pose |

训练样本：`X = [ee_pose_t(12), state_{t-1}(14)] → 26D`, `y = action_t → 14D`

## 环境配置

```bash
conda create -n actibot_sdk python=3.12 -y
conda activate actibot_sdk
conda install pinocchio=3.6.0 -c conda-forge -y
pip install torch scikit-learn matplotlib pandas pyarrow
```

## 快速开始

```bash
# 1. 计算 FK
python example/compute_fk_action.py data/0602_test_for_net -o data/0602_test_for_net_action_fk

# 2. 训练
python ik_net/train.py

# 3. 推理
python ik_net/predict.py --data data/0602_test_for_net_action_fk
python ik_net/predict_ar.py --data data/0602_test_for_net_action_fk

# 4. FK 修正
python ik_net_robust/predict.py --fkfix --fkfix-step 5

# 5. 真机验证分析
python ik_net/analyze_real_error.py
```

## 精度

| 模式 | Joint MAE | 单帧耗时 |
|------|-----------|---------|
| ik_net（离线单步） | ~1.0° | 60 μs |
| ik_net（自回归） | ~5.8° | 60 μs |
| ik_net_robust +噪声 | ~2.9° | 60 μs |
| **+FK修正(每5帧)** | **~0.6°** | **92 μs** |
| **+FK修正(每帧)** | **~0.36°** | 170 μs |
| 真机 pred vs real | **~0.6°** | — |


## 真机验证服务端 (`server/ik_validation_server.py`)

该服务端连接两个世界：**录好的轨迹数据**（parquet）和 **真实的机器人**（ROS2 + ZMQ），输出控制信号和验证日志。

### 启动方式

```bash
# 启动服务端
python server/ik_validation_server.py \
    --parquet data/0602_test_for_net_action_fk/episode_000000_action_fk.parquet \
    --inference-delay 0.05

# 机器人端连接
python main_actibot.py \
    --policy-host 192.168.0.223 \
    --policy-port 5555 \
    --instruction "replay" \
    --open-loop-horizon 1
```

### 输入输出总览

| 序号 | 输入 | 来源 | 维度 | 用途 |
|------|------|------|------|------|
| **①** | `ee_pose_t` | parquet 文件 eeL/eeR 列 | 12D | IK-Net 的目标末端位姿 |
| **②** | `state_{t-1}` | 机器人传感器实时反馈 | 14D | IK-Net 的上一帧真实关节角 |
| **③** | `gripper` | parquet 文件 | 2D | 夹爪控制（IK-Net 不预测） |
| **④** | 模型权重 | 磁盘 `.pt` + `scaler.pkl` | 26D→14D MLP | 推理引擎 |
| **⑤** | 请求端点 | ZMQ 网络（来自 main_actibot.py） | — | get_action / ping / reset |

| 序号 | 输出 | 去向 | 维度 | 内容 |
|------|------|------|------|------|
| **❶** | `pred_action` | 机器人（ZMQ 实时） | 16D | 14 关节 + 2 夹爪控制信号 |
| **❷** | `validation_log` | 磁盘（CSV + parquet） | 多列 | real vs pred 对比 + 输入回溯 |

### 输入详解

#### ① `ee_pose_t` — 目标末端位姿（parquet）

从 `--parquet` 指定的 parquet 文件加载，由 `compute_fk_action.py` 预先计算：

```
action (录制的控制信号) → Pinocchio FK → ee_pose (12D)
```

列名：`eeL_x, eeL_y, eeL_z, eeL_roll, eeL_pitch, eeL_yaw, eeR_x, eeR_y, eeR_z, eeR_roll, eeR_pitch, eeR_yaw`

#### ② `state_{t-1}` — 上一帧真实关节角（传感器实时反馈）

`main_actibot.py` 的 `ROS2ActibotEnv` 节点订阅 ROS2 话题 `/actibot_arm_state`（`sensor_msgs/JointState`），从 `msg.position[:16]` 获取 16D 关节状态（左臂7 + 右臂7 + 夹爪2），然后通过 ZMQ 发送给服务端：

```python
# main_actibot.py _build_request_data()
"state": {
    "left_arm_joint_positions":  left_arm[None, None, ...],   # shape (1,1,7)
    "right_arm_joint_positions": right_arm[None, None, ...],  # shape (1,1,7)
    "left_gripper_position":     left_grip[None, None, ...],  # shape (1,1,1)
    "right_gripper_position":    right_grip[None, None, ...], # shape (1,1,1)
}
```

服务端从 ZMQ 请求中提取并展平：
```
dict {'nd':3, 'type':'float32', 'shape':[1,1,7], 'data':b'...'}
  → np.frombuffer().reshape().squeeze() → (7,) float64
```

#### ③ `gripper` — 夹爪（parquet）

直接从 parquet 原数据拼接，不经过 IK-Net：
```python
gripper = [df["gripper_L"], df["gripper_R"]]  # 2D
```

### 输出详解

#### ❶ `pred_action` — 控制信号（ZMQ 实时）

**构造过程**（`make_response`）：

```
Frame 0: [parquet state_* 列(14D), parquet gripper(2D)] → 16D → 机器人
          （初始对齐，不用模型预测）

Frame 1+: ee_pose(12D) + prev_14(14D) → 26D → IK-Net → pred_14(14D)
           + gripper(2D) → 16D → 机器人执行
```

**响应格式**（`_pack_action`）：
```python
{
    "left_arm_target_joint_positions":  [[pred_L0..L6]],     # 左臂 7 关节
    "right_arm_target_joint_positions": [[pred_R0..R6]],     # 右臂 7 关节
    "left_target_gripper_position":     [[gripper_L]],
    "right_target_gripper_position":    [[gripper_R]],
}
```

客户端 `_decode_action_chunk` 接收后，将 16D 控制信号发布到 ROS2 话题 `/actibot_arm_ctrl` 驱动机器人。

#### ❷ validation_log — 验证日志（磁盘）

保存在 `--output-dir`（默认 `ik_net/save_real/`），每帧记录：

| 列类别 | 列名 | 含义 |
|--------|------|------|
| 元信息 | frame, is_init, timestamp | 帧标识 |
| **预测值** | pred_action_* (16 列) | 模型输出 → 发给机器人的控制信号 |
| **真实值** | real_joint_state_* (16 列) | 机器人传感器测得的实际关节位置 |
| **模型输入** | input_ee_* (12 列) | 目标末端位姿 |
| **模型输入** | input_prev_joint_state_* (14 列) | 上一帧真实关节角（模型输入） |

保存时自动输出统计：
```
[Stats] pred_action vs real_joint_state (N 帧):
  MAE:  0.632 deg
  RMSE: 0.891 deg
  L_sh_pitch: 0.4321 deg
  ...
```

### 控制回路完整数据流

```
                    ┌─────────────────┐
                    │  parquet 轨迹数据 │
                    │ (ee_pose+gripper)│
                    └────────┬────────┘
                             │ ee_pose(12D), gripper(2D)
                             ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ 机器人传感器   │────▶│IKValidationServer│────▶│ 机器人控制器      │
│ (关节位置反馈) │     │                  │     │ (执行动作)        │
└──────────────┘     │  控制回路:        │     └──────────────────┘
   prev_14(14D) ◀────│  1. 接收观测      │────▶ pred_16(16D)
                     │  2. IK-Net 推理   │
                     │  3. 返回控制信号   │
                     └────────┬─────────┘
                              │ 每帧记录
                              ▼
                     ┌──────────────────┐
                     │ validation_log   │
                     │ pred vs real     │
                     └──────────────────┘
```

### 降级与安全机制（无实时关节反馈时）

当服务端无法从机器人观测中提取实时关节状态时，按三级降级逻辑处理：

```
提取成功 → 更新 last_valid_joints，重置计数器
                ↓
提取失败 → 计数器 +1
                ↓
     ┌──────────────────────────────┐
     │ 条件                           │ 行为
     ├──────────────────────────────┤
     │ ≤30帧连续失败，有历史数据      │ 使用 last_valid_joints 继续推理（短时降级）
     │ >30帧连续失败，有历史数据      │ 安全停止：发送 hold 指令，关闭服务
     │ 0帧成功，从未获取过            │ 使用硬编码 fallback_position（仅 Frame 0 对齐）
     └──────────────────────────────┘
```

#### 短时降级（≤30帧失败）

使用 `last_valid_joints`（上次成功提取的 16D 关节）作为 `prev_14`：

```python
real_joints = self.last_valid_joints.copy()  # 不是固定 fallback_position
```

**为什么比固定 fallback 安全**：
- `last_valid_joints` 是机器人不久前真实的位置（毫秒级延迟）
- 模型输入接近实际分布，预测的动作不会突变
- 30帧 ≈ 1秒（@30Hz），机器人在这期间关节位移有限

#### 安全停止（>30帧失败）

当连续 30+ 帧都无法获取到关节状态，说明通信严重异常。此时不再继续推理，直接向机器人发送 **hold 命令**（保持最后一帧有效关节位置）并关闭服务：

```python
hold_action = self._pack_action(self.last_valid_joints)
sock.send([hold_action, {"status": "done"}])
running = False
```

#### 日志标识

| 日志 | 含义 |
|------|------|
| `[Warning] 关节提取失败(3), 使用上次有效关节` | 短时降级中，括号内为连续失败次数 |
| `[FATAL] 关节提取持续失败 35 帧, 安全停止` | 超过阈值，触发安全停止 |
| `[Warning] 从未获取到有效关节, 使用初始位置` | 启动后从未收到过有效数据（常见于机器人未连接时测试） |


### 与 `ik_validation_fk_server.py` 的区别

| 特性 | `ik_validation_server.py` | `ik_validation_fk_server.py` |
|------|--------------------------|------------------------------|
| FK 修正 | ❌ 无 | ✅ Pinocchio Jacobian 阻尼伪逆修正 |
| `--fkfix-step` | ❌ | ✅ 0=关闭, 1=每帧, 5=每5帧 |
| 核心逻辑 | MLP 直接预测 | MLP 预测 + FK 后处理修正 |
