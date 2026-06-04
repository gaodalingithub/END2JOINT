# IK-Net: 基于神经网络的机器人逆运动学求解

用 MLP 从末端执行器 (EE) 位姿 + 上一帧关节角映射到关节空间，左右臂联合训练，Pinocchio FK 一致性监控。

## 问题定义

**输入** (26 维)：
```
[eeL_xyzrpy(6), eeR_xyzrpy(6), joints_L_prev(7), joints_R_prev(7)]
```

**输出** (14 维)：
```
[joints_L(7), joints_R(7)]
```

**为什么需要 `joints_prev`？** IK 是一对多映射——同一末端位姿对应多组关节角解。提供上一帧关节角作为条件，网络学到"保持连续"的解，避免相邻帧之间跳变。

## 数据

来源：`data/0525_workflow_120_fk_results/`（120 个 episode 的 FK 计算结果，每帧 30 列）

每个样本构造：
- 帧 t 的 EE 位姿 → 输入前 12 维
- 帧 t-1 的关节角 → 输入后 14 维
- 帧 t 的关节角 → 输出 (14 维)
- 帧 0 使用自身关节角作为上一帧

数据划分：按 episode 随机拆分，**训练/验证/测试 = 80%/10%/10%**（96/12/12 episodes），避免帧间泄漏。

## 网络架构

```
输入: 26D
  │
  ├── Linear(26→400) + ReLU + Dropout(0.1)
  ├── Linear(400→300) + ReLU + Dropout(0.1)
  ├── Linear(300→200) + ReLU + Dropout(0.1)
  ├── Linear(200→100) + ReLU + Dropout(0.1)
  ├── Linear(100→50) + ReLU + Dropout(0.1)
  │
  └── Linear(50→14) → 输出 14D
        +
       prev_joints (可选残差连接)
  │
  ▼
输出: 14D
```

- 总参数量：~22 万
- 无 BatchNorm（仅 Linear + ReLU + Dropout）
- Dropout = 0.1

### 预测模式

| 模式 | 公式 | 说明 |
|------|------|------|
| 直接预测 | `q_t = MLP(input)` | 网络直接输出关节角 |
| 残差连接 (默认) | `q_t = q_{t-1} + MLP(input)` | 只预测增量，利用帧间连续性 |

通过 `config.py` 中 `hp["use_residual"]` 切换（默认 `True`）。

### 残差连接原理

核心思想：**不直接预测关节角，而是预测相邻帧之间的增量 delta。**

```
残差模式:                         直接模式:
输入: [ee_12, q_{t-1}]           输入: [ee_12, q_{t-1}]
         │                              │
         ▼                              ▼
      delta = MLP(x)                 q_t = MLP(x)
         │
    ┌────┘
    ▼
q_t = q_{t-1} + delta             q_t = MLP(x)
```

为什么有效？机器人以 30fps 采样，相邻帧关节角变化很小（通常 ±0.5° 以内）：

```
帧 t-1: q = (28.3°, -0.2°, 0.3°, 26.9°, ...)
帧 t:   q = (28.3°, -0.2°, 0.3°, 26.8°, ...)  → delta ≈ -0.1°
```

网络只需学习小量 delta（±0.5° 空间），而非完整的关节角（±180° 空间），学习难度降低数百倍：

```python
# model.py forward
out = self.net(x)          # 网络输出 delta
out = x[:, -14:] + out     # q_{t-1} + delta → 完整的 q_t
return out
```

训练时 `loss = MSE(q_{t-1} + delta_pred, q_true)`，梯度通过加法传回 delta。推理时根据当前 ee_pose + 上一帧关节角预测 delta 叠加，帧间自然平滑。

## 训练机制

### 停止条件（三选一）

1. **目标达标**：验证集关节角平均误差 < `target_joint_deg`（默认 0.5°）→ 成功停止
2. **早停**：关节角误差连续 `patience` 轮未下降 → 提前停止
3. **最大轮数**：达到 `num_epochs` → 强制停止

### 模型保存

按验证集关节角平均误差（度）最低保存最佳模型。

### 监控指标

每个 epoch 打印一行：

```
Epoch | Train Loss | Val Loss | Joint(°) | FK Pos(mm) | FK Ori(rad) | Time
```

| 列 | 含义 |
|------|------|
| **Epoch** | 当前训练轮数 |
| **Train Loss** | 训练集 MSE |
| **Val Loss** | 验证集 MSE |
| **Joint(°)** | 关节角平均误差（度），**主指标** |
| **FK Pos(mm)** | 预测关节角 FK 重算后的末端位置误差 |
| **FK Ori(rad)** | 预测关节角 FK 重算后的末端姿态误差 |
| **Time** | 该 epoch 耗时 |

## 配置说明

```python
from config import hp, paths, data_config
```

| 分组 | 说明 |
|------|------|
| `hp` | 超参数，训练时经常调整（维度、架构、学习率、阈值等） |
| `paths` | 路径，与环境相关很少修改 |
| `data_config` | 数据列名，与 parquet 文件格式绑定 |

## 安装与运行

### 环境要求

```bash
conda activate actibot_sdk
pip install torch scikit-learn matplotlib pandas pyarrow
```

### 训练

```bash
python ik_net/train.py
```

结果保存到 `ik_net/results/`。

### 评估

```bash
python ik_net/evaluate.py
```

生成逐 episode 的关节角和末端轨迹对比图到 `results/vis/`。

## 项目结构

```
ik_net/
├── config.py           # 超参数配置
├── dataloader.py       # 数据加载、归一化、样本构造
├── model.py            # MLP 模型定义
├── train.py            # 训练循环 + FK 一致性验证
├── evaluate.py         # 测试评估 + 可视化
├── fk_utils.py         # Pinocchio FK 封装
├── requirements.txt
├── README.md
└── results/
    ├── best_model.pt       # 最佳模型权重
    ├── scaler.pkl          # 归一化参数
    ├── history.json        # 训练历史
    ├── loss_curve.png      # 损失曲线图
    ├── evaluation_summary.json
    └── vis/                # 可视化结果
        ├── ep{xxx}_joints_L.png   # 左臂关节角对比
        ├── ep{xxx}_joints_R.png   # 右臂关节角对比
        ├── ep{xxx}_ee_L.png       # 左末端位置对比
        └── ep{xxx}_ee_R.png       # 右末端位置对比
```

## FK 一致性监控

训练过程中每个 epoch 结束后，用 Pinocchio FK 重算预测关节角的末端位姿，与目标 EE 位姿比较：

```
FK 位置误差 = |FK(q_pred).pos - ee_target.pos|
FK 姿态误差 = |FK(q_pred).rpy - ee_target.rpy|
```

该指标仅用于**监控**（打印输出、选模型参考），不参与梯度反向传播。

## 推理速度

运行 `python ik_net/predict.py` 测试（GPU CUDA）：

| 批量大小 | 单帧耗时 | 帧/秒 |
|---------|---------|-------|
| 1 | **64.47 μs** | 15,512 |
| 64 | 1.09 μs | 920,733 |
| 256 | 0.27 μs | 3,646,304 |

## 结果

### 实际精度

在 120 个 episode（~6.5 万帧）数据上训练，26D→14D：

| 指标 | 残差连接（测试集） | 直接预测（测试集） |
|------|-----------------|-----------------|
| 关节角 MAE | **0.157°** (2.75 mrad) | 1.004° (17.53 mrad) |
| 关节角 RMSE | 0.270° | 1.394° |
| FK 位置误差 | **1.96 mm** | 13.92 mm |
| FK 姿态误差 | **7.71 mrad** | 52.37 mrad |
| R² | **0.9998** | 0.9944 |
| 最佳模型 | epoch 269 | epoch 364 |

残差连接模式下精度高 **6 倍**，因为网络只需学习帧间微小的增量变化。

### 各关节精度（残差连接，测试集）

| 关节 | MAE (mrad) | 关节 | MAE (mrad) |
|------|-----------|------|-----------|
| L_sh_pitch | 2.95 | R_sh_pitch | 2.48 |
| L_sh_roll | 1.92 | R_sh_roll | 1.58 |
| L_sh_yaw | 2.13 | R_sh_yaw | 1.59 |
| L_el_pitch | 4.29 | R_el_pitch | 3.25 |
| L_el_roll | 3.69 | R_el_roll | 3.18 |
| L_wr_yaw | 2.38 | R_wr_yaw | 2.18 |
| L_wr_pitch | 3.99 | R_wr_pitch | 2.82 |

### 各关节精度（直接预测，测试集）

| 关节 | MAE (mrad) | 关节 | MAE (mrad) |
|------|-----------|------|-----------|
| L_sh_pitch | 17.65 | R_sh_pitch | 20.50 |
| L_sh_roll | 12.12 | R_sh_roll | 14.02 |
| L_sh_yaw | 10.75 | R_sh_yaw | 8.29 |
| L_el_pitch | 21.08 | R_el_pitch | 20.62 |
| L_el_roll | 19.00 | R_el_roll | 26.94 |
| L_wr_yaw | 8.58 | R_wr_yaw | 10.16 |
| L_wr_pitch | 28.34 | R_wr_pitch | 27.31 |

## 推理测试

```bash
# 完整测试集评估（精度 + 速度）
python ik_net/predict.py

# 只测推理速度
python ik_net/predict.py --time-only

# 只测试前 1000 帧
python ik_net/predict.py --count 1000
```

## 与 ik_ee2joint 对比

| 对比项 | ik_net | ik_ee2joint |
|--------|--------|-------------|
| 输入维度 | 26D（ee_12 + prev_14） | 12D（仅 ee_12） |
| 输出维度 | 14D（L_7 + R_7） | 14D（L_7 + R_7） |
| 残差连接 | 支持（默认启用） | 不支持 |
| 参数量 | ~22 万 | ~21 万 |
| 架构 | 400→300→200→100→50 | 400→300→200→100→50 |
