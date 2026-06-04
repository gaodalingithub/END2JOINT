# IK-EE2Joint: 从末端位姿预测关节角

用 MLP 从左右臂末端位姿直接预测双臂关节角度，不使用上一帧关节角作为输入。

## 问题定义

**输入** (12 维)：`[eeL_xyzrpy(6), eeR_xyzrpy(6)]`

**输出** (14 维)：`[joints_L(7), joints_R(7)]`

与 `ik_net` 的区别：不使用上一帧关节角作为输入，网络直接学习 `ee_pose → joints` 的映射。

## 网络架构

```
输入: 12D
  │
  ├── Linear(12→400) + ReLU + Dropout(0.1)
  ├── Linear(400→300) + ReLU + Dropout(0.1)
  ├── Linear(300→200) + ReLU + Dropout(0.1)
  ├── Linear(200→100) + ReLU + Dropout(0.1)
  ├── Linear(100→50) + ReLU + Dropout(0.1)
  │
  └── Linear(50→14) → 输出 14D
```

- 总参数量：~21 万
- 无 BatchNorm（仅 Linear + ReLU + Dropout）
- Dropout = 0.1

## 训练机制

### 停止条件（三选一）

1. **目标达标**：验证集关节角平均误差 < `target_joint_deg`（默认 1.0°）→ 成功停止
2. **早停**：关节角误差连续 `patience` 轮未下降 → 提前停止
3. **最大轮数**：达到 `num_epochs` → 强制停止

### 模型保存

按验证集关节角误差（度）最低保存最佳模型。

### 监控指标

每个 epoch 打印一行：

```
Epoch | Train Loss | Val Loss | Joint(°) | FK Pos(mm) | FK Ori(rad) | Time
```

| 列 | 含义 | 说明 |
|------|------|------|
| **Epoch** | 当前训练轮数 | 1 ~ num_epochs |
| **Train Loss** | 训练集 MSE | `mean((q_pred - q_gt)²)`，对所有关节×样本平均 |
| **Val Loss** | 验证集 MSE | 同上，验证集上计算，不参与梯度 |
| **Joint(°)** | 关节角平均误差（度） | `mean(\|q_pred - q_gt\|) × 180/π`，**主指标** |
| **FK Pos(mm)** | FK 重算末端位置误差 | 预测关节角经 Pinocchio 正运动学后与目标 ee 位置的偏差 |
| **FK Ori(rad)** | FK 重算末端姿态误差 | 同上，姿态角（roll/pitch/yaw）偏差 |
| **Time** | 该 epoch 耗时 | 秒 |

## 配置说明

```python
from config import hp, paths, data_config
```

| 分组 | 说明 |
|------|------|
| `hp` | 超参数，训练时经常调整（维度、架构、学习率、阈值等） |
| `paths` | 路径，与环境相关很少修改 |
| `data_config` | 数据列名，与 parquet 文件格式绑定 |

## 训练

```bash
conda activate actibot_sdk
python ik_ee2joint/train.py
```

结果保存到 `ik_ee2joint/results/`。

## 评估

```bash
python ik_ee2joint/evaluate.py
```

生成逐 episode 的关节角和末端轨迹对比图到 `results/vis/`。

## 推理速度

运行 `python ik_ee2joint/predict.py` 测试（GPU CUDA）：

| 批量大小 | 单帧耗时 | 帧/秒 |
|---------|---------|-------|
| 1 | **58.81 μs** | 17,005 |
| 64 | 1.01 μs (分摊) | 985,802 |
| 256 | 0.25 μs (分摊) | 3,976,047 |

单帧推理仅 **59 微秒**，远快于机器人控制周期（30Hz ≈ 33ms），完全满足实时需求。

## 结果

### 实际精度

在 120 个 episode（~6.5 万帧）数据上训练，不使用上一帧关节角，12D→14D 直接预测：

| 指标 | 验证集 | 全量测试集 |
|------|--------|-----------|
| 关节角 MAE | **1.09°** | **0.88°** |
| FK 位置误差 (L2) | — | **14.82 mm** |
| FK 姿态误差 | 0.026 rad | 0.056 rad |
| 最佳模型 | epoch 282 | epoch 282 |

### 各关节精度（测试集）

| 关节 | MAE (°) | 关节 | MAE (°) |
|------|---------|------|---------|
| L_sh_pitch | 0.90 | R_sh_pitch | 1.04 |
| L_sh_roll | 0.62 | R_sh_roll | 0.68 |
| L_sh_yaw | 0.51 | R_sh_yaw | 0.39 |
| L_el_pitch | 1.07 | R_el_pitch | 1.04 |
| L_el_roll | 1.07 | R_el_roll | 1.42 |
| L_wr_yaw | 0.40 | R_wr_yaw | 0.47 |
| L_wr_pitch | 1.40 | R_wr_pitch | 1.30 |


### 精度瓶颈

- 该架构不使用上一帧关节角，IK 的一对多性质导致精度上限 ~1°
- 如果要求更高精度，建议参考 `ik_net/` 改用残差连接

## 推理测试

```bash
# 完整测试集评估（精度 + 速度）
python ik_ee2joint/predict.py

# 只测推理速度，不评估精度
python ik_ee2joint/predict.py --time-only

# 只测试前 1000 帧
python ik_ee2joint/predict.py --count 1000
```
