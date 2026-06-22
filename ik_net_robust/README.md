# IK-Net-Robust: 抗自回归误差累积的改进版 IK

基于 `ik_net`，加入噪声注入训练和 FK 引导修正，解决自回归推理时的误差累积问题。

## 问题

实际部署时上一帧关节角只能使用模型自己的上一步预测值，导致分布偏移：

```
帧 0: 用真实 state_0             → 分布内
帧 1: 用模型预测的 action_0      → 偏离训练分布
帧 2: 用模型预测的 action_1      → 误差累积
```

自回归 Joint MAE 从 ~1° 膨胀到 **~6°**。

## 改进方法

### 方法一：噪声注入训练

训练时对 `state_prev` 添加高斯噪声（±2.9°），让模型学会在输入不精确时仍能正确输出 action。

**效果**：自回归误差从 ~6° 降到 **~3.7°**（↓38%）。

### 方法二：FK 引导修正（后处理）

推理时用 Pinocchio Jacobian 修正 MLP 的预测：

```
MLP → q_pred → FK(q_pred) → ee_pred
                               ↓
                    与 ee_target 比较 → 6D 误差
                               ↓
                    阻尼伪逆 J⁺ → Δq
                               ↓
                    q_corrected = q_pred + Δq
```

**效果**：自回归误差从 ~6° 骤降到 **~0.36°**（↓94%）。

### 精度对比

| 方法 | 自回归 Joint MAE | 单帧耗时 |
|------|-----------------|---------|
| 无改进 | ~6° | 60 μs |
| 噪声注入 | ~3.7° | 60 μs |
| **FK 修正(每5帧)** | **~0.6°** | **92 μs** |
| **FK 修正(每帧)** | **~0.36°** | 170 μs |

## 数据格式

与 `ik_net` 一致，单文件包含全部所需数据（由 `compute_fk_action.py` 生成）：

```
eeL/R_xyzrpy      → 输入 ee_pose_t (12D)
state_*           → 输入 state_{t-1} (14D)
L/R_sh_pitch...   → 目标 action_t (14D)
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.py` | 超参数配置 |
| `dataloader.py` | 数据加载 + 噪声注入 |
| `model.py` | ResidualMLP 模型 |
| `train.py` | 训练循环 |
| `evaluate.py` | 测试评估 |
| `fk_utils.py` | Pinocchio FK + FK 修正函数 |
| `predict.py` | 自回归推理 + FK 修正 + 保存结果 |
| `test_ar.py` | 自回归精度测试 |

## 用法

```bash
# 训练
python ik_net_robust/train.py

# 自回归推理 + 精度评估
python ik_net_robust/predict.py

# 启用 FK 修正（每帧）
python ik_net_robust/predict.py --fkfix

# FK 修正（每5帧）
python ik_net_robust/predict.py --fkfix --fkfix-step 5

# 保存预测结果
python ik_net_robust/predict.py --data /path/to/data --save ./results

# 自回归精度测试
python ik_net_robust/test_ar.py
python ik_net_robust/test_ar.py --fkfix
```
