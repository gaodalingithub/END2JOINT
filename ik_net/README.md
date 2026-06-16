# IK-Net: 基于神经网络的机器人逆运动学求解

用 MLP 从末端位姿 + 上一帧控制信号映射到关节空间，预测 **control 信号的增量**，左右臂联合训练。

## 问题定义

**输入** (26 维)：
```
[eeL_xyzrpy(6), eeR_xyzrpy(6), action_L_prev(7), action_R_prev(7)]
```

**输出** (14 维)：control 信号增量
```
[action_L(7), action_R(7)] - [action_L_prev(7), action_R_prev(7)]
```

模型通过残差连接还原完整控制信号：
```
action_t_pred = action_{t-1} + delta_pred
loss = MSE(action_t_pred, action_t)
```

### 数据来源

只使用 **action FK** 结果（通过 `compute_fk_action.py` 生成），无需 state FK。

| 输入部分 | 来源 | 说明 |
|---------|------|------|
| `ee_pose_t` | action FK | 从 action 算的末端位姿 |
| `action_{t-1}` | action FK | 上一帧 control 信号 |
| `y = delta` | action FK | action_t - action_{t-1} |

## 数据

来源（最多 2 个数据源）：
- `paths["data_dir"]` — action FK 结果（主数据）
- `paths["extra_data_dirs"]` — 额外数据目录列表

每个样本构造：
- 帧 t 的 EE 位姿 → 输入前 12 维
- 帧 t-1 的 action → 输入后 14 维
- 帧 t 与 t-1 的 action 差值 → 输出 (14 维)

数据划分：按 episode 随机拆分，**训练/验证/测试 = 80%/10%/10%**。

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
  └── Linear(50→14) + action_{t-1} (残差连接)
  │
  ▼
输出: 14D (action_t)
```

- 总参数量：~22 万
- 残差连接：`action_t = action_{t-1} + delta`（仅预测控制信号增量）

## 安装与运行

```bash
conda activate actibot_sdk
pip install torch scikit-learn matplotlib pandas pyarrow
```

### 训练

```bash
python ik_net/train.py
```

### 推理测试

```bash
# 单步预测（使用真实 action_{t-1}）
python ik_net/predict.py --data data/0602_test_for_net_action_fk

# 自回归推理（模拟部署）
python ik_net/predict_ar.py --data data/0602_test_for_net_action_fk

# 保存预测结果
python ik_net/predict_ar.py --data /path/to/data --save ./results
```

### 回放到机器人

```bash
python ik_net/playback_predictions.py ./results/predictions.parquet
```

## 项目结构

```
ik_net/
├── config.py              # 超参数配置
├── dataloader.py          # 数据加载、样本构造
├── model.py               # ResidualMLP 模型
├── train.py               # 训练循环
├── evaluate.py            # 测试评估
├── predict.py             # 单步推理测试
├── predict_ar.py          # 自回归推理测试
├── predict_new_data.py    # 新数据集预测
├── playback_predictions.py # ROS2 回放
├── fk_utils.py            # Pinocchio FK
├── requirements.txt
└── README.md
```

## 关键配置

| 参数 | 作用 |
|------|------|
| `hp["ckpt_dir"]` | 模型权重路径 |
| `hp["ckpt_name"]` | 模型文件名 |
| `hp["use_residual"]` | 残差连接开关 |
| `paths["data_dir"]` | action FK 训练数据 |
| `paths["extra_data_dirs"]` | 额外训练数据 |
