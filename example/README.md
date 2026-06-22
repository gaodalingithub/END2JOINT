# example 工具脚本

LeRobot 格式 `.parquet` 数据的 FK 计算工具集。

## compute_fk_batch.py — 基于 observation.state 的 FK 计算

读取原始 LeRobot 数据，用 `observation.state`（实际关节角）计算末端位姿。

```bash
conda activate actibot_sdk
pip install pandas pyarrow

python example/compute_fk_batch.py /path/to/data/folder -o /path/to/output
```

### 选项

| 参数 | 说明 |
|------|------|
| `-o <dir>` | 输出目录（默认: `{folder}_fk_results`） |
| `-f <fmt>` | `parquet`（默认）或 `csv` |
| `-e <id> [id...]` | 只处理指定编号 |
| `--merge` | 合并为一个文件 |

### 输出列（31 列）

| 类别 | 列数 | 列名 |
|------|------|------|
| 元信息 | 3 | episode_index, frame_index, timestamp |
| 左臂关节 | 7 | L_sh_pitch ~ L_wr_pitch |
| 右臂关节 | 7 | R_sh_pitch ~ R_wr_pitch |
| 夹爪 | 2 | gripper_L, gripper_R |
| 左末端位姿 | 6 | eeL_xyz + eeL_rpy |
| 右末端位姿 | 6 | eeR_xyz + eeR_rpy |

---

## compute_fk_action.py — 基于 action 的 FK 计算（推荐）

读取原始 LeRobot 数据，用 `action`（控制信号）计算末端位姿，同时保存 `observation.state` 作为 state 关节角。

**单文件包含训练 IK 网络所需的全部数据**，无需双源加载。

```bash
conda activate actibot_sdk
python example/compute_fk_action.py /path/to/data/folder -o /path/to/output
```

### 示例

```bash
# 批量处理数据集
python example/compute_fk_action.py data/0525_workflow_120

# 指定输出路径
python example/compute_fk_action.py data/0602_test_for_net -o data/0602_test_for_net_action_fk
```

### 输出列（45 列）

| 类别 | 列数 | 列名 | 用途 |
|------|------|------|------|
| 元信息 | 3 | episode_index, frame_index, timestamp | 帧标识 |
| **action 关节** | 14 | L_sh_pitch ~ R_wr_pitch | 训练目标 y |
| **state 关节** | 14 | state_L_sh_pitch ~ state_R_wr_pitch | 训练输入 prev |
| 夹爪 | 2 | gripper_L, gripper_R | 保存用 |
| **末端位姿** | 12 | eeL_xyzrpy + eeR_xyzrpy | 训练输入 ee_pose |

action 值来自原始数据的 `action` 列，state 值来自 `observation.state` 列，ee_pose 从 action 经 Pinocchio FK 计算得到。

---

## 数据格式

### 输入（LeRobot 格式）

```
observation.state  → float32[16]  左臂7 + 右臂7 + 夹爪2  (实际关节角)
action             → float32[16]  左臂7 + 右臂7 + 夹爪2  (控制信号)
timestamp          → float32      时间戳
episode_index      → int64        编号
frame_index        → int64        帧编号
```

### 坐标系

| 轴 | 方向 | 原点 |
|----|------|------|
| X | 向前 | 底座底面中心 |
| Y | 向左 | 底座底面中心 |
| Z | 向上 | 底座底面中心 |
