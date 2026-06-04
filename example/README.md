# FK 批量分析工具

批量读取 LeRobot 格式的 `.parquet` 数据，利用 Pinocchio 正运动学 (FK) 计算双臂末端位姿，自动归类动作类型。

## analyze_fk_batch.py — 动作分析

分析每段 episode 的末端位移、姿态变化，自动归类动作类型（向前/向左/向右/静止等）。

### 依赖

```bash
conda activate actibot_sdk
```

需要 `pandas`、`pyarrow`（读取 parquet）：

```bash
pip install pandas pyarrow
```

### 用法

```bash
python example/analyze_fk_batch.py [data_folder] [选项]
```

```bash
python example/analyze_fk_batch.py /home/ubuntu/code/data_archive/data/lerobot_groot_data/0526_test_for_fk/
```

#### 基本用法

```bash
# 分析默认文件夹下所有 episode
python example/analyze_fk_batch.py

# 指定文件夹
python example/analyze_fk_batch.py /path/to/data/folder
```

#### 选项

| 参数 | 说明 |
|------|------|
| `folder` | 数据文件夹路径（包含 `episode_*.parquet`），默认 `0526_test_for_fk` |
| `-o <file>` | 输出 JSON 结果到文件 |
| `-e <id> [id...]` | 只分析指定编号的 episode，如 `-e 0 2 5` |
| `-v` | 详细模式，显示每个关节变化量 |
| `--info` | 仅列出 episode 基本信息（文件数、帧数、时长），不做 FK 计算 |

#### 示例

```bash
# 查看有哪些数据
python example/analyze_fk_batch.py --info

# 分析全部 15 个 episode
python example/analyze_fk_batch.py

# 只分析第 0、1、2 号 episode
python example/analyze_fk_batch.py -e 0 1 2

# 输出 JSON
python example/analyze_fk_batch.py -o results.json

# 详细输出（含关节角度变化）
python example/analyze_fk_batch.py -v
```

---

## compute_fk_batch.py — 批量 FK 计算

批量读取 `.parquet` 数据，逐帧计算左右臂末端位姿，**保存完整结果**（含原始关节角 + 末端位姿），适合下游分析或训练数据准备。

### 计算依赖

```bash
conda activate actibot_sdk
pip install pandas pyarrow
```

### 计算用法

```bash
python example/compute_fk_batch.py [data_folder] [选项]
```

```bash
python example/compute_fk_batch.py /home/ubuntu/code/End2Joint/data/0525_workflow_120
```

#### 计算选项

| 参数 | 说明 |
|------|------|
| `folder` | 数据文件夹路径（含 `episode_*.parquet`） |
| `-o <dir>` | 输出文件夹（默认: 输入文件夹同级 `{folder}_fk_results`） |
| `-f <fmt>` | 输出格式，`parquet`（默认）或 `csv` |
| `-e <id> [id...]` | 只处理指定编号，如 `-e 0 2 5` |
| `--merge` | 合并所有 episode 为一个文件 |

#### 计算示例

```bash
# 基本用法，结果存到默认输出文件夹
python example/compute_fk_batch.py /path/to/data/folder

# 指定输出目录、CSV 格式
python example/compute_fk_batch.py data/0525_workflow_120 -o results/fk_output -f csv

# 只处理第 0、2、5 号 episode
python example/compute_fk_batch.py data/0525_workflow_120 -e 0 2 5

# 合并所有 episode 为一个文件
python example/compute_fk_batch.py data/0525_workflow_120 --merge
```

### 输出

每个 episode 独立保存，按原始编号命名：

```
{output_dir}/
  ├── episode_000000_fk.parquet
  ├── episode_000001_fk.parquet
  └── ...
```

使用 `--merge` 时额外生成 `all_episodes_fk.parquet`。

#### 输出列（30 列）

| 类别 | 列名 | 说明 |
|------|------|------|
| 元信息 | `episode_index`, `frame_index`, `timestamp` | 来源帧标识 |
| 左臂关节 ×7 | `L_sh_pitch`, `L_sh_roll`, `L_sh_yaw`, `L_el_pitch`, `L_el_roll`, `L_wr_yaw`, `L_wr_pitch` | 弧度 (rad) |
| 右臂关节 ×7 | `R_sh_pitch`, `R_sh_roll`, `R_sh_yaw`, `R_el_pitch`, `R_el_roll`, `R_wr_yaw`, `R_wr_pitch` | 弧度 (rad) |
| 夹爪 ×2 | `gripper_L`, `gripper_R` | 开合度 |
| 左末端位置 ×3 | `eeL_x`, `eeL_y`, `eeL_z` | 米 (m) |
| 左末端姿态 ×3 | `eeL_roll`, `eeL_pitch`, `eeL_yaw` | 弧度 (rad) |
| 右末端位置 ×3 | `eeR_x`, `eeR_y`, `eeR_z` | 米 (m) |
| 右末端姿态 ×3 | `eeR_roll`, `eeR_pitch`, `eeR_yaw` | 弧度 (rad) |

---

## 输出说明（analyze_fk_batch.py）

### 每个 episode 输出

```
============================================================
  Episode   0  |  150 帧  |  5.00 s  |  右臂x(前)运动 → 趋于零位
============================================================
  左臂:
    位置: start (0.3197, 0.2275, 0.7476)  end (0.3292, 0.2377, 0.7496)
           delta (+0.0095, +0.0102, +0.0020)  range (0.0130, 0.0129, 0.0078)  路程 0.0463 m
    姿态: start (-0.9, 0.6, -1.2)  end (-2.1, 2.4, 0.3) deg
           delta (-1.2, +1.8, +1.4)  range (2.3, 2.8, 1.8) deg
  右臂:
    位置: start (0.3196, -0.2260, 0.7536)  end (0.4556, -0.1991, 0.7581)
           delta (+0.1359, +0.0269, +0.0045)  range (0.1390, 0.0286, 0.0231)  路程 0.1827 m
    姿态: start (0.0, -0.5, 0.3)  end (-1.2, -0.4, -2.0) deg
           delta (-1.2, +0.1, -2.3)  range (4.7, 4.3, 2.5) deg
```

## 末端位置定义

### "末端"指哪里

FK 计算的末端 (`ee_left` / `ee_right`) 是**夹爪指尖中心**，不是腕关节本身。

```
左腕关节 (left_wrist_pitch_joint7)
  │
  ├── 向前 +0.15m ──→ ● 末端 (ee_left)
  │                    │
  └── 向内 -0.02m ────┘（偏回手指中心）
```

代码中通过 Pinocchio 添加末端框架来定义：

```python
# actibot_fk.py
pin.Frame('ee_left',
    self.reduced_robot.model.getJointId('left_wrist_pitch_joint7'),
    pin.SE3(pin.Quaternion(1,0,0,0), np.array([0.15, -0.02, 0.0])),
    ...)
```

实际对应机械结构：
- **X(前)**：腕→夹爪的 15cm 结构长度
- **Y(侧)**：-2cm（左）/ +2cm（右），偏移使原点落于两指中心
- 零位时末端在世界系位置：`(0.4644, ±0.2260, 0.7506)`（底座底面中心为原点）

### 坐标系

| 轴 | 方向 | 说明 |
|-----|------|------|
| X   | 向前 | 机器人正前方 |
| Y   | 向左 | 机器人左手方向 |
| Z   | 向上 | 垂直地面 |

原点 = 底座底面中心 (`base_link`)。

### 单位

| 量 | 单位 | 说明 |
|---|------|------|
| `(x, y, z)` 位置 | **米** (m) | 末端在空间中的坐标 |
| `(roll, pitch, yaw)` 姿态 | **弧度** (rad) | 末端绕 X/Y/Z 轴的旋转 |
| 关节角度 q | **弧度** (rad) | FK 输入值 |

### 字段含义

| 字段 | 说明 |
|------|------|
| `start` / `end` | 起始/最后一帧末端位置 (x, y, z) 米或姿态 (roll, pitch, yaw) 弧度 |
| `delta` | 起始→结束变化量（反映净位移） |
| `range` | 全程变化范围（最大−最小，反映振幅） |
| `路程` | 每帧间位移累加（比 delta 更能反映实际运动量） |
| `motion_label` | 动作自动归类结果 |

直线度 = 净位移 / 总路程，越接近 100% 表示单向运动，越低表示往复。

### 动作归类规则

| 条件 | 标签 |
|------|------|
| 两侧末端位移 < 5mm | 静止 / 微调 |
| 一侧位移 > 另一侧 3 倍且 > 3cm | 左/右臂 x/y/z 运动 |
| 两侧均 > 3cm | 双臂运动 |
| 终点关节角度 < 0.3 rad | → 趋于零位 |

## 数据格式

输入 parquet 要求包含列：

| 列 | 类型 | 说明 |
|------|------|------|
| `observation.state` | float32[16] | 左臂7 + 右臂7 + 夹爪2 |
| `action` | float32[16] | 同维度动作指令 |
| `timestamp` | float32 | 时间戳(秒) |
| `episode_index` | int64 | episode 编号 |
| `frame_index` | int64 | 帧编号 |
| `annotation.human.validity` | int64 | 人工标注有效性 |
