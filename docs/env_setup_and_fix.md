# 环境配置与代码修改说明

## 一、环境搭建

### 1. 安装 Miniconda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
```

### 2. 创建 conda 环境并安装依赖

```shell
conda create -n actibot_sdk python=3.12 -y
conda activate actibot_sdk
conda install pinocchio=3.6.0 -c conda-forge -y
pip install meshcat casadi matplotlib numpy
```

> **注意**：PyPI 上有一个同名的 `pinocchio` 包（v0.4.3），它是另一个不相关的项目。必须从 `conda-forge` 频道安装正确的机器人学 Pinocchio 库。

---

## 二、`docs/actibot_fk_example.py` 与 `example/actibot_fk.py` 差异说明

两个文件均实现 Actibot v3 机器人的正运动学 (FK) 计算，`get_fk_solution()` 核心方法完全相同。差异在于环境适配。

### 差异 1：`import utils`

| 文件 | 代码 |
|------|------|
| 原始 | `import utils` |
| 当前 | `# import utils  # not available...` |

**原因**：`actibot_sdk/utils.so` 是编译为 **ARM aarch64** 的 Cython 3.2.1 模块，当前系统为 **x86_64**，无法加载。且 FK 计算不依赖该模块。

### 差异 2：模型加载方式

| 文件 | 代码 |
|------|------|
| 原始 | `pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=...)` |
| 当前 | `buildModelFromUrdf(urdf_path)` + 空 `GeometryModel()` |

**原因**：`BuildFromURDF` 会自动加载 mesh 几何文件（`.STL`），但当前环境的 v3 URDF 引用的 mesh 文件不存在，导致 `ValueError: Mesh ... could not be found`。FK 计算仅需运动学模型，无需几何模型。

### 差异 3：关节名大小写

| 文件 | 锁定关节名 |
|------|-----------|
| 原始 | `LeftGripperA_Joint`, `LeftGripperB_Joint` (CamelCase) |
| 当前 | `left_gripper_a_joint`, `left_gripper_b_joint` (snake_case) |

**原因**：v3 URDF（`v3_urdf_251121-2.urdf`）使用 snake_case 命名，原始 v1 URDF 使用 CamelCase。`buildReducedRobot` 按实际 URDF 名称查找关节，必须匹配。

### 差异 4：末端 frame 父关节名

| 文件 | frame 父关节 |
|------|-------------|
| 原始 | `LeftWrist_Joint7`, `RightWrist_Joint7` |
| 当前 | `left_wrist_pitch_joint7`, `right_wrist_pitch_joint7` |

**原因**：同上，v3 URDF 的实际关节名。

### 差异 5：URDF 版本与路径

| 文件 | `test()` 中 URDF 路径 |
|------|---------------------|
| 原始 | `actibot-v1-urdf-0929/urdf/actibot-v1-urdf-0929.urdf`（v1，不存在） |
| 当前 | `actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf`（v3，可用） |

### 差异 6：`test()` q 向量维度

| 文件 | 代码 |
|------|------|
| 原始 | `q = np.zeros(14)` |
| 当前 | `q = np.zeros(ik.reduced_robot.model.nq)` |

**原因**：简化模型有 19 DOF（23 总关节 − 4 锁定夹爪），硬编码 14 会导致维度不匹配。

---

## 三、相同部分

以下代码在两个文件中**完全一致**，是 FK 计算的核心：

### `get_fk_solution()` 方法

```python
def get_fk_solution(self, q):
    pin.forwardKinematics(self.model, self.data, q)
    pin.updateFramePlacement(self.model, self.data, self.left_gripper_id)
    pin.updateFramePlacement(self.model, self.data, self.right_gripper_id)
    T_left = np.eye(4)
    T_left[:3, :3] = self.data.oMf[self.left_gripper_id].rotation
    T_left[:3, 3] = self.data.oMf[self.left_gripper_id].translation
    T_right = np.eye(4)
    T_right[:3, :3] = self.data.oMf[self.right_gripper_id].rotation
    T_right[:3, 3] = self.data.oMf[self.right_gripper_id].translation
    return T_left, T_right
```

### 末端偏置定义

```python
pin.Frame('ee_left', ..., pin.SE3(..., np.array([0.15, -0.02, 0.0])))
pin.Frame('ee_right', ..., pin.SE3(..., np.array([0.15, 0.02, 0.0])))
```

两个文件中偏置量完全相同：腕前 **15cm** + 向内 **2cm**。

### Casadi 符号模型

```python
self.cmodel = cpin.Model(self.reduced_robot.model)
self.cdata = self.cmodel.createData()
self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
```

---

## 四、运行验证

```shell
conda activate actibot_sdk
python example/actibot_fk.py
```

### 输出结果

零位姿态下（所有关节角度为 0），左右末端的 4×4 齐次变换矩阵：

**Left end-effector（左臂末端）：**
```
[[1.         0.         0.         0.464373  ]
 [0.         1.         0.         0.225976  ]
 [0.         0.         1.         0.75060466]
 [0.         0.         0.         1.        ]]
```

- 位置：(x=0.464, y=0.226, z=0.751) 米
- 姿态：单位矩阵（无旋转）

**Right end-effector（右臂末端）：**
```
[[ 1.          0.          0.          0.464371  ]
 [ 0.          1.          0.         -0.225977  ]
 [ 0.          0.          1.          0.75060333]
 [ 0.          0.          0.          1.        ]]
```

- 位置：(x=0.464, y=-0.226, z=0.751) 米
- 姿态：单位矩阵（无旋转）

左右臂在零位时对称分布在身体两侧，高度约 0.75 m，符合机器人结构预期。

---

## 五、URDF 关节结构

完整模型共 **23 个关节**（按运动树顺序）：

| # | 关节名 | 类型 |
|---|--------|------|
| 0 | universe | — |
| 1 | `up_down_joint` | prismatic（升降） |
| 2 | `waist_yaw_joint` | revolute（腰部偏航） |
| 3 | `waist_pitch_joint` | revolute（腰部俯仰） |
| 4 | `head_yaw_joint` | revolute（头部偏航） |
| 5 | `head_pitch_joint` | revolute（头部俯仰） |
| 6-12 | `left_shoulder_pitch_joint1` ~ `left_wrist_pitch_joint7` | revolute（左臂 7 自由度） |
| 13-14 | `left_gripper_a/b_joint` | prismatic（左手夹爪） |
| 15-21 | `right_shoulder_pitch_joint1` ~ `right_wrist_pitch_joint7` | revolute（右臂 7 自由度） |
| 22-23 | `right_gripper_a/b_joint` | prismatic（右手夹爪） |

简化模型（锁定 4 个夹爪关节）：**19 DOF**。

## 六、坐标系定义

| 轴 | 方向 | 说明 |
|----|------|------|
| X | 向前 | 机器人正前方 |
| Y | 向左 | 机器人左手方向 |
| Z | 向上 | 垂直地面 |

原点 `(0, 0, 0)` = 底座底面中心 (`base_link`)。
