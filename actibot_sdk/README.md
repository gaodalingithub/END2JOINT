# actibot_sdk
机器人控制接口、遥操作、应用接口

## SDK操作说明：

### 整体架构
- **actibot_ros_main（底层控制节点）**：启动 `actibot_ros` ROS2 节点，负责读取电机状态、下发关节/底盘/升降控制指令，并提供电机使能、零位、零力拖动等底层服务。
- **application_main（应用节点）**：启动 `actibot_application` ROS2 节点，订阅遥操作（Teleop）和 VLA 的控制话题，根据模式（TELEOP / VLA / ACTION / FROZEN / IDLE）转发到底层控制话题，并提供一组上层任务/动作服务。
- **teleop_main（VR 遥操作节点）**：启动 `teleop_vr` ROS2 节点，从 VR 设备读取数据，经过 IK 求解生成控制命令，发布到遥操作控制话题，并提供启动/停止遥操作的服务。

建议启动顺序：**先配置 CAN → 启动 `actibot_ros_main` → 启动 `application_main` → 启动 `teleop_main` 与 VR APP**。
如果不需要application 中的功能，可以不启动application_main。

### 1. 前置步骤：CAN 配置

```shell

# 使用一下命令查看can 情况，可以插拔一下检查是否正确对应
ip link show 

# 可参考一下脚本，需确认一下can端口对应的哪部分电机
./setup.sh
```

### 2. 底层控制节点：actibot_ros_main（actibot_ros）


- **直接在源码目录运行（开发调试）**
  ```shell
  python actibot_ros_main.py
  python actibot_ros_main.py --wo_init_motors
  ```

- **话题（topics）**
  - **状态发布（由 `actibot_ros` 发布，`sensor_msgs/JointState`）**
    - `actibot_arm_state`：左右臂 + 夹爪的关节状态。
    - `actibot_body_state`：腰部、头部关节状态。
    - `actibot_lift_state`：升降机构高度状态。


  - **控制订阅（由上层/遥操作发布，`actibot_ros` 订阅，`sensor_msgs/JointState`）**
    - `actibot_arm_ctrl`：
      - `position[0:14]`：14 个手臂关节目标角度；
      - `position[14:16]`：左右夹爪开合量；
      - `velocity/effort` 为空时内部补 0；启用重力补偿时，若 `effort` 为 0，将自动计算前馈力矩。
    - `actibot_body_ctrl`：腰/头关节控制。
    - `actibot_lift_ctrl`：`position[0]` 为目标升降高度。
    - `actibot_chassis_ctrl`：使用 `velocity[0]`、`velocity[1]` 表示左右轮角速度，内部转换为底盘 `Twist`。

- **服务（services，由 `actibot_ros` 提供）**
  - `/init_robot` (`std_srvs/Trigger`)：电机使能、进入控制模式。
    ```shell
    ros2 service call /init_robot std_srvs/srv/Trigger "{}"
    ```
  - `/deinit_robot` (`std_srvs/Trigger`)：电机去使能。
    ```shell
    ros2 service call /deinit_robot std_srvs/srv/Trigger "{}"
    ```
  - `/go_zero_position` (`std_srvs/Trigger`)：插值回零位（手臂 + 腰部零位由 `config/control_config.yaml` 配置）。
    ```shell
    ros2 service call /go_zero_position std_srvs/srv/Trigger "{}"
    ```
  - `/set_zero_force_mode` (`std_srvs/SetBool`)：零力拖动模式开关（需要配置中开启 `zero_force_control` 且 `auto_gravity_compensation` 为 true）。
    ```shell
    ros2 service call /set_zero_force_mode std_srvs/srv/SetBool "{data: true}"   # 开启零力拖动
    ros2 service call /set_zero_force_mode std_srvs/srv/SetBool "{data: false}"  # 关闭零力拖动
    ```
  - `/fold_robot` (`std_srvs/Trigger`)：折叠机器人，仅在特定升降结构（`lift_type == 2`）下可用。
    ```shell
    ros2 service call /fold_robot std_srvs/srv/Trigger "{}"
    ```
  - `/close_actibot_ros_node` (`std_srvs/Trigger`)：请求关闭 `actibot_ros` 节点（设置内部 `shutdown`，主循环退出）。

### 3. 应用节点：application_main（actibot_application）

- **启动方式（源码目录）**
  ```shell
  python application_main.py                       # 只做任务调度，不自动检测 CAN
  python application_main.py --auto_can_activate   # 自动检测并激活各段 CAN 口（推荐实机）
  ```

- **订阅的话题**
  - **底层状态（来自 `actibot_ros`，`sensor_msgs/JointState`）**
    - `actibot_arm_state`
    - `actibot_body_state`
    - `actibot_lift_state`
    - `actibot_chassis_state`
  - **遥操作控制（来自 `teleop_vr`，`sensor_msgs/JointState`）**
    - `actibot_arm_ctrl`
    - `actibot_body_ctrl`
    - `actibot_lift_ctrl`
    - `actibot_chassis_ctrl`
    - 在 **TELEOP 模式** 下，这些消息会被直接转发为对底层的控制命令。
  - **VLA 控制（来自算法/VLA 模块，`sensor_msgs/JointState`）**
    - `vla/arm_ctrl`
    - `vla/body_ctrl`
    - `vla/lift_ctrl`
    - `vla/chassis_ctrl`
    - 在 **VLA 模式** 下，这些消息将被转发到底层控制话题。

- **发布的话题**
  - **下发到底层的控制命令（`sensor_msgs/JointState`）**
    - `actibot_arm_ctrl`
    - `actibot_body_ctrl`
    - `actibot_lift_ctrl`
    - `actibot_chassis_ctrl`
    - 实际来源取决于当前模式：TELEOP 使用 teleop_* 命令，VLA 使用 vla_* 命令。
  - **应用状态（`std_msgs/String`）**
    - `application/mode_state`：当前模式字符串，取值为 `IDLE`、`TELEOP`、`VLA`、`ACTION`、`FROZEN`。
      ```shell
      ros2 topic echo application/mode_state
      ```

- **服务（由 `actibot_application` 提供，名称见 `config/ros_config.yaml`）**
  - **模式切换相关**
    - `application/set_teleop` (`std_srvs/SetBool`)：
      - `data: true` → 进入遥操作模式（TELEOP）；
      - `data: false` → 回到空闲（IDLE）。
      ```shell
      ros2 service call application/set_teleop std_srvs/srv/SetBool "{data: true}"
      ```
    - `application/set_vla` (`std_srvs/SetBool`)：
      - `data: true` → 进入 VLA 模式；
      - `data: false` → 回到空闲（IDLE）。
      ```shell
      ros2 service call application/set_vla std_srvs/srv/SetBool "{data: true}"
      ```
    - `application/set_action` (`std_srvs/Trigger`)：进入 ACTION 模式，由内部动作线程执行特定动作。
      ```shell
      ros2 service call application/set_action std_srvs/srv/Trigger "{}"
      ```
    - `application/stop` (`std_srvs/Trigger`)：进入 FROZEN 模式，保持当前姿态，不再继续下发新的控制命令。
      ```shell
      ros2 service call application/stop std_srvs/srv/Trigger "{}"
      ```

  - **预定义动作相关**
    - `application/unfold_robot` (`std_srvs/Trigger`)：展开机器人（手臂、腰、升降按照 `config/application_config.yaml` 中预定义位姿插值运动），结束后进入 FROZEN。
      ```shell
      ros2 service call application/unfold_robot std_srvs/srv/Trigger "{}"
      ```
    - `application/fold_robot` (`std_srvs/Trigger`)：折叠机器人，内部会调用底层 `fold_robot` 服务并安全下电。
      ```shell
      ros2 service call application/fold_robot std_srvs/srv/Trigger "{}"
      ```
    - `application/wave_hand` (`std_srvs/Trigger`)：挥手动作（按配置中的 `wave_hand.arm_pos_start/arm_pos_end` 执行动作序列）。
      ```shell
      ros2 service call application/wave_hand std_srvs/srv/Trigger "{}"
      ```
    - `application/stretch_hand` (`std_srvs/Trigger`)：伸展手臂动作。
      ```shell
      ros2 service call application/stretch_hand std_srvs/srv/Trigger "{}"
      ```

> 推荐流程：**先通过 `/init_robot` 使能电机，调用 `application/unfold_robot` 展开，再切到 TELEOP 或 VLA 模式，最后根据需要调用 `application/fold_robot` + `/deinit_robot` 完成收拢与下电。**

### 4. 遥操作节点：teleop_main（teleop_vr）

- **启动方式（源码目录）**
  ```shell
  # 直接启动，内部依靠 VR 手柄按键 A/Y 控制校准与零位
  python teleop_main.py

  # 启动后先进入 WAIT 状态，需要通过 ROS service 显式开启/停止遥操作
  python teleop_main.py --wait_teleop_call
  ```

- **发布的话题**
  - **控制命令（`sensor_msgs/JointState`）**
    - `actibot_arm_ctrl`：由 VR 手柄位姿 + IK 求解得到的 14 关节 + 2 个夹爪目标，内部包含速度限制与前馈力矩计算。
    - `actibot_body_ctrl`：腰部 4 自由度命令（实际只使用前 2 维 yaw/pitch，其余为占位）。
    - `actibot_chassis_ctrl`：底盘左右轮速度命令（映射左摇杆）。
    - `actibot_lift_ctrl`：升降高度命令（映射右摇杆）。
  - **遥操作状态与按键**
    - `teleop/joy` (`sensor_msgs/Joy`)：发布 VR 手柄的摇杆和按键状态，便于其他节点复用或录制。
    - `teleop/status` (`std_msgs/Bool`)：当内部状态为 ACTIVE（已标定且在遥操作）时为 `True`，否则为 `False`。

- **订阅的话题（用于校准与状态对齐）**
  - `actibot_arm_state`、`actibot_body_state`、`actibot_lift_state`（来自 `actibot_ros`）：
    - 在校准阶段（按 A 键）会读取当前机器人实际关节状态作为遥操作锚点，保证接管时姿态连续、不跳变。

- **服务（由 `teleop_vr` 提供）**
  - `teleop/start_teleop` (`std_srvs/SetBool`)：
    - 若使用 `--wait_teleop_call` 启动：
      - `data: true`：从 WAIT 切换到 FROZEN，允许通过 VR 手柄 A/Y 进入校准和运动；
      - `data: false`：切回 FROZEN，停止遥操作输出。
    ```shell
    ros2 service call teleop/start_teleop std_srvs/srv/SetBool "{data: true}"
    ```
  - `go_zero_position` 客户端（`std_srvs/Trigger`）：
    - `teleop_vr` 在 `init()` 时会调用底层 `/go_zero_position` 服务，让机器人平滑回到零位，用户通常只需保证该服务可用即可。

### 5. 配置文件说明（config/*.yaml）

#### 5.1 `config/control_config.yaml`

- **CAN 口与平台选择**
  - `can_id.platform`：
    - `auto`：自动检测 CAN 口顺序（推荐实机，重启/插拔后顺序自动适配）。
    - `manual`：手动指定每一段 CAN 名称，需要与 `ip link show` 输出一致。
  - `can_id.left_arm/right_arm/body/chassis`：
    - 在 `manual` 模式下使用，例如：
      - `left_arm: can1_fd`、`right_arm: can2_fd`、`body: can3`、`chassis: null`。

- **机械结构类型**
  - `lift_type`：
    - `0`：推缸升降；
    - `1`：导轨升降；
    - `2`：折叠升降（配合腰部折叠结构，支持 `/fold_robot`）。
  - `gripper_type`：
    - `0`：无夹爪；
    - `1`：二指夹爪。

- **控制参数（电机模式与 PID）**
  - `control_params.arm`：
    - `mode`：`"MIT"`（位置+速度+力矩 MIT 控制）或 `"POS"`（位置+速度模式）。
    - `Kp`、`Kd`：MIT 模式下的关节 PID；`VEL` 为 POS 模式下的最大速度。
  - `control_params.gripper`：
    - `Kp`、`Kd`：夹爪力控/位控 PID。
  - `control_params.waist`：
    - `mode`：腰部控制模式（MIT/POS），决定 `waist_control` 下发方式。

- **初始姿态与重力补偿**
  - `init_arm_position`：
    - 14 维数组，对应两臂零位姿态，供 `/go_zero_position` 和若干动作使用。
  - `auto_gravity_compensation`：
    - `true`：启用重力补偿，`actibot_ros` 会创建 IK 求解器并在未指定 `effort` 时自动填充前馈力矩。

- **运动保护与急停**
  - `move_protect` / `move_protect_clip_value`：
    - 控制手臂指令的单步关节变化限幅（14 关节按 7+7 复制），超过则自动裁剪，避免大幅跳变。
  - `emergency_stop_param`：
    - `emergency_stop`：是否启用基于力矩的碰撞检测与紧急停止。
    - `emergency_stop_time`：超过设定时间仍持续碰撞则触发急停。
    - `arm_effort_limit_value`：各关节在“重力补偿后剩余力矩”的上限。

- **零力拖动（Zero Force Control）**
  - `zero_force_control_param.zero_force_control`：
    - 打开后，才允许通过 `/set_zero_force_mode` 切换零力拖动。
  - `zero_force_control_value`：
    - 手臂 7 关节 + 1 个夹爪的力矩死区，数值越大，需要更大外力才会开始“跟随拖动”。


#### 5.2 `config/ros_config.yaml`

- 统一配置了 **所有节点用到的 ROS 话题与服务名**，便于修改命名空间或与其他系统对接。
  - `actibot_ros.topics.state/ctrl.*`：
    - 对应 `actibot_ros` 节点中 `create_publisher`/`create_subscription` 使用的话题名（如 `actibot_arm_state`、`actibot_arm_ctrl`）。
  - `actibot_ros.services.*`：
    - 定义 `/init_robot`、`/deinit_robot`、`/fold_robot`、`/set_zero_force_mode` 等服务名，需与命令行调用保持一致。
  - `teleop.topics.*`：
    - `teleop_vr` 使用的遥操作话题和状态话题：`actibot_*_ctrl`、`teleop/joy`、`teleop/status`。
  - `teleop.services.start_teleop`：
    - `teleop/start_teleop` 服务名绑定到 `teleop_vr` 的 `start_teleop_callback`。
  - `vla.topics.*`：
    - VLA 上位控制话题，`application` 节点在 VLA 模式下订阅这些话题并转发到 `actibot_*_ctrl`。
  - `application.topics/services.*`：
    - 应用节点发布的 `application/mode_state` 及所有动作/模式服务（`application/unfold_robot`、`application/fold_robot` 等）的名字。

> **修改话题名或服务名时，务必同时更新**：对应的 `yaml` 配置和外部 ROS 调用（例如 `ros2 topic pub` / `ros2 service call` 命令），保持一致即可生效。

#### 5.3 `config/application_config.yaml`

- 主要配置 **上层应用动作的目标位姿和高度**，由 `ActibotApplicationNode` 使用：
  - `action.unfold_robot`：
    - `arm_pos`：展开动作结束时的 16 维手臂+夹爪关节数组（应用节点插值生成中间轨迹）。
    - `body_pos`：腰/头目标位姿。
    - `height`：升降机构目标高度（单位与 `control_config` 中一致）。
  - `action.fold_robot`：
    - 对应折叠动作（回收手臂、腰、升降），`application/fold_robot` 会使用这里的值。
  - `action.wave_hand.arm_pos_start/arm_pos_end`：
    - 挥手动作的起始与结束关节组；`application/wave_hand` 会在两组姿态之间多次往返插值。
  - `action.stretch_hand.arm_pos`：
    - 伸展手臂动作的目标姿态；`application/stretch_hand` 使用。

> 如果你希望**定制自己的“展开/折叠/挥手/伸展”动作**，可以直接修改这些数组，保持长度与当前机器人 DOF 一致即可。

#### 5.4 `config/teleop_config.yaml`

- **teleop 运行模式与功能开关**

  - `teleop.control_arm/control_chassis/control_waist/control_lift`：
    - 决定是否启用相应子系统控制，例如只用手臂 + 升降时可关掉底盘与腰部控制。

- **VR 安全与运动限制**
  - `teleop.max_controller_jump_m`：
    - 相邻两帧手柄线位移超过该值则视为异常帧丢弃，避免“瞬移”导致的突发大动作。
  - `teleop.dt`：
    - 主循环周期（秒），决定遥操作控制频率。
  - `teleop.arm_step_limit_rad`：
    - 手臂每周期允许的最大关节变化（单位弧度），超出则自动限幅，防止控制过猛。
  - `teleop.lock_waist_zero_when_disabled`：
    - 当不控制腰部时，是否在第一次接管时把腰锁在 0 位。

- **IK 求解与 URDF 配置（`ik` 段）**
  - `ik.urdf_path`：
    - 使用的 URDF 路径，与 `control_config.yaml` 中的 `urdf_path` 一致时最简单，若更换机器人模型只需要修改这里。
  - `ik.ik_type`：
    - 当前为 `casadi`，也支持 `dls`；




### VR 操作说明
1. VR APP连接后，点击发送(数据)
2. 启动遥操作节点，默认为暂停状态
   - 暂停按Y初始化，机器人会抬起到零位
   - 按A标定当前位置，即可开始遥操作, 其中左遥杆控制底盘、右摇杆控制升降，双臂由手柄位姿控制，腰部可使用头显位姿控制，扳机可以控制夹爪
   - 按B进入暂停状态暂停



## conda环境搭建(默认直接使用docker环境，docker容器中使用)

初始化、配置conda环境
```shell
conda create -n actibot_sdk python=3.12
conda activate actibot_sdk
conda install pinocchio=3.6.0 -c conda-forge
pip install meshcat casadi matplotlib

```
安装openarm can python, 在actibot_can仓库的 /python中，执行：
```
./actibot_can/python/build.sh
```