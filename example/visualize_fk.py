"""
可视化验证 — 用 meshcat 显示机器人骨架 + 末端坐标轴

用法:
  conda activate actibot_sdk
  python example/visualize_fk.py

打开浏览器查看机器人，循环演示多种动作。
"""
import sys, os, time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pinocchio as pin
import meshcat
import meshcat.geometry as mg
import meshcat.transformations as mtf

from example.actibot_fk import Arm_IK

# ─── 加载模型 ──────────────────────────────────────────────────
urdf_path = "actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf"
ik = Arm_IK(urdf_path)
model = ik.reduced_robot.model
data = model.createData()
NQ = model.nq

# ─── 启动 meshcat ──────────────────────────────────────────────
vis = meshcat.Visualizer()
print(f"\n🌐 浏览器打开: {vis.url()}")
print("   Ctrl+C 退出\n")

# 关节索引命名（方便阅读）
J = {
    "up_down": 0, "waist_yaw": 1, "waist_pitch": 2,
    "head_yaw": 3, "head_pitch": 4,
    "L_sh_pitch": 5, "L_sh_roll": 6, "L_sh_yaw": 7,
    "L_el_pitch": 8, "L_el_roll": 9, "L_wr_yaw": 10, "L_wr_pitch": 11,
    "R_sh_pitch": 12, "R_sh_roll": 13, "R_sh_yaw": 14,
    "R_el_pitch": 15, "R_el_roll": 16, "R_wr_yaw": 17, "R_wr_pitch": 18,
}

# ─── 计算关节 + 末端的世界坐标 ────────────────────────────────
def get_joint_positions(q):
    pin.forwardKinematics(model, data, q)
    pos = {}
    for j in range(1, model.njoints):
        pos[j] = data.oMi[j].translation.copy()
    for ee in ["ee_left", "ee_right"]:
        fid = model.getFrameId(ee)
        pin.updateFramePlacement(model, data, fid)
        pos[ee] = data.oMf[fid].translation.copy()
    return pos

# ─── 构建 meshcat 场景 ─────────────────────────────────────────
def build_scene(q):
    pos = get_joint_positions(q)
    vis.delete()

    # 关节球
    for j, p in pos.items():
        if isinstance(j, str):
            continue
        vis[f"joints/j{j}"].set_object(
            mg.Sphere(0.025), mg.MeshPhongMaterial(color=0x4488ff))
        vis[f"joints/j{j}"].set_transform(mtf.translation_matrix(p))

    # 骨架线
    torso = [pos[1], pos[2], pos[3]]
    vis["torso"].set_object(
        mg.Line(mg.PointsGeometry(np.array(torso).T),
                mg.LineBasicMaterial(color=0x888888, linewidth=2)))

    left_arm = [pos[3]] + [pos[j] for j in range(6, 13)] + [pos["ee_left"]]
    vis["links/left_arm"].set_object(
        mg.Line(mg.PointsGeometry(np.array(left_arm).T),
                mg.LineBasicMaterial(color=0xff6644, linewidth=3)))

    right_arm = [pos[3]] + [pos[j] for j in range(13, 20)] + [pos["ee_right"]]
    vis["links/right_arm"].set_object(
        mg.Line(mg.PointsGeometry(np.array(right_arm).T),
                mg.LineBasicMaterial(color=0x44aaff, linewidth=3)))

    # 末端坐标轴 (RGB = XYZ)
    for side, ee_key in [("left", "ee_left"), ("right", "ee_right")]:
        fid = model.getFrameId(f"ee_{side}")
        pin.updateFramePlacement(model, data, fid)
        R = data.oMf[fid].rotation
        p = pos[ee_key]
        for ax_idx, color in enumerate([0xff0000, 0x00ff00, 0x0000ff]):
            end = p + R[:, ax_idx] * 0.08
            vis[f"axes_{side}/{ax_idx}"].set_object(
                mg.Line(mg.PointsGeometry(np.array([p, end]).T),
                        mg.LineBasicMaterial(color=color, linewidth=3)))

    # 末端标记
    for side, ee_key, color in [("left", "ee_left", 0xff4444), ("right", "ee_right", 0x44aaff)]:
        vis[f"ee_{side}/point"].set_object(
            mg.Sphere(0.04), mg.MeshPhongMaterial(color=color))
        vis[f"ee_{side}/point"].set_transform(mtf.translation_matrix(pos[ee_key]))


# ─── 动作库 ────────────────────────────────────────────────────
def action_zero(_phase=None):
    """零位"""
    return np.zeros(NQ)

def action_arms_side(phase):
    """双臂侧平举 — shoulder_roll"""
    q = np.zeros(NQ)
    q[J["L_sh_roll"]] = 1.2 * np.sin(phase)
    q[J["R_sh_roll"]] = -1.2 * np.sin(phase)
    return q

def action_arms_front(phase):
    """双臂前平举 — shoulder_pitch"""
    q = np.zeros(NQ)
    q[J["L_sh_pitch"]] = 0.6 * np.sin(phase)
    q[J["R_sh_pitch"]] = -0.6 * np.sin(phase)
    return q

def action_elbow_curl(phase):
    """弯举 — elbow_pitch"""
    q = np.zeros(NQ)
    q[J["L_el_pitch"]] = -1.2 * (np.sin(phase) * 0.5 + 0.5)
    q[J["R_el_pitch"]] = -1.2 * (np.sin(phase) * 0.5 + 0.5)
    return q

def action_wave(phase):
    """右手挥手 — 综合"""
    q = np.zeros(NQ)
    q[J["R_sh_roll"]] = 0.5                          # 右臂侧抬固定
    q[J["R_sh_pitch"]] = -0.8                         # 右臂前抬固定
    q[J["R_el_pitch"]] = -0.5                         # 右肘微弯
    q[J["R_wr_yaw"]] = 0.4 * np.sin(phase * 2)        # 手腕左右摆动
    return q

def action_waist_twist(phase):
    """扭腰 — 上半身整体旋转"""
    q = np.zeros(NQ)
    q[J["waist_yaw"]] = 0.4 * np.sin(phase)
    return q

def action_reach_up(phase):
    """双手上举"""
    q = np.zeros(NQ)
    q[J["L_sh_roll"]] = 0.3 * (1 - np.cos(phase))   # 左臂从侧边抬起
    q[J["L_sh_pitch"]] = -0.8 * (1 - np.cos(phase))
    q[J["R_sh_roll"]] = -0.3 * (1 - np.cos(phase))  # 右臂镜像
    q[J["R_sh_pitch"]] = 0.8 * (1 - np.cos(phase))
    q[J["L_el_pitch"]] = -0.3
    q[J["R_el_pitch"]] = -0.3
    return q

def action_hello(phase):
    """双手打招呼"""
    q = np.zeros(NQ)
    q[J["R_sh_roll"]] = 0.6
    q[J["R_sh_pitch"]] = -1.2
    q[J["R_el_roll"]] = 0.3
    q[J["R_wr_yaw"]] = 0.3 * np.sin(phase * 2)       # 手腕摆动
    return q

# 动作列表：每个动作 (名称, 函数, 相位速度, 持续秒数)
ACTIONS = [
    ("零位 — 所有关节归零",            action_zero,       0.0,  1.5),
    ("双臂侧平举 — shoulder_roll 运动", action_arms_side,  2.0,  3.0),
    ("双臂前平举 — shoulder_pitch 运动",action_arms_front, 2.0,  3.0),
    ("弯举 — elbow_pitch 运动",         action_elbow_curl, 2.0,  3.0),
    ("右手挥手 — 多关节协调",          action_wave,       1.5,  4.0),
    ("扭腰 — waist_yaw 运动",          action_waist_twist, 1.5,  3.0),
    ("双手上举 — 多关节大范围",        action_reach_up,   1.2,  4.0),
    ("打招呼 — 综合",                  action_hello,       1.5,  4.0),
]


# ─── 主循环 ────────────────────────────────────────────────────
print("演示动作列表:")
for i, (name, *_) in enumerate(ACTIONS):
    print(f"  {i+1}. {name}")
print("-" * 40)

try:
    while True:
        for action_name, action_fn, speed, duration in ACTIONS:
            print(f"\r▶ {action_name}", end="", flush=True)
            t0 = time.time()
            while time.time() - t0 < duration:
                elapsed = time.time() - t0
                phase = elapsed * speed
                q = action_fn(phase)
                build_scene(q)
                time.sleep(0.03)
except KeyboardInterrupt:
    print("\n退出")
