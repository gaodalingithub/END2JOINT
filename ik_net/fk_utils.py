"""FK 工具：加载 Pinocchio 模型，计算末端位姿，用于 FK 一致性评估。"""
import sys
import os
import numpy as np

# ── 确保能导入 actibot_fk ──
_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_example_dir = os.path.join(_project_dir, "example")
for p in [_example_dir, _project_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


def load_ik():
    """加载 Arm_IK 实例，lazy import 避免启动时依赖 conda 环境。"""
    from actibot_fk import Arm_IK
    from config import paths
    urdf = os.path.join(paths["project_dir"], paths["urdf_path"])
    return Arm_IK(urdf)


def compute_ee_pose(ik, q_joints):
    """从 14D 关节角计算末端位姿（body 关节置零）。

    q_joints: (14,) = [L_7, R_7]
    Returns: (12,) = [eeL_xyz, eeL_rpy, eeR_xyz, eeR_rpy]
    """
    q_19 = np.zeros(19)
    q_19[5:12] = q_joints[:7]   # left arm
    q_19[12:19] = q_joints[7:]  # right arm
    T_l, T_r = ik.get_fk_solution(q_19)

    from pinocchio.rpy import matrixToRpy
    ee = np.concatenate([
        T_l[:3, 3],               # eeL position
        matrixToRpy(T_l[:3, :3]),  # eeL orientation (rad)
        T_r[:3, 3],               # eeR position
        matrixToRpy(T_r[:3, :3]),  # eeR orientation (rad)
    ])
    return ee


def batch_fk_error(ik, q_pred, ee_target):
    """批量计算 FK 一致性误差。

    q_pred:    (N, 14)  预测关节角
    ee_target: (N, 12)  目标末端位姿 [eeL_xyzrpy, eeR_xyzrpy]
    Returns:   pos_err_mean, pos_err_max, ori_err_mean, ori_err_max
    """
    N = q_pred.shape[0]
    pos_errs = []
    ori_errs = []
    for i in range(N):
        ee_pred = compute_ee_pose(ik, q_pred[i])
        # 位置误差 (前3+L + 后6中的前3+R)
        pos_err = np.mean(np.abs(ee_pred[:3] - ee_target[i, :3])) + \
                  np.mean(np.abs(ee_pred[6:9] - ee_target[i, 6:9]))
        pos_err /= 2  # 平均左右臂
        # 姿态误差 (RPY 角度差)
        ori_err = np.mean(np.abs(ee_pred[3:6] - ee_target[i, 3:6])) + \
                  np.mean(np.abs(ee_pred[9:12] - ee_target[i, 9:12]))
        ori_err /= 2
        pos_errs.append(pos_err)
        ori_errs.append(ori_err)

    return (float(np.mean(pos_errs)), float(np.max(pos_errs)),
            float(np.mean(ori_errs)), float(np.max(ori_errs)))
