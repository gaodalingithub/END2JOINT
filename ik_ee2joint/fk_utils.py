"""FK 工具：加载 Pinocchio 模型，用于 FK 一致性评估。"""
import sys
import os
import numpy as np

_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_example_dir = os.path.join(_project_dir, "example")
for p in [_example_dir, _project_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import paths


def load_ik():
    from actibot_fk import Arm_IK
    urdf = os.path.join(paths["project_dir"], paths["urdf_path"])
    return Arm_IK(urdf)


def compute_ee_pose(ik, q_joints):
    """从 14D 关节角计算末端位姿（body 关节置零）。"""
    q_19 = np.zeros(19)
    q_19[5:12] = q_joints[:7]
    q_19[12:19] = q_joints[7:]
    T_l, T_r = ik.get_fk_solution(q_19)
    from pinocchio.rpy import matrixToRpy
    return np.concatenate([
        T_l[:3, 3], matrixToRpy(T_l[:3, :3]),
        T_r[:3, 3], matrixToRpy(T_r[:3, :3]),
    ])


def batch_fk_error(ik, q_pred, ee_target):
    """批量计算 FK 一致性误差。"""
    N = q_pred.shape[0]
    pos_errs, ori_errs = [], []
    for i in range(N):
        ee_pred = compute_ee_pose(ik, q_pred[i])
        pos = np.mean(np.abs(ee_pred[:3] - ee_target[i, :3])) + \
              np.mean(np.abs(ee_pred[6:9] - ee_target[i, 6:9]))
        pos /= 2
        ori = np.mean(np.abs(ee_pred[3:6] - ee_target[i, 3:6])) + \
              np.mean(np.abs(ee_pred[9:12] - ee_target[i, 9:12]))
        ori /= 2
        pos_errs.append(pos)
        ori_errs.append(ori)
    return (float(np.mean(pos_errs)), float(np.max(pos_errs)),
            float(np.mean(ori_errs)), float(np.max(ori_errs)))
