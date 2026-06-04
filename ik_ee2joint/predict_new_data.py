#!/usr/bin/env python3
"""用训练好的 ik_ee2joint 模型预测新数据集。

用法:
  conda activate actibot_sdk
  python ik_ee2joint/predict_new_data.py
"""
import os
import sys
import time
import pickle
import numpy as np
import torch
import pandas as pd

_this_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, ".."))
_example_dir = os.path.join(_project_root, "example")
for p in [_this_dir, _project_root, _example_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import paths, data_config
from model import IKMLP
from fk_utils import load_ik, compute_ee_pose

DATA_DIR = "/home/ubuntu/code/End2Joint/data/my_dataset_groot_fk_results"
EE_COLS = data_config["col_eeL"] + data_config["col_eeR"]
JOINT_COLS = data_config["col_joints_l"] + data_config["col_joints_r"]
N_EPISODES = 3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 加载模型 ──
    ckpt_path = os.path.join(paths["results_dir"], "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"错误: 未找到模型 {ckpt_path}"); return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    with open(os.path.join(paths["results_dir"], "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    model = IKMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"模型加载完成 (epoch {ckpt['epoch']})")
    ik = load_ik()

    # ── 遍历前 N 个 episode ──
    for ep in range(N_EPISODES):
        fpath = os.path.join(DATA_DIR, f"episode_{ep:06d}_fk.parquet")
        if not os.path.exists(fpath):
            print(f"  文件不存在: {fpath}"); continue

        df = pd.read_parquet(fpath)
        n = len(df)
        print(f"\n{'='*60}")
        print(f"Episode {ep} ({n} 帧)")
        print('='*60)

        # 输入仅 12D ee_pose
        ee_data = df[EE_COLS].values.astype(np.float64)
        joints_gt = df[JOINT_COLS].values.astype(np.float64)

        # 归一化 → 推理
        X_norm = scaler.transform_X(ee_data)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_norm = model(X_tensor).cpu().numpy()
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - t0

        q_pred = scaler.inverse_y(pred_norm)
        q_gt = joints_gt

        # 精度指标
        abs_err = np.abs(q_pred - q_gt)
        mae_deg = float(np.mean(abs_err)) * 180.0 / np.pi
        rmse_deg = float(np.sqrt(np.mean(abs_err ** 2))) * 180.0 / np.pi

        # FK 重算末端误差（使用真实 ee_data 作为目标）
        pos_errs, ori_errs = [], []
        for i in range(n):
            ee_pred = compute_ee_pose(ik, q_pred[i])
            pos_errs.append(np.linalg.norm(ee_pred[:3] - ee_data[i, :3]))
            pos_errs.append(np.linalg.norm(ee_pred[6:9] - ee_data[i, 6:9]))
            ori_errs.append(np.linalg.norm(ee_pred[3:6] - ee_data[i, 3:6]))
            ori_errs.append(np.linalg.norm(ee_pred[9:12] - ee_data[i, 9:12]))

        # 首帧 vs 末帧对比
        jn = JOINT_COLS
        print(f"\n  ├─ 首帧预测 vs 真实 (度):")
        for i in range(7):
            print(f"  │  {jn[i]:>15s}: pred={np.degrees(q_pred[0,i]):>7.3f}  gt={np.degrees(q_gt[0,i]):>7.3f}  delta={np.degrees(q_pred[0,i]-q_gt[0,i]):+7.3f}")
        print("  │  " + "─" * 45)
        for i in range(7, 14):
            print(f"  │  {jn[i]:>15s}: pred={np.degrees(q_pred[0,i]):>7.3f}  gt={np.degrees(q_gt[0,i]):>7.3f}  delta={np.degrees(q_pred[0,i]-q_gt[0,i]):+7.3f}")

        last = n - 1
        print(f"  └─ 末帧 (frame {last}) 预测 vs 真实 (度):")
        for i in range(7):
            print(f"     {jn[i]:>15s}: pred={np.degrees(q_pred[last,i]):>7.3f}  gt={np.degrees(q_gt[last,i]):>7.3f}  delta={np.degrees(q_pred[last,i]-q_gt[last,i]):+7.3f}")
        print("     " + "─" * 45)
        for i in range(7, 14):
            print(f"     {jn[i]:>15s}: pred={np.degrees(q_pred[last,i]):>7.3f}  gt={np.degrees(q_gt[last,i]):>7.3f}  delta={np.degrees(q_pred[last,i]-q_gt[last,i]):+7.3f}")

        print(f"\n  推理时间: {elapsed*1000:.2f} ms ({elapsed/n*1e6:.1f} μs/帧)")
        print(f"  关节角 MAE: {mae_deg:.3f}° ({np.mean(abs_err)*1000:.2f} mrad)")
        print(f"  关节角 RMSE: {rmse_deg:.3f}°")
        print(f"  FK 位置误差: {np.mean(pos_errs)*1000:.2f} mm (max {np.max(pos_errs)*1000:.2f})")
        print(f"  FK 姿态误差: {np.mean(ori_errs)*1000:.2f} mrad (max {np.max(ori_errs)*1000:.2f})")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
