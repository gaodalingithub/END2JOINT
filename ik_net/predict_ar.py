#!/usr/bin/env python3
"""自回归推理：当前步的预测关节角作为下一步的输入。

每一步的 prev_joints 使用模型上一步的输出，模拟实际部署场景。

用法:
  conda activate actibot_sdk
  python ik_net/predict_ar.py --data data/0602_test_for_net_action_fk --save ik_net/save/0602_test_for_net_action_fk_results_ar_moredata  
  python ik_net/predict_ar.py --data /path/to/data --fkfix         # 启用 FK 修正
  python ik_net/predict_ar.py --count 1000
"""
import os
import sys
import time
import argparse
import pickle
import glob
import numpy as np
import torch
import pandas as pd

_this_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, ".."))
_example_dir = os.path.join(_project_root, "example")
for p in [_this_dir, _project_root, _example_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import hp, paths, data_config
from model import ResidualMLP
from fk_utils import load_ik, compute_ee_pose

EE_COLS = data_config["col_eeL"] + data_config["col_eeR"]
JOINT_COLS = data_config["col_joints_l"] + data_config["col_joints_r"]


def main():
    parser = argparse.ArgumentParser(description="IK-Net 自回归推理")
    parser.add_argument("--data", default=None, help="数据文件夹路径")
    parser.add_argument("--save", default=None, help="保存预测结果到目录")
    parser.add_argument("--count", type=int, default=None, help="只处理前 N 帧")
    parser.add_argument("--fkfix", action="store_true", help="启用 FK 修正")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {hp['input_dim']}D → {hp['output_dim']}D, residual={hp['use_residual']}")

    # ── 加载模型 ──
    ckpt_dir = hp.get("ckpt_dir", paths["results_dir"])
    ckpt_path = os.path.join(ckpt_dir, hp.get("ckpt_name", "best_model.pt"))
    if not os.path.exists(ckpt_path):
        print(f"错误: 未找到模型 {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    with open(os.path.join(ckpt_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    model = ResidualMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"模型加载完成 (epoch {ckpt['epoch']})")
    ik = None
    fk_correction = None
    if args.fkfix:
        import importlib
        fk_robust = importlib.import_module("ik_net_robust.fk_utils")
        fk_correction = fk_robust.fk_correction
        ik = load_ik()

    # ── 加载数据 ──
    data_dir = args.data or paths["data_dir"]
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*_fk.parquet")))
    if not files:
        print(f"未找到数据: {data_dir}"); return

    if args.count:
        first = pd.read_parquet(files[0])
        ee_data = first[EE_COLS].values.astype(np.float64)[:args.count]
        joint_gt = first[JOINT_COLS].values.astype(np.float64)[:args.count]
    else:
        frames = [pd.read_parquet(f) for f in files]
        ee_data = np.vstack([f[EE_COLS].values.astype(np.float64) for f in frames])
        joint_gt = np.vstack([f[JOINT_COLS].values.astype(np.float64) for f in frames])

    n_total = len(ee_data)
    print(f"数据: {n_total} 帧")

    # ── 自回归推理 ──
    q_preds = np.zeros_like(joint_gt)
    prev = joint_gt[0].copy()  # 首帧用真实值初始化
    fkfix_time = 0

    for t in range(n_total):
        ee = ee_data[t:t+1]
        prev_use = joint_gt[0:1] if t == 0 else prev.reshape(1, -1)
        X = np.hstack([ee, prev_use])
        X_norm = scaler.transform_X(X)

        with torch.no_grad():
            pred_norm = model(torch.tensor(X_norm, dtype=torch.float32, device=device)).cpu().numpy()
        q_pred = scaler.inverse_y(pred_norm)[0]

        # FK 修正（可选）
        if args.fkfix and t > 0:
            t0 = time.perf_counter()
            qL, qR, _, _ = fk_correction(ik, q_pred, ee[0])
            fkfix_time += time.perf_counter() - t0
            q_pred[:7] = qL
            q_pred[7:] = qR

        q_preds[t] = q_pred
        prev = q_pred.copy()  # ← 自回归关键：用上一步预测值

    # ── 精度评估 ──
    abs_err = np.abs(q_preds - joint_gt)
    mae_deg = float(np.mean(abs_err)) * 180.0 / np.pi
    rmse_deg = float(np.sqrt(np.mean(abs_err ** 2))) * 180.0 / np.pi

    pos_errs, ori_errs = [], []
    fk = load_ik()  # 用于 FK 精度评估
    for i in range(n_total):
        ep = compute_ee_pose(fk, q_preds[i])
        pos_errs.append(np.linalg.norm(ep[:3] - ee_data[i, :3]))
        ori_errs.append(np.linalg.norm(ep[3:6] - ee_data[i, 3:6]))

    print(f"\n自回归精度{' + FK修正' if args.fkfix else ''}:")
    print(f"  Joint MAE:  {mae_deg:.3f}° ({np.mean(abs_err)*1000:.2f} mrad)")
    print(f"  Joint RMSE: {rmse_deg:.3f}°")
    print(f"  FK Pos:     {np.mean(pos_errs)*1000:.2f} mm")
    print(f"  FK Ori:     {np.mean(ori_errs)*1000:.2f} mrad")
    if args.fkfix:
        print(f"  FK修正耗时: {fkfix_time/n_total*1e6:.1f} μs/帧")

    # ── 保存结果 ──
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        meta_cols = ["episode_index", "frame_index", "timestamp", "gripper_L", "gripper_R"]
        meta_dfs = []
        if args.count:
            meta_dfs.append(pd.read_parquet(files[0])[meta_cols].iloc[:args.count])
        else:
            for f in files:
                meta_dfs.append(pd.read_parquet(f)[meta_cols])
        meta = pd.concat(meta_dfs, ignore_index=True)

        jn = JOINT_COLS
        out_df = pd.DataFrame(index=range(n_total))
        out_df["episode_index"] = meta["episode_index"].values
        out_df["frame_index"]   = meta["frame_index"].values
        out_df["timestamp"]     = meta["timestamp"].values
        for i, j in enumerate(jn):
            out_df[j] = q_preds[:, i]
        out_df["gripper_L"] = meta["gripper_L"].values
        out_df["gripper_R"] = meta["gripper_R"].values

        out_path = os.path.join(args.save, "predictions_ar.parquet")
        out_df.to_parquet(out_path, index=False)
        print(f"\n保存: {out_path}  ({n_total} 帧)")

    print("\nDone.")


if __name__ == "__main__":
    main()
