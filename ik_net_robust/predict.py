#!/usr/bin/env python3
"""自回归推理 + 精度评估 + 结果保存。

自回归模式：当前步的预测关节角作为下一步的 prev_joints 输入，模拟实际部署场景。

用法:
  conda activate actibot_sdk
  python ik_net_robust/predict.py                        # 自回归精度 + 速度
  python ik_net_robust/predict.py --fkfix                # 启用 FK 修正（每帧）
  python ik_net_robust/predict.py --fkfix --fkfix-step 5  # 每 5 帧修正一次
  python ik_net_robust/predict.py --data /path/to/data   # 指定数据集
  python ik_net_robust/predict.py --save ./results       # 保存预测结果
  python ik_net_robust/predict.py --time-only            # 只测速度
  python ik_net_robust/predict.py --count 1000           # 只测前 1000 帧

  python ik_net_robust/predict.py --data data/0602_test_for_net_action_fk --save ik_net_robust/save/0602_test_for_net_action_fk_results
  python ik_net_robust/predict.py --fkfix --fkfix-step 10 --data data/0602_test_for_net_action_fk --save ik_net_robust/save/0602_test_for_net_action_fk_results_correct_10
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
from fk_utils import load_ik, compute_ee_pose, fk_correction

EE_COLS = data_config["col_eeL"] + data_config["col_eeR"]
JOINT_COLS = data_config["col_joints_l"] + data_config["col_joints_r"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-only", action="store_true", help="只测推理速度")
    parser.add_argument("--count", type=int, default=None, help="只处理前 N 帧")
    parser.add_argument("--fkfix", action="store_true", help="启用 FK 修正")
    parser.add_argument("--fkfix-step", type=int, default=1, help="FK 修正间隔帧数（默认1=每帧修正，5=每5帧修正一次）")
    parser.add_argument("--data", default=None, help="数据集路径")
    parser.add_argument("--save", default=None, help="保存预测结果到指定目录（parquet 格式）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {hp['input_dim']}D → {hp['output_dim']}D, residual={hp['use_residual']}")

    data_dir = args.data or paths.get("test_data_dir") or paths["data_dir"]

    ckpt_path = os.path.join(paths["results_dir"], "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"错误: 未找到模型 {ckpt_path}"); return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    with open(os.path.join(paths["results_dir"], "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    model = ResidualMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ik = load_ik()
    print(f"模型加载完成 (epoch {ckpt['epoch']})")

    files = sorted(glob.glob(os.path.join(data_dir, "episode_*_fk.parquet")))
    if not files:
        print(f"未找到数据: {data_dir}"); return

    if args.count:
        df = pd.read_parquet(files[0])
        ee_data = df[EE_COLS].values.astype(np.float64)[:args.count]
        joint_data = df[JOINT_COLS].values.astype(np.float64)[:args.count]
    else:
        frames = [pd.read_parquet(f) for f in files]
        ee_data = np.vstack([f[EE_COLS].values.astype(np.float64) for f in frames])
        joint_data = np.vstack([f[JOINT_COLS].values.astype(np.float64) for f in frames])

    n_total = len(ee_data)
    print(f"数据: {n_total} 帧")

    # 构造 26D 输入
    joints_prev = np.vstack([joint_data[0:1], joint_data[:-1]])
    X_raw = np.hstack([ee_data, joints_prev])
    X_norm = scaler.transform_X(X_raw)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)

    # warmup
    for _ in range(100):
        _ = model(X_tensor[:256])

    # 测速
    batch_sizes = [1, 64, 256, 1024, n_total]
    print(f"\n{'Batch':>6} | {'推理时间':>10} | {'帧/秒':>10} | {'单帧(μs)':>10}")
    print("-" * 45)
    for bs in batch_sizes:
        if bs > n_total: continue
        n_batch = n_total // bs
        X_b = X_tensor[:n_batch * bs].reshape(n_batch, bs, -1)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(n_batch):
                _ = model(X_b[i])
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - t0
        avg_us = elapsed / (n_batch * bs) * 1e6
        print(f"{bs:>6d} | {elapsed*1000:>8.2f} ms | {n_batch*bs/elapsed:>8.0f} | {avg_us:>8.2f}")

    if args.time_only:
        return

    # 精度评估（自回归模式）
    print(f"\n{'='*50}")
    fkfix_label = f" + FK修正(每{args.fkfix_step}帧)" if args.fkfix and args.fkfix_step > 1 else (" + FK修正" if args.fkfix else "")
    print(f"精度评估 (自回归模式{fkfix_label})")
    print('='*50)

    q_preds = np.zeros_like(joint_data)
    prev = joint_data[0].copy()
    fk_time = 0

    for t in range(n_total):
        ee = ee_data[t:t+1]
        prev_use = joint_data[0:1] if t == 0 else prev.reshape(1, -1)
        X = np.hstack([ee, prev_use])
        X_n = scaler.transform_X(X)
        with torch.no_grad():
            pn = model(torch.tensor(X_n, dtype=torch.float32, device=device)).cpu().numpy()
        q_pred = scaler.inverse_y(pn)[0]

        if args.fkfix and t > 0 and (t % args.fkfix_step == 0):
            t0 = time.perf_counter()
            qL, qR, _, _ = fk_correction(ik, q_pred, ee[0])
            fk_time += time.perf_counter() - t0
            q_pred[:7] = qL
            q_pred[7:] = qR

        q_preds[t] = q_pred
        prev = q_pred.copy()

    abs_err = np.abs(q_preds - joint_data)
    mae_deg = float(np.mean(abs_err)) * 180.0 / np.pi
    rmse_deg = float(np.sqrt(np.mean(abs_err ** 2))) * 180.0 / np.pi

    pos_errs, ori_errs = [], []
    for i in range(n_total):
        ep = compute_ee_pose(ik, q_preds[i])
        pos_errs.append(np.linalg.norm(ep[:3] - ee_data[i, :3]))
        ori_errs.append(np.linalg.norm(ep[3:6] - ee_data[i, 3:6]))

    print(f"  Joint MAE:  {mae_deg:.3f}° ({np.mean(abs_err)*1000:.2f} mrad)")
    print(f"  Joint RMSE: {rmse_deg:.3f}°")
    print(f"  R²:         {max(0, 1 - np.sum(abs_err**2) / np.sum((joint_data - joint_data.mean(0))**2)):.4f}")
    print(f"  FK Pos:     {np.mean(pos_errs)*1000:.2f} mm (max {np.max(pos_errs)*1000:.2f})")
    print(f"  FK Ori:     {np.mean(ori_errs)*1000:.2f} mrad (max {np.max(ori_errs)*1000:.2f})")
    if args.fkfix:
        print(f"  FK修正耗时: {fk_time/n_total*1e6:.1f} μs/帧 (共 {fk_time*1000:.1f} ms)")

    # 各关节 MAE
    jn = JOINT_COLS
    per_joint = np.mean(abs_err, axis=0) * 180.0 / np.pi
    print(f"\n各关节 MAE (°):")
    for name, err in zip(jn, per_joint):
        print(f"  {name:>15s}: {err:.3f}")

    # ── 保存预测结果 ──
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        jn = data_config["col_joints_l"] + data_config["col_joints_r"]

        meta_cols = ["episode_index", "frame_index", "timestamp", "gripper_L", "gripper_R"]
        meta_dfs = []
        if args.count:
            meta_dfs.append(pd.read_parquet(files[0])[meta_cols].iloc[:args.count])
        else:
            for f in files:
                meta_dfs.append(pd.read_parquet(f)[meta_cols])
        meta = pd.concat(meta_dfs, ignore_index=True)

        out_df = pd.DataFrame(index=range(n_total))
        out_df["episode_index"] = meta["episode_index"].values
        out_df["frame_index"]   = meta["frame_index"].values
        out_df["timestamp"]     = meta["timestamp"].values
        for i, j in enumerate(jn):
            out_df[j] = q_preds[:, i]
        if "gripper_L" in meta.columns and "gripper_R" in meta.columns:
            out_df["gripper_L"] = meta["gripper_L"].values
            out_df["gripper_R"] = meta["gripper_R"].values

        out_path = os.path.join(args.save, "predictions.parquet")
        out_df.to_parquet(out_path, index=False)
        print(f"\n预测结果: {out_path}  ({n_total} 帧)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
