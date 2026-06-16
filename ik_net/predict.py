#!/usr/bin/env python3
"""推理测试：加载训练好的模型，测量推理速度和精度，保存预测结果。

用法:
  conda activate actibot_sdk
  python ik_net/predict.py                          # 完整测试集评估
  python ik_net/predict.py --time-only              # 只测推理速度
  python ik_net/predict.py --count 1000             # 只测前 1000 帧
  python ik_net/predict.py --data /path/to/data --save ./results  # 指定数据+保存结果
  python ik_net/predict.py --data data/0602_test_for_net_action_fk --save ik_net/save/0616_test_for_net_action_fk_results_moredata
"""
import os
import sys
import time
import argparse
import pickle
import numpy as np
import torch

_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_example_dir = os.path.join(_project_dir, "example")
for p in [_example_dir, _project_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import hp, paths, data_config
from model import ResidualMLP
from fk_utils import load_ik, compute_ee_pose

EE_COLS = data_config["col_eeL"] + data_config["col_eeR"]
JOINT_COLS = data_config["col_joints_l"] + data_config["col_joints_r"]


def main():
    parser = argparse.ArgumentParser(description="IK-Net 推理测试")
    parser.add_argument("--time-only", action="store_true", help="只测推理速度，不评估精度")
    parser.add_argument("--count", type=int, default=None, help="只处理前 N 帧")
    parser.add_argument("--data", default=None, help="数据文件夹路径（默认: config 中的 data_dir）")
    parser.add_argument("--save", default=None, help="保存预测结果到指定目录（parquet 格式）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {hp['input_dim']}D → {hp['output_dim']}D, "
          f"hidden={hp['hidden_dims']}, residual={hp['use_residual']}")

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

    # ── 加载数据 ──
    import glob
    import pandas as pd
    data_dir = args.data or paths["data_dir"]
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*_fk.parquet")))

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

    # 构造 26D 输入: ee_12 + prev_joints_14
    joints_prev = np.vstack([joint_data[0:1], joint_data[:-1]])
    X_raw = np.hstack([ee_data, joints_prev])  # (n, 26)

    X_norm = scaler.transform_X(X_raw)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)

    # warmup
    for _ in range(100):
        _ = model(X_tensor[:256])

    # ── 批量推理测速 ──
    batch_sizes = [1, 64, 256, 1024, n_total]
    print(f"\n{'Batch':>6} | {'推理时间':>10} | {'帧/秒':>10} | {'单帧(μs)':>10}")
    print("-" * 45)

    for bs in batch_sizes:
        if bs > n_total:
            continue
        n_batch = n_total // bs
        X_batch = X_tensor[:n_batch * bs].reshape(n_batch, bs, -1)

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(n_batch):
                _ = model(X_batch[i])
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - t0

        avg_us = elapsed / (n_batch * bs) * 1e6
        fps = (n_batch * bs) / elapsed
        print(f"{bs:>6d} | {elapsed*1000:>8.2f} ms | {fps:>8.0f} | {avg_us:>8.2f}")

    # ── 精度评估 ──
    if not args.time_only:
        print(f"\n{'='*50}")
        print("精度评估")
        print('='*50)

        ik = load_ik()

        with torch.no_grad():
            pred_norm = model(X_tensor).cpu().numpy()
        q_pred = scaler.inverse_y(pred_norm)
        q_gt = joint_data

        # 关节角误差
        abs_err = np.abs(q_pred - q_gt)
        joint_mae_deg = float(np.mean(abs_err)) * 180.0 / np.pi
        joint_rmse_deg = float(np.sqrt(np.mean(abs_err ** 2))) * 180.0 / np.pi
        joint_max_deg = float(np.max(abs_err)) * 180.0 / np.pi

        print(f"\n关节角精度:")
        print(f"  MAE:  {joint_mae_deg:.3f}° ({np.mean(abs_err)*1000:.2f} mrad)")
        print(f"  RMSE: {joint_rmse_deg:.3f}°")
        print(f"  Max:  {joint_max_deg:.3f}°")
        print(f"  R²:   {max(0, 1 - np.sum(abs_err**2) / np.sum((q_gt - q_gt.mean(0))**2)):.4f}")

        # 各关节 MAE
        jn = data_config["col_joints_l"] + data_config["col_joints_r"]
        per_joint = np.mean(abs_err, axis=0) * 180.0 / np.pi
        print(f"\n各关节 MAE (°):")
        for name, err in zip(jn, per_joint):
            print(f"  {name:>15s}: {err:.3f}")

        # FK 重算末端误差
        N_test = min(n_total, 10000)
        pos_errs, ori_errs = [], []
        for i in range(N_test):
            ee_pred = compute_ee_pose(ik, q_pred[i])
            # 使用真实 ee_target
            pos_errs.append(np.linalg.norm(ee_pred[:3] - ee_data[i, :3]))
            pos_errs.append(np.linalg.norm(ee_pred[6:9] - ee_data[i, 6:9]))
            ori_errs.append(np.linalg.norm(ee_pred[3:6] - ee_data[i, 3:6]))
            ori_errs.append(np.linalg.norm(ee_pred[9:12] - ee_data[i, 9:12]))

        print(f"\n末端位姿精度 (FK 重算, {N_test} 帧):")
        print(f"  位置误差: {np.mean(pos_errs)*1000:.2f} mm (max {np.max(pos_errs)*1000:.2f})")
        print(f"  姿态误差: {np.mean(ori_errs)*1000:.2f} mrad (max {np.max(ori_errs)*1000:.2f})")

    # ── 保存预测结果 ──
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        import pandas as pd
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
            out_df[j] = q_pred[:, i]
        # 从原数据继承夹爪信号
        if "gripper_L" in meta.columns and "gripper_R" in meta.columns:
            out_df["gripper_L"] = meta["gripper_L"].values
            out_df["gripper_R"] = meta["gripper_R"].values

        out_path = os.path.join(args.save, "predictions.parquet")
        out_df.to_parquet(out_path, index=False)
        print(f"\n预测结果: {out_path}  ({n_total} 帧)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
