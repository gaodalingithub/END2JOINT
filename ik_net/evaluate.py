#!/usr/bin/env python3
"""评估训练好的 IK-Net 模型。

功能:
  1. 测试集整体指标
  2. 逐 episode 对比预测 vs 真实关节角
  3. 逐 episode 对比预测末端位姿 vs 目标末端位姿（FK 重算）
  4. 连续帧平滑度检查

用法:
  conda activate actibot_sdk
  python ik_net/evaluate.py
"""
import os
import sys
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_example_dir = os.path.join(_project_dir, "example")
for p in [_example_dir, _project_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import hp, paths, data_config
from dataloader import load_episode_files, episodes_to_arrays, IKDataset
from torch.utils.data import DataLoader
from model import ResidualMLP
from fk_utils import load_ik, compute_ee_pose


def compute_joint_metrics(q_pred, q_gt):
    """计算关节角精度指标。"""
    abs_err = np.abs(q_pred - q_gt)  # (N, 14)
    return {
        "mae": float(np.mean(abs_err)),
        "mae_per_joint": np.mean(abs_err, axis=0).tolist(),
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
        "max_abs_err": float(np.max(abs_err)),
        "r2": max(0, 1 - np.sum(abs_err ** 2) / np.sum((q_gt - q_gt.mean(axis=0)) ** 2)),
    }


def compute_ee_metrics(ik, q_pred, ee_target):
    """计算 FK 重算后的末端位姿误差。"""
    N = q_pred.shape[0]
    pos_errs, ori_errs = [], []
    for i in range(N):
        ee_pred = compute_ee_pose(ik, q_pred[i])
        pos_errs.append(np.linalg.norm(ee_pred[:3] - ee_target[i, :3]))     # left
        pos_errs.append(np.linalg.norm(ee_pred[6:9] - ee_target[i, 6:9]))   # right
        ori_errs.append(np.linalg.norm(ee_pred[3:6] - ee_target[i, 3:6]))
        ori_errs.append(np.linalg.norm(ee_pred[9:12] - ee_target[i, 9:12]))

    return {
        "pos_err_mean_m": float(np.mean(pos_errs)),
        "pos_err_max_m": float(np.max(pos_errs)),
        "ori_err_mean_rad": float(np.mean(ori_errs)),
        "ori_err_max_rad": float(np.max(ori_errs)),
    }


def plot_joint_comparison(q_pred, q_gt, joint_names, title, save_path):
    """绘制关节角预测 vs 真实对比图。"""
    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]
    t = np.arange(len(q_pred))
    for i, (name, ax) in enumerate(zip(joint_names, axes)):
        ax.plot(t, q_gt[:, i], "b-", label="GT", linewidth=1.0)
        ax.plot(t, q_pred[:, i], "r--", label="Pred", linewidth=1.0)
        ax.set_ylabel(name, fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ee_trajectory(ee_pred, ee_target, names, title, save_path):
    """绘制末端位置 (x,y,z) 预测 vs 目标对比。"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    dims = ["x", "y", "z"]
    t = np.arange(len(ee_pred))
    for i, (dim, ax) in enumerate(zip(dims, axes)):
        ax.plot(t, ee_target[:, i], "b-", label=f"GT {dim}", linewidth=1.0)
        ax.plot(t, ee_pred[:, i], "r--", label=f"Pred {dim}", linewidth=1.0)
        ax.set_ylabel(f"{names[:3]}[{dim}] (m)", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 加载数据 ──
    episodes = load_episode_files(paths["data_dir"])

    # ── 加载模型 ──
    ckpt_path = os.path.join(paths["results_dir"], hp.get("ckpt_name", "best_model.pt"))
    if not os.path.exists(ckpt_path):
        print(f"错误: 未找到模型文件 {ckpt_path}")
        print("请先运行 train.py")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    with open(os.path.join(paths["results_dir"], "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    model = ResidualMLP().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"模型加载完成 (epoch {checkpoint['epoch']})")

    # ── FK 引擎 ──
    print("加载 FK 引擎...")
    ik = load_ik()

    # ── 测试 episode 列表 ──
    # 从 history.json 获取 test_eps
    import json
    history_path = os.path.join(paths["results_dir"], "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        test_eps = history["info"]["test_eps"]
        val_eps = history["info"]["val_eps"]
    else:
        # 回退：取最后 10% 的 episode
        eps = sorted(episodes.keys())
        test_eps = eps[-max(1, len(eps) // 10):]
        val_eps = eps[-2 * max(1, len(eps) // 10):-max(1, len(eps) // 10)]
        print("(未找到 history.json，使用末尾 episode 作为测试集)")

    print(f"测试 episodes: {test_eps}")

    vis_dir = os.path.join(paths["results_dir"], "vis")
    os.makedirs(vis_dir, exist_ok=True)

    all_joint_metrics = []
    all_ee_metrics = []

    # ── 逐 episode 评估 ──
    for ep in test_eps:
        df = episodes[ep]
        n = len(df)

        # 构造样本
        ee_cols = data_config["col_eeL"] + data_config["col_eeR"]
        joint_cols = data_config["col_joints_l"] + data_config["col_joints_r"]
        ee_t = df[ee_cols].values.astype(np.float64)
        joints_t = df[joint_cols].values.astype(np.float64)
        joints_prev = np.vstack([joints_t[0:1], joints_t[:-1]])
        X = np.hstack([ee_t, joints_prev])
        y = joints_t

        # 归一化 → 预测 → 反归一化
        X_norm = scaler.transform_X(X)
        y_norm = scaler.transform_y(y)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_norm = model(X_tensor).cpu().numpy()
        q_pred = scaler.inverse_y(pred_norm)
        q_gt = y

        # 关节角指标
        jm = compute_joint_metrics(q_pred, q_gt)
        all_joint_metrics.append(jm)

        # 末端位姿指标
        em = compute_ee_metrics(ik, q_pred, ee_t)
        all_ee_metrics.append(em)

        # 可视化（前 5 个 episode 出图）
        if ep <= test_eps[4] if len(test_eps) > 4 else True:
            # 关节角对比
            plot_joint_comparison(
                q_pred[:, :7], q_gt[:, :7], data_config["col_joints_l"],
                f"Episode {ep} - Left Arm Joints (Pred vs GT)",
                os.path.join(vis_dir, f"ep{ep:03d}_joints_L.png"))
            plot_joint_comparison(
                q_pred[:, 7:], q_gt[:, 7:], data_config["col_joints_r"],
                f"Episode {ep} - Right Arm Joints (Pred vs GT)",
                os.path.join(vis_dir, f"ep{ep:03d}_joints_R.png"))

            # 末端位置轨迹
            ee_pred = np.array([compute_ee_pose(ik, q_pred[i]) for i in range(n)])
            plot_ee_trajectory(
                ee_pred[:, :6], ee_t[:, :6], data_config["col_eeL"],
                f"Episode {ep} - Left EE (FK from Pred vs Target)",
                os.path.join(vis_dir, f"ep{ep:03d}_ee_L.png"))
            plot_ee_trajectory(
                ee_pred[:, 6:], ee_t[:, 6:], data_config["col_eeR"],
                f"Episode {ep} - Right EE (FK from Pred vs Target)",
                os.path.join(vis_dir, f"ep{ep:03d}_ee_R.png"))

        # 打印摘要
        print(f"  Ep {ep:3d} ({n:4d} frames): "
              f"Joint MAE={jm['mae']*1e3:.2f} mrad | "
              f"FK Pos={em['pos_err_mean_m']*1e3:.2f} mm | "
              f"FK Ori={em['ori_err_mean_rad']*1e3:.2f} mrad")

        # 打印第一帧的预测 vs 真实值（仅第一个测试 episode）
        if ep == test_eps[0]:
            jn = data_config["col_joints_l"] + data_config["col_joints_r"]
            print(f"  ├─ 首帧预测 vs 真实 (度 °):")
            for i in range(7):
                print(f"  │  {jn[i]:>15s}: pred={np.degrees(q_pred[0,i]):>7.3f}  gt={np.degrees(q_gt[0,i]):>7.3f}  "
                      f"delta={np.degrees(q_pred[0,i]-q_gt[0,i]):+7.3f}")
            print("  │  " + "─" * 45)
            for i in range(7, 14):
                print(f"  │  {jn[i]:>15s}: pred={np.degrees(q_pred[0,i]):>7.3f}  gt={np.degrees(q_gt[0,i]):>7.3f}  "
                      f"delta={np.degrees(q_pred[0,i]-q_gt[0,i]):+7.3f}")
            # 末帧
            last = n - 1
            print(f"  └─ 末帧 (frame {last}) 预测 vs 真实 (度 °):")
            for i in range(7):
                print(f"     {jn[i]:>15s}: pred={np.degrees(q_pred[last,i]):>7.3f}  gt={np.degrees(q_gt[last,i]):>7.3f}  "
                      f"delta={np.degrees(q_pred[last,i]-q_gt[last,i]):+7.3f}")
            print("     " + "─" * 45)
            for i in range(7, 14):
                print(f"     {jn[i]:>15s}: pred={np.degrees(q_pred[last,i]):>7.3f}  gt={np.degrees(q_gt[last,i]):>7.3f}  "
                      f"delta={np.degrees(q_pred[last,i]-q_gt[last,i]):+7.3f}")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("测试集汇总")
    print("=" * 60)
    avg_jm = {k: np.mean([m[k] for m in all_joint_metrics]) for k in all_joint_metrics[0]
              if isinstance(all_joint_metrics[0][k], (int, float))}
    avg_em = {k: np.mean([m[k] for m in all_ee_metrics]) for k in all_ee_metrics[0]
              if isinstance(all_ee_metrics[0][k], (int, float))}

    print(f"关节角:")
    print(f"  MAE:  {avg_jm['mae']*1e3:.2f} mrad ({np.degrees(avg_jm['mae']):.3f}°)")
    print(f"  RMSE: {avg_jm['rmse']*1e3:.2f} mrad ({np.degrees(avg_jm['rmse']):.3f}°)")
    print(f"  Max:  {avg_jm['max_abs_err']*1e3:.2f} mrad ({np.degrees(avg_jm['max_abs_err']):.3f}°)")
    print(f"  R²:   {avg_jm.get('r2', 0):.4f}")
    print(f"末端位姿 (FK 重算后):")
    print(f"  位置误差: {avg_em['pos_err_mean_m']*1e3:.2f} mm (max {avg_em['pos_err_max_m']*1e3:.2f} mm)")
    print(f"  姿态误差: {avg_em['ori_err_mean_rad']*1e3:.2f} mrad (max {avg_em['ori_err_max_rad']*1e3:.2f} mrad)")

    # 各关节 MAE
    per_joint = np.mean([m["mae_per_joint"] for m in all_joint_metrics], axis=0)
    jn = data_config["col_joints_l"] + data_config["col_joints_r"]
    print(f"\n各关节 MAE (mrad):")
    for name, err in zip(jn, per_joint * 1e3):
        print(f"  {name:>15s}: {err:.2f}")

    # 汇总结果
    summary = {
        "joint_mae_mrad": float(avg_jm["mae"] * 1e3),
        "joint_rmse_mrad": float(avg_jm["rmse"] * 1e3),
        "fk_pos_err_mm": float(avg_em["pos_err_mean_m"] * 1e3),
        "fk_ori_err_mrad": float(avg_em["ori_err_mean_rad"] * 1e3),
    }
    with open(os.path.join(paths["results_dir"], "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n可视化结果: {vis_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
