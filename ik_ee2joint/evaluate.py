#!/usr/bin/env python3
"""评估训练好的 IK-MLP 模型。"""
import os
import sys
import pickle
import json
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
from dataloader import load_episode_files, EE_COLS, JOINT_COLS
from model import IKMLP
from fk_utils import load_ik, compute_ee_pose

EE_TARGET_COLS = data_config["col_eeL"] + data_config["col_eeR"]


def compute_joint_metrics(q_pred, q_gt):
    abs_err = np.abs(q_pred - q_gt)
    return {"mae": float(np.mean(abs_err)),
            "mae_per_joint": np.mean(abs_err, axis=0).tolist(),
            "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
            "max_abs_err": float(np.max(abs_err)),
            "r2": max(0, 1 - np.sum(abs_err ** 2) / np.sum((q_gt - q_gt.mean(0)) ** 2))}


def compute_ee_metrics(ik, q_pred, ee_target):
    N = q_pred.shape[0]
    pos_errs, ori_errs = [], []
    for i in range(N):
        ee_pred = compute_ee_pose(ik, q_pred[i])
        pos_errs.append(np.linalg.norm(ee_pred[:3] - ee_target[i, :3]))
        pos_errs.append(np.linalg.norm(ee_pred[6:9] - ee_target[i, 6:9]))
        ori_errs.append(np.linalg.norm(ee_pred[3:6] - ee_target[i, 3:6]))
        ori_errs.append(np.linalg.norm(ee_pred[9:12] - ee_target[i, 9:12]))
    return {"pos_err_mean_m": float(np.mean(pos_errs)),
            "pos_err_max_m": float(np.max(pos_errs)),
            "ori_err_mean_rad": float(np.mean(ori_errs)),
            "ori_err_max_rad": float(np.max(ori_errs))}


def plot_joints(q_pred, q_gt, names, title, path):
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
    if n == 1: axes = [axes]
    t = np.arange(len(q_pred))
    for i, (nm, ax) in enumerate(zip(names, axes)):
        ax.plot(t, q_gt[:, i], "b-", label="GT", lw=1)
        ax.plot(t, q_pred[:, i], "r--", label="Pred", lw=1)
        ax.set_ylabel(nm, fontsize=9); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    fig.suptitle(title, fontsize=13); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close(fig)


def plot_ee(ee_pred, ee_target, title, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(ee_pred))
    for i, (dim, ax) in enumerate(zip(["x", "y", "z"], axes)):
        ax.plot(t, ee_target[:, i], "b-", label="GT", lw=1)
        ax.plot(t, ee_pred[:, i], "r--", label="Pred", lw=1)
        ax.set_ylabel(f"{dim} (m)", fontsize=9); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    fig.suptitle(title, fontsize=13); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    episodes = load_episode_files(paths["data_dir"])

    ckpt_path = os.path.join(paths["results_dir"], "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"模型 {ckpt_path} 不存在"); return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    with open(os.path.join(paths["results_dir"], "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    model = IKMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"模型加载 (epoch {ckpt['epoch']})")

    ik = load_ik()
    hist_path = os.path.join(paths["results_dir"], "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            test_eps = json.load(f)["info"]["test_eps"]
    else:
        eps = sorted(episodes.keys())
        test_eps = eps[-max(1, len(eps) // 10):]
    print(f"测试: {test_eps}")

    vis_dir = os.path.join(paths["results_dir"], "vis")
    os.makedirs(vis_dir, exist_ok=True)
    all_jm, all_em = [], []

    for ep in test_eps:
        df = episodes[ep]
        ee_t = df[EE_COLS].values.astype(np.float64)
        joints_t = df[JOINT_COLS].values.astype(np.float64)

        X_norm = scaler.transform_X(ee_t)
        with torch.no_grad():
            pred_norm = model(torch.tensor(X_norm, dtype=torch.float32, device=device)).cpu().numpy()
        q_pred = scaler.inverse_y(pred_norm)
        q_gt = joints_t
        jm = compute_joint_metrics(q_pred, q_gt)
        all_jm.append(jm)

        ee_target_full = df[EE_TARGET_COLS].values.astype(np.float64)
        em = compute_ee_metrics(ik, q_pred, ee_target_full)
        all_em.append(em)

        if ep <= test_eps[4] if len(test_eps) > 4 else True:
            n_j = hp["output_dim"]
            names = data_config["joint_names_l"] + data_config["joint_names_r"]
            plot_joints(q_pred[:, :min(7, n_j)], q_gt[:, :min(7, n_j)],
                        names[:min(7, n_j)], f"Ep{ep} Joints (L)",
                        os.path.join(vis_dir, f"ep{ep:03d}_joints_L.png"))
            if n_j > 7:
                plot_joints(q_pred[:, 7:n_j], q_gt[:, 7:n_j],
                           names[7:n_j], f"Ep{ep} Joints (R)",
                           os.path.join(vis_dir, f"ep{ep:03d}_joints_R.png"))
            ee_pred = np.array([compute_ee_pose(ik, q_pred[i]) for i in range(len(df))])
            plot_ee(ee_pred[:, :6], ee_target_full[:, :6],
                    f"Ep{ep} Left EE", os.path.join(vis_dir, f"ep{ep:03d}_ee_L.png"))
            plot_ee(ee_pred[:, 6:], ee_target_full[:, 6:],
                    f"Ep{ep} Right EE", os.path.join(vis_dir, f"ep{ep:03d}_ee_R.png"))

        print(f"  Ep {ep:3d} ({len(df):4d}f): MAE={jm['mae']*1e3:.2f}mrad "
              f"FK Pos={em['pos_err_mean_m']*1e3:.2f}mm "
              f"FK Ori={em['ori_err_mean_rad']*1e3:.2f}mrad")

    # 汇总
    print("\n" + "=" * 50)
    avg_jm = {k: np.mean([m[k] for m in all_jm]) for k in all_jm[0]
              if isinstance(all_jm[0][k], (int, float))}
    avg_em = {k: np.mean([m[k] for m in all_em]) for k in all_em[0]
              if isinstance(all_em[0][k], (int, float))}
    print(f"Joint MAE: {avg_jm['mae']*1e3:.2f}mrad ({np.degrees(avg_jm['mae']):.3f}°)")
    print(f"FK Pos: {avg_em['pos_err_mean_m']*1e3:.2f}mm")
    print(f"FK Ori: {avg_em['ori_err_mean_rad']*1e3:.2f}mrad")

    with open(os.path.join(paths["results_dir"], "evaluation_summary.json"), "w") as f:
        json.dump({"joint_mae_mrad": avg_jm['mae'] * 1e3,
                   "fk_pos_err_mm": avg_em['pos_err_mean_m'] * 1e3}, f, indent=2)
    print(f"Vis: {vis_dir}/")


if __name__ == "__main__":
    main()
