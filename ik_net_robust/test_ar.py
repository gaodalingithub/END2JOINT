#!/usr/bin/env python3
"""测试自回归推理精度，含可选 FK 修正。

用法:
  conda activate actibot_sdk
  python ik_net_robust/test_ar.py                           # 默认数据
  python ik_net_robust/test_ar.py --data my_dataset_groot_fk_results
  python ik_net_robust/test_ar.py --fkfix                   # 启用 FK 修正
  python ik_net_robust/test_ar.py --fkfix --data my_dataset_groot_fk_results
"""
import sys, os, argparse, pickle, time
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
from model import ResidualMLP
from fk_utils import load_ik, compute_ee_pose, fk_correction

EE_COLS = data_config["col_eeL"] + data_config["col_eeR"]
JOINT_COLS = data_config["col_joints_l"] + data_config["col_joints_r"]
DEFAULT_DATA = paths["test_data_dir"]


def run_episode(df, model, scaler, device, ik, mode, use_fkfix=False):
    """
    mode: "gt"=真实prev, "ar"=自回归, "zero"=零prev
    use_fkfix: 是否对每帧 MLP 输出做 FK 修正
    """
    n = len(df)
    ee_data = df[EE_COLS].values.astype(np.float64)
    q_gt = df[JOINT_COLS].values.astype(np.float64)
    q_preds = np.zeros_like(q_gt)
    mlp_time = 0
    fk_time = 0

    for t in range(n):
        ee = ee_data[t:t+1]
        if t == 0:
            prev = q_gt[0:1]
        elif mode == "gt":
            prev = q_gt[t-1:t]
        elif mode == "ar":
            prev = q_preds[t-1:t]
        elif mode == "zero":
            prev = np.zeros((1, 14))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        X = np.hstack([ee, prev])
        X_norm = scaler.transform_X(X)
        x_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            pn = model(x_tensor).cpu().numpy()
        torch.cuda.synchronize() if device.type == "cuda" else None
        mlp_time += time.perf_counter() - t0
        q_pred = scaler.inverse_y(pn)[0]

        if use_fkfix and t > 0:
            t0 = time.perf_counter()
            qL, qR, _, _ = fk_correction(ik, q_pred, ee[0])
            fk_time += time.perf_counter() - t0
            q_pred[:7] = qL
            q_pred[7:] = qR

        q_preds[t] = q_pred

    abs_err = np.abs(q_preds - q_gt)
    mae = np.mean(abs_err) * 180 / np.pi
    rmse = np.sqrt(np.mean(abs_err ** 2)) * 180 / np.pi

    pos_errs, ori_errs = [], []
    for i in range(n):
        ep = compute_ee_pose(ik, q_preds[i])
        pos_errs.append(np.linalg.norm(ep[:3] - ee_data[i, :3]))
        ori_errs.append(np.linalg.norm(ep[3:6] - ee_data[i, 3:6]))

    avg_fk_time = fk_time / max(n - 1, 1) * 1e6 if use_fkfix else 0
    return {"mae": mae, "rmse": rmse, "fk_pos": np.mean(pos_errs) * 1000,
            "fk_ori": np.mean(ori_errs) * 1000, "fk_time_us": avg_fk_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="数据集路径")
    parser.add_argument("--n", type=int, default=3, help="测试前 N 个 episode")
    parser.add_argument("--fkfix", action="store_true", help="启用 FK 修正")
    args = parser.parse_args()

    data_dir = args.data or DEFAULT_DATA
    if not os.path.exists(data_dir):
        print(f"数据集不存在: {data_dir}"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(paths["results_dir"], "best_model.pt"),
                      map_location=device, weights_only=False)
    with open(os.path.join(paths["results_dir"], "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    model = ResidualMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ik = load_ik()

    print(f"模型: epoch {ckpt['epoch']} | {'FK修正' if args.fkfix else '无FK修正'}")

    # 测试模式
    modes = [("gt", "真实prev"), ("ar", "自回归"), ("zero", "零基线")]
    if args.fkfix:
        modes.append(("ar", "自回归+FK修正"))

    print(f"\n{'Episode':>6} | {'模式':<14} | {'Joint MAE':>10} | {'RMSE':>8} | {'FK Pos':>9} | {'FK Ori':>9} | {'耗时':>8}")
    print("-" * 75)

    for ep in range(args.n):
        fpath = os.path.join(data_dir, f"episode_{ep:06d}_fk.parquet")
        if not os.path.exists(fpath):
            continue
        df = pd.read_parquet(fpath)
        for mode_key, mode_label in modes:
            use_fk = (mode_label == "自回归+FK修正")
            r = run_episode(df, model, scaler, device, ik, mode_key, use_fkfix=use_fk)
            t_str = f"{r['fk_time_us']:.0f}μs" if r['fk_time_us'] > 0 else "—"
            print(f"  {ep:>4d}  | {mode_label:<14} {r['mae']:>8.3f}° {r['rmse']:>8.3f}° "
                  f"{r['fk_pos']:>8.2f}mm {r['fk_ori']:>8.2f}mrad {t_str:>8}")

    print("\nDone.")


if __name__ == "__main__":
    main()
