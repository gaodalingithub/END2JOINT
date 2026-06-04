#!/usr/bin/env python3
"""训练 IK-MLP。"""
import os
import sys
import time
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_example_dir = os.path.join(_project_dir, "example")
for p in [_example_dir, _project_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import hp, paths
from dataloader import build_dataloaders
from model import IKMLP
from fk_utils import load_ik, batch_fk_error


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(loader, model, scaler, device, ik=None):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    n_batches = 0
    all_q_pred, all_y, all_X = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            total_loss += criterion(pred, y_batch).item()
            n_batches += 1
            all_q_pred.append(pred.cpu().numpy())
            all_y.append(y_batch.cpu().numpy())
            all_X.append(X_batch.cpu().numpy())
    result = {"loss": total_loss / max(n_batches, 1)}

    # 关节角平均误差（度）
    y_norm = np.vstack(all_y)
    q_gt = scaler.inverse_y(y_norm)
    q_pred = scaler.inverse_y(np.vstack(all_q_pred))
    joint_mae_rad = float(np.mean(np.abs(q_pred - q_gt)))
    result["joint_mae_deg"] = joint_mae_rad * 180.0 / np.pi

    if ik is not None:
        X_np = scaler.inverse_X(np.vstack(all_X))
        ee_target = X_np[:, :12]
        pos_m, pos_max, ori_m, ori_max = batch_fk_error(ik, q_pred, ee_target)
        result.update({"fk_pos_err_mean": pos_m, "fk_pos_err_max": pos_max,
                       "fk_ori_err_mean": ori_m, "fk_ori_err_max": ori_max})
    return result


def train_epoch(loader, model, optimizer, criterion, device):
    model.train()
    total, n = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * X_batch.size(0)
        n += X_batch.size(0)
    return total / max(n, 1)


def main():
    set_seed(hp["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("加载数据...")
    train_loader, val_loader, test_loader, scaler, info = build_dataloaders(
        paths["data_dir"])
    print(f"  训练: {info['train_samples']} | 验证: {info['val_samples']} | 测试: {info['test_samples']}")

    print("加载 FK 引擎...")
    ik = load_ik()

    model = IKMLP().to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=hp["learning_rate"],
                            weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hp["lr_step"],
                                           gamma=hp["lr_gamma"])
    criterion = nn.MSELoss()

    os.makedirs(paths["results_dir"], exist_ok=True)
    best_joint_deg, best_epoch = float("inf"), -1
    train_history, val_history, deg_history, patience_counter = [], [], [], 0

    target_deg = hp["target_joint_deg"]
    print(f"\n训练 {hp['num_epochs']} epochs ... 目标: 关节角误差 < {target_deg}°")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Joint(°)':>8} | {'FK Pos(mm)':>10} | {'FK Ori(rad)':>10} | Time")
    print("-" * 80)

    for epoch in range(1, hp["num_epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(train_loader, model, optimizer, criterion, device)
        train_history.append(train_loss)

        val_result = evaluate(val_loader, model, scaler, device, ik=ik)
        val_history.append(val_result["loss"])
        joint_deg = val_result.get("joint_mae_deg", 0)
        fk_pos = val_result.get("fk_pos_err_mean", 0) * 1000
        fk_ori = val_result.get("fk_ori_err_mean", 0)

        print(f"{epoch:>6d} | {train_loss:>10.6f} | {val_result['loss']:>10.6f} | "
              f"{joint_deg:>8.3f} | {fk_pos:>10.3f} | {fk_ori:>10.4f} | {time.time()-t0:.1f}s")

        deg_history.append(joint_deg)

        # 达到目标精度 → 提前停止
        if joint_deg < target_deg:
            print(f"\n✓ 达到目标精度: joint MAE = {joint_deg:.3f}° < {target_deg}°")
            best_joint_deg, best_epoch = joint_deg, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_result["loss"], "joint_mae_deg": joint_deg},
                       os.path.join(paths["results_dir"], "best_model.pt"))
            with open(os.path.join(paths["results_dir"], "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            break

        scheduler.step()

        if joint_deg < best_joint_deg:
            best_joint_deg, best_epoch = joint_deg, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_result["loss"], "joint_mae_deg": joint_deg,
                        "fk_pos_err_mean": val_result.get("fk_pos_err_mean", 0),
                        "fk_ori_err_mean": val_result.get("fk_ori_err_mean", 0)},
                       os.path.join(paths["results_dir"], "best_model.pt"))
            with open(os.path.join(paths["results_dir"], "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= hp["patience"]:
            print(f"\nEarly stop at epoch {epoch} (no improvement for {hp['patience']} epochs)")
            break

    print(f"\n最佳模型: epoch {best_epoch}, joint MAE = {best_joint_deg:.3f}°")

    # 测试
    print("\n测试集...")
    ckpt = torch.load(os.path.join(paths["results_dir"], "best_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_r = evaluate(test_loader, model, scaler, device, ik=ik)
    print(f"  Joint MAE: {test_r.get('joint_mae_deg',0):.3f}° | "
          f"MSE: {test_r['loss']:.6f} | "
          f"FK Pos: {test_r.get('fk_pos_err_mean',0)*1000:.2f}mm | "
          f"FK Ori: {test_r.get('fk_ori_err_mean',0):.4f}rad")

    # 保存历史
    with open(os.path.join(paths["results_dir"], "history.json"), "w") as f:
        def conv(o):
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            raise TypeError
        json.dump({"train_loss": train_history, "val_loss": val_history,
                   "joint_deg": deg_history, "best_epoch": best_epoch,
                   "best_joint_deg": best_joint_deg, "test_result": test_r,
                   "info": info, "hp": hp}, f, indent=2, default=conv)

    # 损失曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_list = range(1, len(train_history) + 1)
    for ax, scale in zip([ax1, ax2], ["linear", "log"]):
        ax.plot(ep_list, train_history, label="Train", color="#2196F3", lw=0.8)
        ax.plot(ep_list, val_history, label="Val", color="#FF5722", lw=0.8)
        ax.axvline(best_epoch, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if scale == "log": ax.set_yscale("log")
    ax1.set_title("Loss"); ax2.set_title("Loss (Log)")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["results_dir"], "loss_curve.png"), dpi=150)
    plt.close(fig)
    print(f"结果: {paths['results_dir']}/")
    print("Done.")


if __name__ == "__main__":
    main()
