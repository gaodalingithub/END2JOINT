#!/usr/bin/env python3
"""
IK-Net 真机验证服务端。

IK-Net 输出作为控制信号驱动机器人（闭环 IK 验证）。

控制回路:
  Frame 0:   parquet 原数据 joint + gripper → 机器人（初始对齐）
  Frame 1+:  [ee_pose(parquet), prev_joint_state(机器人传感器反馈)]
             ──→ IK-Net ──→ pred_action(14) + gripper(parquet)
             ──→ 16D 控制信号 → 机器人执行

数据流:
  输入: ee_pose           ← save/episode_000000_action_fk.parquet
        prev_joint_state  ← main_actibot 观测中的 ROS2 传感器反馈
  输出: pred_action       ← IK-Net 推理结果（= 发给机器人的控制信号）
  保存: real_joint_state  ← 机器人传感器测得的真实关节位置
        pred_action       ← IK预测（= 控制信号）
        input_ee_*        ← IK 输入的末端位姿
        input_prev_*      ← IK 输入的上一帧真实关节

IK-Net 模型:
  输入 26D = [eeL_xyzrpy(6), eeR_xyzrpy(6), prev_joint_state_L(7), prev_joint_state_R(7)]
  输出 14D = [pred_action_L(7), pred_action_R(7)]  → 作为控制信号发给机器人

用法:
  # 本机启动
  python examples/ACTIBOT/ik_validation_server.py \
    --parquet save/episode_000000_action_fk.parquet

  # 推理延时
  python examples/ACTIBOT/ik_validation_server.py \
    --parquet save/episode_000000_action_fk.parquet \
    --inference-delay 0.05

  # 机器人端
  python main_actibot.py \
    --policy-host 192.168.0.223 \
    --policy-port 5555 \
    --instruction "replay" \
    --open-loop-horizon 1
"""

import argparse
import datetime
import os
import pickle
import time
from pathlib import Path

import msgpack_numpy as mnp
import numpy as np
import pandas as pd
import torch
import zmq


# ── Scaler 兼容加载（model_ik/ik_net/dataloader.py 的 Scaler 类）──
class _Scaler:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.X = StandardScaler()
        self.y = StandardScaler()

    def fit(self, X, y):
        self.X.fit(X)
        self.y.fit(y)
        return self

    def transform_X(self, X):
        return self.X.transform(X)

    def transform_y(self, y):
        return self.y.transform(y)

    def inverse_y(self, y_norm):
        return self.y.inverse_transform(y_norm)

    def inverse_X(self, X_norm):
        return self.X.inverse_transform(X_norm)


# ── 数据列名（对齐 predict.py 的 data_config）──
EE_COLS = [
    "eeL_x", "eeL_y", "eeL_z", "eeL_roll", "eeL_pitch", "eeL_yaw",
    "eeR_x", "eeR_y", "eeR_z", "eeR_roll", "eeR_pitch", "eeR_yaw",
]
STATE_COLS = [
    "state_L_sh_pitch", "state_L_sh_roll", "state_L_sh_yaw",
    "state_L_el_pitch", "state_L_el_roll", "state_L_wr_yaw", "state_L_wr_pitch",
    "state_R_sh_pitch", "state_R_sh_roll", "state_R_sh_yaw",
    "state_R_el_pitch", "state_R_el_roll", "state_R_wr_yaw", "state_R_wr_pitch",
]
ACTION_COLS = [
    "L_sh_pitch", "L_sh_roll", "L_sh_yaw",
    "L_el_pitch", "L_el_roll", "L_wr_yaw", "L_wr_pitch",
    "R_sh_pitch", "R_sh_roll", "R_sh_yaw",
    "R_el_pitch", "R_el_roll", "R_wr_yaw", "R_wr_pitch",
]
ACTIBOT_NAMES_16 = [
    "left_shoulder_pitch_joint1", "left_shoulder_roll_joint2", "left_shoulder_yaw_joint3",
    "left_elbow_pitch_joint4", "left_elbow_roll_joint5", "left_wrist_yaw_joint6",
    "left_wrist_pitch_joint7",
    "right_shoulder_pitch_joint1", "right_shoulder_roll_joint2", "right_shoulder_yaw_joint3",
    "right_elbow_pitch_joint4", "right_elbow_roll_joint5", "right_wrist_yaw_joint6",
    "right_wrist_pitch_joint7",
    "left_gripper_joint", "right_gripper_joint",
]


# ── IK-Net 模型 ──
class ResidualMLP(torch.nn.Module):
    def __init__(self, input_dim=26, hidden_dims=None, output_dim=14,
                 dropout=0.1, use_residual=False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [400, 300, 200, 100, 50]
        self.use_residual = use_residual
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            prev = h
        layers.append(torch.nn.Linear(prev, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if self.use_residual:
            out = x[:, -14:] + out
        return out


class IKValidationServer:
    """IK-Net 真机验证服务端：IK-Net 输出作为控制信号驱动机器人。"""

    def __init__(self, parquet_path, ee_parquet_path=None,
                 model_ckpt_dir=None,
                 model_ckpt_name=None, device=None):
        self.device = torch.device(device)

        # ── 加载 IK-Net 模型 ──
        ckpt_path = os.path.join(model_ckpt_dir, model_ckpt_name)
        scaler_path = os.path.join(model_ckpt_dir, "scaler.pkl")

        self.model = ResidualMLP()
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # scaler.pkl 依赖 dataloader.Scaler
        import sys as _sys
        import types as _types
        _mod = _types.ModuleType("dataloader")
        _mod.Scaler = _Scaler
        _sys.modules["dataloader"] = _mod

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        print(f"[IK] Model: {ckpt_path} (epoch {ckpt.get('epoch', '?')}) | 26D -> 14D")

        # ── 加载 parquet ──
        self.df = pd.read_parquet(parquet_path)
        self.total_frames = len(self.df)

        self.ee_df = self.df if ee_parquet_path is None else pd.read_parquet(ee_parquet_path)

        missing_ee = [c for c in EE_COLS if c not in self.ee_df.columns]
        if missing_ee:
            raise ValueError(f"EE 列缺失: {missing_ee}")
        for c in ["gripper_L", "gripper_R"]:
            if c not in self.df.columns:
                self.df[c] = 0.0

        print(f"[Data] {parquet_path} ({self.total_frames} frames) | ee: {len(self.ee_df)}")

        # ── 运行状态 ──
        self.frame_index = 0
        self.total_served = 0
        self.is_first_frame = True
        self.records = []
        self.inference_delay = 0.0
        self.fallback_position = np.array(
            [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
             -0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0,
             0.5, 0.5], dtype=np.float64)

    # ── 从观测提取真实关节 ──
    def _extract_joints_from_observation(self, observation):
        """从 main_actibot 观测中提取真实 joint_state (16D)。"""
        state = observation.get("state", {})
        try:
            left_arm = np.asarray(state.get("left_arm_joint_positions", [])).ravel()
            right_arm = np.asarray(state.get("right_arm_joint_positions", [])).ravel()
            left_grip = np.asarray(state.get("left_gripper_position", [])).ravel()
            right_grip = np.asarray(state.get("right_gripper_position", [])).ravel()
            if len(left_arm) >= 7 and len(right_arm) >= 7:
                return np.concatenate([
                    left_arm[-7:], right_arm[-7:],
                    left_grip[-1:] if len(left_grip) > 0 else [0.0],
                    right_grip[-1:] if len(right_grip) > 0 else [0.0],
                ]).astype(np.float64)
        except Exception:
            pass
        return None

    def _get_ee_pose(self, idx):
        idx = min(idx, len(self.ee_df) - 1)
        return self.ee_df[EE_COLS].iloc[idx].values.astype(np.float64)

    def _get_gripper(self, idx):
        idx = min(idx, self.total_frames - 1)
        return np.array([float(self.df["gripper_L"].iloc[idx]),
                         float(self.df["gripper_R"].iloc[idx])], dtype=np.float64)

    def _get_parquet_joints(self, idx):
        """读取 parquet state_* 列 (14D) 作为初始 prev_joints，仅用于 Frame 0。
        对齐 predict.py: 使用 state_* 列（传感器测量值）而非 action 列（控制目标）。"""
        idx = min(idx, self.total_frames - 1)
        return np.array([float(self.df[c].iloc[idx]) for c in STATE_COLS], dtype=np.float64)

    # ── IK 推理 ──
    def predict(self, ee_pose, prev_joints_14):
        x = np.concatenate([ee_pose, prev_joints_14]).astype(np.float64).reshape(1, -1)
        x_norm = self.scaler.transform_X(x)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred_norm = self.model(x_tensor).cpu().numpy()
        return self.scaler.inverse_y(pred_norm).ravel()

    # ── 响应构造 ──
    def make_response(self, real_joints_16):
        """IK 预测 → 控制信号。Frame 0 使用 parquet 原数据。"""
        if self.is_first_frame:
            self.is_first_frame = False
            ctrl_joints = self._get_parquet_joints(0)
            gripper = self._get_gripper(0)
            action_16 = np.concatenate([ctrl_joints, gripper])
            print("[IK] Frame 0: parquet original control")
            self.records.append({
                "frame": 0, "is_init": True,
                "real_joint_state": real_joints_16.copy(),
                "pred_action": action_16.copy(),
                "ee_pose": np.zeros(12),
                "input_prev_14": np.zeros(14),
            })
            self.total_served += 1
            return [self._pack_action(action_16), {"status": "continue"}]

        if self.frame_index >= self.total_frames:
            return [self._pack_action(real_joints_16), {"status": "done"}]

        idx = self.frame_index
        ee_pose = self._get_ee_pose(idx)
        gripper = self._get_gripper(idx)
        prev_14 = real_joints_16[:14]
        pred_14 = self.predict(ee_pose, prev_14)
        pred_16 = np.concatenate([pred_14, gripper])

        self.records.append({
            "frame": idx + 1,
            "is_init": False,
            "timestamp": (self.df["timestamp"].iloc[min(idx, self.total_frames - 1)]
                          if "timestamp" in self.df.columns else None),
            "real_joint_state": real_joints_16.copy(),
            "pred_action": pred_16.copy(),
            "ee_pose": ee_pose.copy(),
            "input_prev_14": prev_14.copy(),
        })

        self.frame_index += 1
        self.total_served += 1
        pct = min(self.frame_index / self.total_frames * 100, 100.0)
        print(f"[IK] frame={idx} ({pct:.1f}%)  "
              f"pred_L0={pred_14[0]:.3f}  real_L0={real_joints_16[0]:.3f}  "
              f"target_ee=({ee_pose[0]:.3f},{ee_pose[1]:.3f},{ee_pose[2]:.3f})")

        return [self._pack_action(pred_16), {"status": "continue"}]

    @staticmethod
    def _pack_action(action_16):
        return {
            "left_arm_target_joint_positions":  action_16[0:7][None, None, :],
            "right_arm_target_joint_positions": action_16[7:14][None, None, :],
            "left_target_gripper_position":     action_16[14:15][None, None, :],
            "right_target_gripper_position":    action_16[15:16][None, None, :],
        }

    # ── 记录保存 ──
    def save_records(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        rows = []
        for r in self.records:
            row = {"frame": r["frame"], "is_init": int(r["is_init"])}
            if r.get("timestamp") is not None:
                row["timestamp"] = r["timestamp"]

            for i, name in enumerate(ACTIBOT_NAMES_16):
                row[f"real_joint_state_{name}"] = r["real_joint_state"][i]
            for i, name in enumerate(ACTIBOT_NAMES_16):
                row[f"pred_action_{name}"] = r["pred_action"][i]

            if not r.get("is_init", False):
                for i, col in enumerate(EE_COLS):
                    row[f"input_ee_{col}"] = r["ee_pose"][i]
                for i, col in enumerate(ACTION_COLS):
                    row[f"input_prev_joint_state_{col}"] = r["input_prev_14"][i]

            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "ik_validation_log.csv")
        df.to_csv(csv_path, index=False)
        pq_path = os.path.join(output_dir, "ik_validation_log.parquet")
        df.to_parquet(pq_path, index=False)
        print(f"\n[Save] {len(rows)} records -> {csv_path}")
        print(f"[Save] {len(rows)} records -> {pq_path}")

        non_init = [r for r in self.records if not r.get("is_init", False)]
        if non_init:
            real_all = np.stack([r["real_joint_state"][:14] for r in non_init])
            pred_all = np.stack([r["pred_action"][:14] for r in non_init])
            err = np.abs(pred_all - real_all)
            print(f"\n[Stats] pred_action vs real_joint_state ({len(non_init)} 帧):")
            print(f"  MAE:  {np.mean(err)*180/np.pi:.3f} deg")
            print(f"  RMSE: {np.sqrt(np.mean(err**2))*180/np.pi:.3f} deg")
            for i, name in enumerate(ACTION_COLS):
                print(f"  {name:>15s}: {np.mean(err[:, i])*180/np.pi:.4f} deg")

    # ── 主循环 ──
    def run(self, host, port, output_dir):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://{host}:{port}")
        addr = sock.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"\n[Server] IK validation ready on {addr}")
        print(f"[Server] Robot: main_actibot.py --open-loop-horizon 1\n")

        running = True
        while running:
            try:
                msg = sock.recv()
                req = mnp.unpackb(msg, raw=False)
            except Exception as e:
                print(f"[Error] recv: {e}")
                continue

            ep = req.get("endpoint", "get_action")
            try:
                if ep == "ping":
                    result = {"status": "ok", "frame": self.frame_index,
                              "total": self.total_frames}

                elif ep == "get_action":
                    if self.inference_delay > 0:
                        time.sleep(self.inference_delay)

                    obs = req.get("data", {}).get("observation", {})
                    real_joints = self._extract_joints_from_observation(obs)
                    if real_joints is None:
                        real_joints = self.fallback_position.copy()
                        print("[Warning] 观测中未提取到关节状态, 使用初始位置")

                    _start = time.time()
                    result = self.make_response(real_joints)
                    _dt = (time.time() - _start) * 1000
                    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"[{ts}] IK: {_dt:.1f}ms  "
                          f"frame={self.frame_index}/{self.total_frames}")

                    info = result[1] if len(result) > 1 else {}
                    if info.get("status") == "done":
                        print("=" * 55)
                        print(f"[Server] 所有 {self.total_frames} 帧处理完成")
                        print("[Server] 正在保存记录...")
                        self.save_records(output_dir)
                        print("[Server] 验证完成, 服务停止")
                        print("=" * 55)
                        running = False

                elif ep == "get_status":
                    result = {"status": "ok", "frame": self.frame_index,
                              "total": self.total_frames, "records": len(self.records)}

                elif ep == "save":
                    self.save_records(output_dir)
                    result = {"status": "ok", "records": len(self.records)}

                elif ep == "reset":
                    self.frame_index = 0
                    self.total_served = 0
                    self.is_first_frame = True
                    self.records = []
                    result = {}
                    print("[Server] Reset")

                elif ep == "kill":
                    result = {}
                    running = False
                    if self.records:
                        self.save_records(output_dir)
                    print("[Server] Kill")

                else:
                    result = {"error": f"Unknown: {ep}"}

                sock.send(mnp.packb(result))

            except Exception:
                import traceback
                traceback.print_exc()
                sock.send(mnp.packb({"error": "internal error"}))

        sock.close()
        ctx.term()
        if self.records:
            self.save_records(output_dir)
        print(f"[Server] Done. Total: {self.total_served}")


def main():
    p = argparse.ArgumentParser(description="IK-Net real robot validation server")
    p.add_argument("--parquet", required=True,
                   help="parquet (ACTION_COLS + gripper + EE_COLS)")
    p.add_argument("--ee-parquet", default=None,
                   help="parquet with ee pose (default = --parquet)")
    p.add_argument("--model-dir", default="model_ik/0616_end2action_model_moredata",
                   help="IK-Net checkpoint dir")
    p.add_argument("--model-name", default="0616_model.pt",
                   help="IK-Net checkpoint name")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=5555, help="Bind port")
    p.add_argument("--output-dir", default="/home/ubuntu/code/End2Joint/ik_net/save_real",
                   help="Output dir")
    p.add_argument("--device", default="cuda:0", help="cpu or cuda:0")
    p.add_argument("--inference-delay", type=float, default=0.03,
                   help="Simulated inference delay (s)")
    args = p.parse_args()

    sv = IKValidationServer(
        parquet_path=args.parquet,
        ee_parquet_path=args.ee_parquet,
        model_ckpt_dir=args.model_dir,
        model_ckpt_name=args.model_name,
        device=args.device,
    )
    sv.inference_delay = args.inference_delay
    if args.inference_delay > 0:
        print(f"[Server] Inference delay: {args.inference_delay*1000:.0f} ms")
    try:
        sv.run(args.host, args.port, args.output_dir)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        if sv.records:
            sv.save_records(args.output_dir)


if __name__ == "__main__":
    main()
