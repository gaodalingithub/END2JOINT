#!/usr/bin/env python3
"""
回放 predictions.parquet 控制信号的 ZMQ 服务端。

使用 msgpack_numpy 序列化，与机器人端 server_client.py 协议完全一致。

用法（本机 192.168.1.27）:
# 启动服务
python examples/ACTIBOT/replay_server.py \
  --parquet save/0602_test_for_net_action_fk_results/predictions.parquet \
  --host 0.0.0.0 \
  --port 5555

python examples/ACTIBOT/replay_server.py \
  --parquet save/0602_test_for_net_action_fk_results/predictions.parquet\
  --host 0.0.0.0 \
  --port 5555

python examples/ACTIBOT/replay_server.py \
  --parquet save/0602_test_for_net_action_fk_results_ar/predictions_ar.parquet \
  --host 0.0.0.0 \
  --port 5555  

机器人端 (192.168.1.50):
cd /opt/acti/install/actibot_vla_client/examples/Isaac-GROOT_dev/examples/ACTIBOT/
python main_actibot.py \
  --policy-host 192.168.1.27 \
  --policy-port 5555 \
  --instruction "replay" \
  --open-loop-horizon 8
"""

import argparse
import datetime
import time

import msgpack_numpy as mnp
import numpy as np
import pandas as pd
import zmq

JOINT_NAMES = [
    "L_sh_pitch", "L_sh_roll", "L_sh_yaw",
    "L_el_pitch", "L_el_roll", "L_wr_yaw", "L_wr_pitch",
    "R_sh_pitch", "R_sh_roll", "R_sh_yaw",
    "R_el_pitch", "R_el_roll", "R_wr_yaw", "R_wr_pitch",
    "gripper_L", "gripper_R",
]
PARQUET_TO_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}


ACTIBOT_CTRL_JOINT_NAMES_16 = [
    "left_shoulder_pitch_joint1",
    "left_shoulder_roll_joint2",
    "left_shoulder_yaw_joint3",
    "left_elbow_pitch_joint4",
    "left_elbow_roll_joint5",
    "left_wrist_yaw_joint6",
    "left_wrist_pitch_joint7",
    "right_shoulder_pitch_joint1",
    "right_shoulder_roll_joint2",
    "right_shoulder_yaw_joint3",
    "right_elbow_pitch_joint4",
    "right_elbow_roll_joint5",
    "right_wrist_yaw_joint6",
    "right_wrist_pitch_joint7",
    "left_gripper_joint",
    "right_gripper_joint",
]


class ParquetReplayServer:
    """读取 parquet 并通过 ZMQ 逐段回放动作（msgpack_numpy 序列化）。

    同时缓存机器人端 main_actibot.py 发送的实时关节状态，
    可通过 get_joint_state / get_named_joint_state 端点查询。
    """

    def __init__(self, parquet_path: str, action_horizon: int = 8, loop: bool = False):
        # 加载 parquet 数据 → [N, 16]
        df = pd.read_parquet(parquet_path)
        self.total_frames = len(df)
        raw = np.zeros((self.total_frames, 16), dtype=np.float64)
        for col in df.columns:
            if col in PARQUET_TO_INDEX:
                raw[:, PARQUET_TO_INDEX[col]] = df[col].values.astype(np.float64)

        self.actions = raw
        self.left_arm = raw[:, 0:7]       # [N, 7]
        self.right_arm = raw[:, 7:14]      # [N, 7]
        self.left_grip = raw[:, 14:15]     # [N, 1]
        self.right_grip = raw[:, 15:16]    # [N, 1]

        self.action_horizon = action_horizon
        self.loop = loop
        self.frame_index = 0
        self.total_served = 0

        # 缓存机器人端发来的实时关节状态（16-D ndarray）
        self.latest_joint_state = np.zeros(16, dtype=np.float64)
        self.joint_state_update_count = 0

        print(f"[ReplayServer] Loaded {self.total_frames} frames | "
              f"action_horizon={action_horizon} | loop={loop}")

    def _cache_robot_state(self, observation: dict):
        """从机器人端 get_action 请求的 observation 中提取关节状态并缓存。"""
        state = observation.get("state", {})
        try:
            left_arm = np.asarray(state.get("left_arm_joint_positions", [[[0.0]*7]])).ravel()
            right_arm = np.asarray(state.get("right_arm_joint_positions", [[[0.0]*7]])).ravel()
            left_grip = np.asarray(state.get("left_gripper_position", [[[0.0]]])).ravel()
            right_grip = np.asarray(state.get("right_gripper_position", [[[0.0]]])).ravel()
            # 拼接为 16-D（取最后7/7/1/1 个值）
            self.latest_joint_state = np.concatenate([
                left_arm[-7:], right_arm[-7:], left_grip[-1:], right_grip[-1:],
            ]).astype(np.float64)
            self.joint_state_update_count += 1
        except Exception:
            pass  # 观测格式不对时静默跳过

    def next_chunk(self):
        """获取下一段动作块，返回 (left_arm, right_arm, left_grip, right_grip) 各为 [T, D]。"""
        start = self.frame_index
        end = start + self.action_horizon

        if end <= self.total_frames:
            left = self.left_arm[start:end]
            right = self.right_arm[start:end]
            lg = self.left_grip[start:end]
            rg = self.right_grip[start:end]
            self.frame_index = end
        else:
            left = self.left_arm[start:]
            right = self.right_arm[start:]
            lg = self.left_grip[start:]
            rg = self.right_grip[start:]

            if self.loop:
                need = self.action_horizon - len(left)
                left = np.concatenate([left, self.left_arm[:need]], axis=0)
                right = np.concatenate([right, self.right_arm[:need]], axis=0)
                lg = np.concatenate([lg, self.left_grip[:need]], axis=0)
                rg = np.concatenate([rg, self.right_grip[:need]], axis=0)
                self.frame_index = need
            else:
                # 不足或耗尽时，用最后一帧重复填充至 action_horizon
                last_l = self.left_arm[-1:] if len(left) == 0 else left[-1:]
                last_r = self.right_arm[-1:] if len(right) == 0 else right[-1:]
                last_lg = self.left_grip[-1:] if len(lg) == 0 else lg[-1:]
                last_rg = self.right_grip[-1:] if len(rg) == 0 else rg[-1:]
                pad = self.action_horizon - len(left)
                if pad > 0:
                    left = np.concatenate([left] + [last_l] * pad, axis=0)
                    right = np.concatenate([right] + [last_r] * pad, axis=0)
                    lg = np.concatenate([lg] + [last_lg] * pad, axis=0)
                    rg = np.concatenate([rg] + [last_rg] * pad, axis=0)
                self.frame_index = self.total_frames

        return left, right, lg, rg

    def make_response(self):
        """构造 get_action 响应，格式对齐 main_actibot._decode_action_chunk。"""
        left, right, lg, rg = self.next_chunk()
        T = left.shape[0]
        action_dict = {
            "left_arm_target_joint_positions":  left[None, :, :],    # [1, T, 7] ndarray
            "right_arm_target_joint_positions": right[None, :, :],   # [1, T, 7]
            "left_target_gripper_position":     lg[None, :, :],      # [1, T, 1]
            "right_target_gripper_position":    rg[None, :, :],      # [1, T, 1]
        }
        start_frame = max(0, self.frame_index - T)
        end_frame = min(self.total_frames, self.frame_index) - 1
        pct = self.frame_index / self.total_frames * 100
        self.total_served += T
        print(f"[Replay] frame={start_frame}-{end_frame}  "
              f"({min(pct, 100.0):.1f}%)  served={self.total_served}")
        # 返回格式：policy.get_action() 返回 (action_dict, info_dict) → msgpack 序列化为 list
        return [action_dict, {}]

    def run(self, host: str, port: int):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{host}:{port}")
        addr = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Replay server ready on {addr}")

        running = True
        while running:
            try:
                message = socket.recv()
                request = mnp.unpackb(message, raw=False)
            except Exception as e:
                print(f"[Error] recv/unpack failed: {e}")
                continue

            endpoint = request.get("endpoint", "get_action")

            try:
                if endpoint == "ping":
                    result = {"status": "ok", "message": "Replay server is running"}

                elif endpoint == "get_action":
                    # 缓存机器人端发来的实时关节状态
                    obs = request.get("data", {}).get("observation", {})
                    self._cache_robot_state(obs)

                    _start = time.time()
                    result = self.make_response()
                    _dt = (time.time() - _start) * 1000
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                          f"Replay responded in {_dt:.1f} ms")

                elif endpoint == "get_joint_state":
                    result = {
                        "status": "ok",
                        "position": self.latest_joint_state.copy(),
                        "names": ACTIBOT_CTRL_JOINT_NAMES_16,
                        "update_count": self.joint_state_update_count,
                    }

                elif endpoint == "get_named_joint_state":
                    pos = self.latest_joint_state
                    named = {}
                    for i, name in enumerate(ACTIBOT_CTRL_JOINT_NAMES_16):
                        if i < len(pos):
                            named[name] = float(pos[i])
                    result = {
                        "status": "ok",
                        "joints": named,
                        "update_count": self.joint_state_update_count,
                    }

                elif endpoint == "reset":
                    self.frame_index = 0
                    self.total_served = 0
                    result = {}
                    print("[Replay] Reset — 回到第一帧")

                elif endpoint == "kill":
                    result = {}
                    running = False
                    print("[Replay] Kill signal received")

                elif endpoint == "get_modality_config":
                    result = {}

                else:
                    result = {"error": f"Unknown endpoint: {endpoint}"}

                socket.send(mnp.packb(result))

            except Exception as e:
                import traceback
                print(f"[Error] handling {endpoint}: {e}")
                traceback.print_exc()
                socket.send(mnp.packb({"error": str(e)}))

        socket.close()
        context.term()
        print(f"Server stopped. Total frames served: {self.total_served}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay parquet control signals as a ZMQ policy server "
                    "(msgpack_numpy protocol, compatible with robot's server_client.py)"
    )
    parser.add_argument("--parquet", required=True, help="Path to predictions.parquet")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument("--port", type=int, default=5555, help="Server bind port")
    parser.add_argument("--action-horizon", type=int, default=8,
                        help="Number of actions per chunk (default: 8)")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args()

    server = ParquetReplayServer(
        parquet_path=args.parquet,
        action_horizon=args.action_horizon,
        loop=args.loop,
    )
    try:
        server.run(args.host, args.port)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()
