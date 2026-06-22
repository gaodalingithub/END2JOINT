#!/usr/bin/env python3
"""
Replay 客户端 — 连接 replay_server 的 ZMQ 服务，接收预录动作并驱动机器人。

用法（在机器人上运行）:
  python examples/ACTIBOT/replay_client.py \
    --host 192.168.1.27 \
    --port 5555

参数:
  --host       replay_server IP (默认: 192.168.1.27)
  --port       replay_server 端口 (默认: 5555)
  --loop       循环播放
  --speed      播放速度倍率 (默认: 1.0)
"""

from __future__ import annotations

import argparse
import io
import threading
import time
from collections import deque
from typing import Any

import msgpack
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import zmq


# 关节名称（与 main_actibot.py 保持一致）
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


def decode_custom_classes(obj: Any) -> Any:
    """msgpack object_hook: 将 numpy 数组从自定义格式还原。"""
    if isinstance(obj, dict):
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class ReplayClient:
    """连接到 replay_server 并逐帧接收动作。"""

    def __init__(self, host: str, port: int, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{port}")
        self.total_frames_received = 0

    def call(self, endpoint: str, data: dict | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self.socket.send(msgpack.packb(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        response = msgpack.unpackb(message, object_hook=decode_custom_classes)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call("ping")
            return True
        except zmq.error.ZMQError:
            return False

    def get_action_chunk(self) -> np.ndarray:
        """请求下一段动作块 → 返回 [action_horizon, 16] numpy 数组。"""
        response = self.call("get_action", {"observation": {"video": {}}, "options": None})
        # response = [action_dict, info_dict]
        action_dict = response[0]
        left = action_dict["left_arm_target_joint_positions"][0]      # [T, 7]
        right = action_dict["right_arm_target_joint_positions"][0]    # [T, 7]
        lg = action_dict["left_target_gripper_position"][0]           # [T, 1]
        rg = action_dict["right_target_gripper_position"][0]          # [T, 1]
        chunk = np.concatenate([left, right, lg, rg], axis=-1)       # [T, 16]
        self.total_frames_received += chunk.shape[0]
        return chunk

    def reset(self):
        """重置服务端帧指针到起点。"""
        self.call("reset", {"options": {}})

    def close(self):
        self.socket.close()
        self.context.term()


class ROS2ActionPub(Node):
    """ROS2 动作发布节点。"""

    def __init__(self):
        super().__init__("replay_client_node")
        self.pub = self.create_publisher(JointState, "/actibot_arm_ctrl", 1)
        self.get_logger().info("[ReplayClient] ROS2 publisher ready on /actibot_arm_ctrl")

    def step(self, action: np.ndarray):
        if action.shape[0] != 16:
            self.get_logger().warn(f"Unexpected action shape {action.shape}")
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "actibot_arm_ctrl"
        msg.name = ACTIBOT_CTRL_JOINT_NAMES_16
        msg.position = action.tolist()
        self.pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="Replay client for replay_server.py")
    parser.add_argument("--host", type=str, default="192.168.1.27", help="replay_server IP")
    parser.add_argument("--port", type=int, default=5555, help="replay_server port")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率")
    parser.add_argument("--delay", type=float, default=3.0, help="启动前等待秒数")
    args = parser.parse_args()

    # 连接服务端
    client = ReplayClient(host=args.host, port=args.port)
    if not client.ping():
        print(f"ERROR: 无法连接 {args.host}:{args.port}")
        return
    print(f"Connected to replay_server at {args.host}:{args.port}")

    # 初始化 ROS2
    rclpy.init()
    node = ROS2ActionPub()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 控制频率 30Hz
    CONTROL_INTERVAL = 1.0 / 30.0
    episode = 0

    print(f"Starting in {args.delay}s... (Ctrl+C to stop)")
    time.sleep(args.delay)

    try:
        while rclpy.ok():
            client.reset()  # 重置帧指针到起点
            print(f"\n--- Episode {episode} ---")

            server_times = deque(maxlen=50)
            remaining_in_chunk = 0
            chunk_actions = np.empty((0, 16), dtype=np.float64)

            step = 0
            while rclpy.ok():
                step_start = time.monotonic()

                # 当前 chunk 用完 → 请求下一段
                if remaining_in_chunk <= 0:
                    t0 = time.time()
                    chunk_actions = client.get_action_chunk()
                    dt = (time.time() - t0) * 1000
                    server_times.append(dt)
                    remaining_in_chunk = chunk_actions.shape[0]
                    print(f"  [request] +{remaining_in_chunk} actions  ({dt:.0f}ms)  "
                          f"total_received={client.total_frames_received}")

                # 取当前帧动作
                action = chunk_actions[chunk_actions.shape[0] - remaining_in_chunk]
                remaining_in_chunk -= 1
                node.step(action)
                step += 1

                # 帧间隔控制
                elapsed = time.monotonic() - step_start
                sleep_time = CONTROL_INTERVAL / args.speed - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if not args.loop:
                    # 单次播放：最后一步停顿后退出
                    if remaining_in_chunk <= 0 and chunk_actions.shape[0] < 8:
                        # 最后一段不足 8 帧，说明播完了
                        time.sleep(CONTROL_INTERVAL / args.speed)
                        print(f"Episode {episode} done. {step} steps played.")
                        break

                # 进度打印（每 30 帧）
                if step % 30 == 0:
                    avg = np.mean(server_times) if server_times else 0
                    print(f"  [{step}]  frames played  |  server avg: {avg:.0f}ms")

            episode += 1
            if not args.loop:
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        client.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
