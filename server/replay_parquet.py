#!/usr/bin/env python3
"""
回放 predictions.parquet 中的 control 信号到真实机器人。

流程：
  读取 parquet → 按帧顺序组装 16-D 动作向量 → 以 30Hz 通过 ROS2 发布 JointState 到 /actibot_arm_ctrl

用法：
  # 单次执行
  python replay_parquet.py --parquet <path>

  # 循环播放
  python replay_parquet.py --parquet <path> --loop

  # 指定播放速度倍率 (0.5 = 半速, 2.0 = 两倍速)
  python replay_parquet.py --parquet <path> --speed 0.5

注意：
  - 默认帧率为数据中的间隔 (约 30 FPS / 0.0333s)
  - 发布频率会尽量贴合数据集原始节奏，受 ROS2 时钟精度限制
"""

import argparse
import time

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


# 左臂 + 右臂 + 左夹爪 + 右夹爪 = 16 维
JOINT_NAMES = [
    # 左臂 (7)
    "L_sh_pitch", "L_sh_roll", "L_sh_yaw",
    "L_el_pitch", "L_el_roll", "L_wr_yaw", "L_wr_pitch",
    # 右臂 (7)
    "R_sh_pitch", "R_sh_roll", "R_sh_yaw",
    "R_el_pitch", "R_el_roll", "R_wr_yaw", "R_wr_pitch",
    # 夹爪 (2)
    "gripper_L", "gripper_R",
]

PARQUET_TO_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}


class ParquetReplayNode(Node):
    """读取 parquet 并按帧顺序发布关节控制指令。"""

    def __init__(self, parquet_path: str, speed: float = 1.0, loop: bool = False):
        super().__init__("parquet_replay_node")
        self.pub = self.create_publisher(JointState, "/actibot_arm_ctrl", 1)

        # 加载数据
        df = pd.read_parquet(parquet_path)
        self.frames = self._df_to_actions(df)
        self.speed = speed
        self.loop = loop
        self.total_frames = len(self.frames)

        # 计算帧间隔（从数据集的 timestamp 列推导）
        if "timestamp" in df.columns and self.total_frames > 1:
            ts = df["timestamp"].values
            diffs = ts[1:] - ts[:-1]
            self.frame_interval = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0 / 30.0
        else:
            self.frame_interval = 1.0 / 30.0  # 默认 30Hz

        self.get_logger().info(
            f"Loaded {self.total_frames} frames | "
            f"frame_interval={self.frame_interval:.4f}s ({1.0/self.frame_interval:.1f} FPS) | "
            f"speed={speed:.2f}x | loop={loop}"
        )

    @staticmethod
    def _df_to_actions(df: pd.DataFrame) -> np.ndarray:
        """将 parquet DataFrame 转换为 [N, 16] 动作数组。"""
        actions = np.zeros((len(df), 16), dtype=np.float64)
        for col in df.columns:
            if col in PARQUET_TO_INDEX:
                actions[:, PARQUET_TO_INDEX[col]] = df[col].values.astype(np.float64)
        return actions

    def run(self):
        """主播放循环。"""
        self.get_logger().info("Starting replay in 3 seconds...")
        time.sleep(3)

        episode = 0
        while rclpy.ok():
            self.get_logger().info(f"--- Episode {episode} ({self.total_frames} frames) ---")

            play_start = time.monotonic()
            for i in range(self.total_frames):
                if not rclpy.ok():
                    return

                expected_time = play_start + i * self.frame_interval / self.speed
                now = time.monotonic()
                if now < expected_time:
                    time.sleep(expected_time - now)

                action = self.frames[i]

                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = JOINT_NAMES
                msg.position = action.tolist()
                self.pub.publish(msg)

                if i % 50 == 0 or i == self.total_frames - 1:
                    progress = (i + 1) / self.total_frames * 100
                    self.get_logger().info(
                        f"[{i + 1:4d}/{self.total_frames}] {progress:5.1f}% | "
                        f"gripper_R={action[15]:.3f}"
                    )

            elapsed = time.monotonic() - play_start
            real_fps = self.total_frames / elapsed if elapsed > 0 else 0
            self.get_logger().info(
                f"Episode {episode} done in {elapsed:.2f}s ({real_fps:.1f} FPS actual)"
            )

            if not self.loop:
                break
            episode += 1

        self.get_logger().info("Replay finished.")


def main():
    parser = argparse.ArgumentParser(description="Replay parquet control signals to Actibot via ROS2")
    parser.add_argument("--parquet", type=str, required=True,
                        help="Path to predictions.parquet")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (0.5=half, 2.0=double)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop playback indefinitely")
    args = parser.parse_args()

    rclpy.init()
    node = ParquetReplayNode(args.parquet, speed=args.speed, loop=args.loop)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
