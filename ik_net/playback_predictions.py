#!/usr/bin/env python3
"""按预测结果回放机器人动作。

从 predictions.parquet 读取关节角，按原始时间戳发布到 /actibot_arm_ctrl。

用法:
  conda activate actibot_sdk
  python ik_net/playback_predictions.py ik_net/save/0602_test_for_net_action_fk_results/predictions.parquet
  python ik_net/playback_predictions.py path/to/predictions.parquet --speed 2.0  # 2 倍速
  python ik_net/playback_predictions.py path/to/predictions.parquet --loop       # 循环播放
"""
import sys
import time
import argparse
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class PlaybackNode(Node):
    def __init__(self, df, speed=1.0, loop=False):
        super().__init__("playback_node")
        self.pub = self.create_publisher(JointState, "/actibot_arm_ctrl", 10)
        self.df = df
        self.speed = speed
        self.loop = loop
        self.N = len(df)
        self.joint_names = [
            # left arm 7
            "left_shoulder_pitch_joint1", "left_shoulder_roll_joint2",
            "left_shoulder_yaw_joint3", "left_elbow_pitch_joint4",
            "left_elbow_roll_joint5", "left_wrist_yaw_joint6",
            "left_wrist_pitch_joint7",
            # right arm 7
            "right_shoulder_pitch_joint1", "right_shoulder_roll_joint2",
            "right_shoulder_yaw_joint3", "right_elbow_pitch_joint4",
            "right_elbow_roll_joint5", "right_wrist_yaw_joint6",
            "right_wrist_pitch_joint7",
            # gripper 2
            "left_gripper_a_joint", "left_gripper_b_joint",
        ]

        self.get_logger().info(f"加载 {self.N} 帧，速度 {speed}x{' 循环' if loop else ''}")

    def play(self):
        """按原始时间戳逐帧发布"""
        timestamps = self.df["timestamp"].values
        joint_cols = [c for c in self.df.columns if c.startswith(("L_", "R_"))]
        grip_cols = [c for c in self.df.columns if c.startswith("gripper_")]

        t_start = time.time()
        t0 = timestamps[0] if not np.isnan(timestamps[0]) else 0.0

        while rclpy.ok():
            for i in range(self.N):
                if not rclpy.ok():
                    break

                # 计算等待时间
                t_target = timestamps[i] if not np.isnan(timestamps[i]) else i / 30.0
                t_elapsed = (time.time() - t_start) * self.speed
                t_wait = (t_target - t0) - t_elapsed
                if t_wait > 0:
                    time.sleep(t_wait)

                # 构建 16 维命令
                positions = []
                for c in joint_cols[:14]:
                    positions.append(float(self.df[c].iloc[i]))

                # 夹爪
                if grip_cols:
                    gripper_l = float(self.df[grip_cols[0]].iloc[i])
                    gripper_r = float(self.df[grip_cols[1]].iloc[i]) if len(grip_cols) > 1 else gripper_l
                    positions.append(gripper_l)
                    positions.append(gripper_r)
                else:
                    positions.extend([0.0, 0.0])

                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.joint_names
                msg.position = positions
                msg.velocity = [0.0] * 16
                msg.effort = [0.0] * 16
                self.pub.publish(msg)

                if i % 30 == 0:
                    self.get_logger().info(
                        f"帧 {i:4d}/{self.N} | t={t_target:.2f}s | "
                        f"L_sh={np.degrees(positions[0]):.1f}° "
                        f"R_sh={np.degrees(positions[7]):.1f}°"
                    )

            if not self.loop:
                break
            self.get_logger().info("循环播放...")
            t_start = time.time()

        self.get_logger().info("播放完成")


def main():
    parser = argparse.ArgumentParser(description="回放预测的关节角动作")
    parser.add_argument("file", help="predictions.parquet 文件路径")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    print(f"加载 {len(df)} 帧预测数据")

    rclpy.init()
    node = PlaybackNode(df, speed=args.speed, loop=args.loop)
    try:
        node.play()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
