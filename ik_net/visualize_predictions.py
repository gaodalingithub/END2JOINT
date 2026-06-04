#!/usr/bin/env python3
"""仿真可视化预测的关节角动作。

用法:
  conda activate actibot_sdk
  python ik_net/visualize_predictions.py ik_net/save/0602_test_for_net_action_fk_results/predictions.parquet
  python ik_net/visualize_predictions.py path/to/predictions.parquet --speed 2.0
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

_this_dir = os.path.abspath(os.path.dirname(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, ".."))
_example_dir = os.path.join(_project_root, "example")
for p in [_example_dir, _project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from actibot_fk import Arm_IK
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

URDF_PATH = os.path.join(_project_root,
    "actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf")


def main():
    parser = argparse.ArgumentParser(description="可视化预测关节角动作")
    parser.add_argument("file", help="predictions.parquet 文件路径")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率")
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    N = len(df)
    joint_cols = [c for c in df.columns if c.startswith(("L_", "R_"))]
    print(f"加载 {N} 帧预测数据, 关节列: {len(joint_cols)}")

    # 加载机器人模型
    robot = Arm_IK(URDF_PATH)
    model = robot.reduced_robot.model
    data = model.createData()

    # Meshcat 可视化
    try:
        viz = MeshcatVisualizer(robot.reduced_robot, robot.reduced_robot.collision_model,
                                 robot.reduced_robot.visual_model)
        viz.initViewer(loadModel=True)
        viz.display(np.zeros(model.nq))
        print("Meshcat 可视化已启动 (浏览器打开 http://127.0.0.1:7000/static/)")
    except Exception as e:
        print(f"可视化初始化失败: {e}")
        print("请确保 meshcat 已安装: pip install meshcat")
        return

    # 构建 19D q
    q = np.zeros(19)
    print(f"\n开始播放 ({N} 帧, {args.speed}x 速度)...")
    print("按 Ctrl+C 停止")

    try:
        for i in range(N):
            for j, col in enumerate(joint_cols[:14]):
                if col.startswith("L_"):
                    idx = 5 + j  # L_ 是前7个，对应 q[5:12]
                elif col.startswith("R_"):
                    idx = 12 + (j - 7)  # R_ 是后7个，对应 q[12:19]
                q[idx] = float(df[col].iloc[i])

            viz.display(q)

            # 按时间戳或固定频率播放
            if i < N - 1:
                dt = 0.033 / args.speed  # 30fps
                time.sleep(dt)

            if i % 30 == 0:
                print(f"  帧 {i:4d}/{N} | "
                      f"L_sh={np.degrees(q[5]):.1f}° "
                      f"L_el={np.degrees(q[8]):.1f}° "
                      f"R_sh={np.degrees(q[12]):.1f}° "
                      f"R_el={np.degrees(q[15]):.1f}°")

    except KeyboardInterrupt:
        print("\n播放停止")

    print("Done.")


if __name__ == "__main__":
    main()
