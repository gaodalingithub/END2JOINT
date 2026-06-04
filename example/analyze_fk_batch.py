#!/usr/bin/env python3
"""
批量读取 LeRobot 格式的 .parquet 数据，使用 FK 计算末端位姿并分析动作。

用法:
  # 处理文件夹下所有 episode
  conda activate actibot_sdk
  python example/analyze_fk_batch.py /path/to/data/folder

  # 指定输出文件 (JSON)
  python example/analyze_fk_batch.py /path/to/data/folder -o results.json

  # 只处理指定编号的 episode (如 0, 1, 2)
  python example/analyze_fk_batch.py /path/to/data/folder -e 0 2 5

  # 列出基本信息，不做完整 FK 计算
  python example/analyze_fk_batch.py /path/to/data/folder --info
"""
import sys, os, json, glob, argparse

# 添加 example 目录到 path，使 actibot_fk 可直接导入
_example_dir = os.path.abspath(os.path.dirname(__file__))
if _example_dir not in sys.path:
    sys.path.insert(0, _example_dir)

import numpy as np
import pandas as pd
from actibot_fk import Arm_IK

URDF_PATH = "actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf"

JOINT_NAMES_L = ["L_sh_pitch", "L_sh_roll", "L_sh_yaw",
                 "L_el_pitch", "L_el_roll", "L_wr_yaw", "L_wr_pitch"]
JOINT_NAMES_R = ["R_sh_pitch", "R_sh_roll", "R_sh_yaw",
                 "R_el_pitch", "R_el_roll", "R_wr_yaw", "R_wr_pitch"]
JOINT_NAMES_ALL = JOINT_NAMES_L + JOINT_NAMES_R


def load_fk():
    return Arm_IK(URDF_PATH)


def state_to_q(state_16):
    """16 维 state → 19 维 q 向量"""
    q = np.zeros(19)
    q[5:12] = state_16[0:7]     # left arm
    q[12:19] = state_16[7:14]   # right arm
    return q


def analyze_motion(ik, df, max_disp_thresh=0.005):
    """
    对单个 episode 计算 FK 并分析动作。

    返回 dict:
      episode_index, n_frames, duration_s,
      left/right: {start/end/min/max_pose, delta, dominant_axis, total_dist},
      joint_deltas, motion_label
    """
    N = len(df)
    state_0 = df.iloc[0]["observation.state"]
    state_last = df.iloc[N - 1]["observation.state"]
    ts = df["timestamp"].values
    duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    # 每帧 FK
    poses_l = []
    poses_r = []
    rpys_l = []
    rpys_r = []
    for i in range(N):
        q = state_to_q(df.iloc[i]["observation.state"])
        T_l, T_r = ik.get_fk_solution(q)
        poses_l.append(T_l[:3, 3].copy())
        poses_r.append(T_r[:3, 3].copy())
        from pinocchio.rpy import matrixToRpy
        rpys_l.append(matrixToRpy(T_l[:3, :3]))
        rpys_r.append(matrixToRpy(T_r[:3, :3]))

    pos_l = np.array(poses_l)  # (N, 3)
    pos_r = np.array(poses_r)
    rpy_l = np.array(rpys_l)  # (N, 3)  radians
    rpy_r = np.array(rpys_r)

    def pose_stats(pos, rpy=None):
        s = {
            "start": pos[0].tolist(),
            "end":   pos[-1].tolist(),
            "delta": (pos[-1] - pos[0]).tolist(),
            "min":   pos.min(axis=0).tolist(),
            "max":   pos.max(axis=0).tolist(),
            "range": (pos.max(axis=0) - pos.min(axis=0)).tolist(),
            "total_dist_3d": float(np.sum(np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=1)))),
        }
        if rpy is not None:
            s["rpy_start"] = [round(float(rpy[0, 0]), 4), round(float(rpy[0, 1]), 4), round(float(rpy[0, 2]), 4)]
            s["rpy_end"]   = [round(float(rpy[-1, 0]), 4), round(float(rpy[-1, 1]), 4), round(float(rpy[-1, 2]), 4)]
            s["rpy_delta"] = [round(float(rpy[-1, 0] - rpy[0, 0]), 4),
                              round(float(rpy[-1, 1] - rpy[0, 1]), 4),
                              round(float(rpy[-1, 2] - rpy[0, 2]), 4)]
            s["rpy_range"] = [round(float(rpy[:, 0].max() - rpy[:, 0].min()), 4),
                              round(float(rpy[:, 1].max() - rpy[:, 1].min()), 4),
                              round(float(rpy[:, 2].max() - rpy[:, 2].min()), 4)]
        return s

    result = {
        "episode_index": int(df.iloc[0]["episode_index"]),
        "n_frames": N,
        "duration_s": round(duration, 2),
        "left":  pose_stats(pos_l, rpy_l),
        "right": pose_stats(pos_r, rpy_r),
    }

    # 关节变化
    joints = np.array([df.iloc[i]["observation.state"][:14] for i in range(N)])
    jd = {}
    for j, name in enumerate(JOINT_NAMES_L + JOINT_NAMES_R):
        jd[name] = {
            "start": round(float(joints[0, j]), 4),
            "end":   round(float(joints[-1, j]), 4),
            "delta": round(float(joints[-1, j] - joints[0, j]), 4),
            "range": round(float(joints[:, j].max() - joints[:, j].min()), 4),
        }
    result["joint_deltas"] = jd

    # 夹爪
    gripper = np.array([df.iloc[i]["observation.state"][14:16] for i in range(N)])
    result["gripper"] = {
        "left_start":  round(float(gripper[0, 0]), 4),
        "left_end":    round(float(gripper[-1, 0]), 4),
        "right_start": round(float(gripper[0, 1]), 4),
        "right_end":   round(float(gripper[-1, 1]), 4),
        "left_range":  round(float(gripper[:, 0].max() - gripper[:, 0].min()), 4),
        "right_range": round(float(gripper[:, 1].max() - gripper[:, 1].min()), 4),
    }

    # 动作分类
    delta_l = np.abs(pos_l[-1] - pos_l[0])
    delta_r = np.abs(pos_r[-1] - pos_r[0])
    dist_l = float(np.sum(np.sqrt(np.sum(np.diff(pos_l, axis=0)**2, axis=1))))
    dist_r = float(np.sum(np.sqrt(np.sum(np.diff(pos_r, axis=0)**2, axis=1))))

    # 判断主要运动侧和轴
    max_d_l = float(delta_l.max())
    max_d_r = float(delta_r.max())
    dominant_side = "left" if max_d_l > max_d_r else "right"
    dominant_axis_idx = int(np.argmax(delta_r if dominant_side == "right" else delta_l))
    axis_labels = ["x (前)", "y (侧)", "z (垂直)"]
    dominant_axis = axis_labels[dominant_axis_idx]

    # 判断静止 vs 运动
    both_still = max_d_l < max_disp_thresh and max_d_r < max_disp_thresh
    if both_still:
        label = "静止 / 微调"
    elif max_d_r > max_d_l * 3 and max_d_r > 0.03:
        label = f"右臂{dominant_axis}运动"
    elif max_d_l > max_d_r * 3 and max_d_l > 0.03:
        label = f"左臂{dominant_axis}运动"
    elif max_d_l > 0.03 and max_d_r > 0.03:
        label = "双臂运动"
    else:
        label = "小幅度调整"

    # 判断是否接近零位
    q0_r = state_to_q(state_last)[12:19]
    near_zero_r = float(np.max(np.abs(q0_r))) < 0.3
    if "右臂" in label and near_zero_r:
        label += " → 趋于零位"
    elif "左臂" in label:
        q0_l = state_to_q(state_last)[5:12]
        if float(np.max(np.abs(q0_l))) < 0.3:
            label += " → 趋于零位"

    result["motion_label"] = label

    # 动作向量 (action 的 delta)
    actions = np.array([df.iloc[i]["action"][:14] for i in range(N)])
    result["action_deltas"] = {
        JOINT_NAMES_ALL[j]: {
            "start": round(float(actions[0, j]), 4),
            "end": round(float(actions[-1, j]), 4),
            "delta": round(float(actions[-1, j] - actions[0, j]), 4),
        }
        for j in range(14)
    }

    return result


def print_report(result, verbose=False):
    ep = result["episode_index"]
    lbl = result["motion_label"]
    dur = result["duration_s"]
    nf = result["n_frames"]

    print(f"\n{'='*60}")
    print(f"  Episode {ep:3d}  |  {nf:3d} 帧  |  {dur:5.2f} s  |  {lbl}")
    print(f"{'='*60}")

    for side, key in [("左臂", "left"), ("右臂", "right")]:
        s = result[key]
        d = s["delta"]
        r = s["range"]
        dist = s["total_dist_3d"]
        print(f"  {side}:")
        print(f"    位置: start ({s['start'][0]:.4f}, {s['start'][1]:.4f}, {s['start'][2]:.4f})  "
              f"end ({s['end'][0]:.4f}, {s['end'][1]:.4f}, {s['end'][2]:.4f})")
        print(f"           delta ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})  "
              f"range ({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})  路程 {dist:.4f} m")
        if "rpy_start" in s:
            rpys = s["rpy_start"]
            rpye = s["rpy_end"]
            rpyd = s["rpy_delta"]
            rpyr = s["rpy_range"]
            print(f"    姿态: start ({rpys[0]:.4f}, {rpys[1]:.4f}, {rpys[2]:.4f})  "
                  f"end ({rpye[0]:.4f}, {rpye[1]:.4f}, {rpye[2]:.4f}) rad")
            print(f"           delta ({rpyd[0]:+.4f}, {rpyd[1]:+.4f}, {rpyd[2]:+.4f})  "
                  f"range ({rpyr[0]:.4f}, {rpyr[1]:.4f}, {rpyr[2]:.4f}) rad")

    if verbose:
        print(f"\n  关节变化 (delta / range):")
        for jn in JOINT_NAMES_ALL:
            jd = result["joint_deltas"][jn]
            print(f"    {jn:>12s}: {jd['delta']:+7.4f}  / {jd['range']:.4f}")


def list_episodes(folder):
    """只列出基本信息，不做 FK"""
    pattern = os.path.join(folder, "**", "episode_*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        # 尝试直接 data/chunk-000/episode_*.parquet
        pattern2 = os.path.join(folder, "data", "chunk-000", "episode_*.parquet")
        files = sorted(glob.glob(pattern2))

    if not files:
        print(f"错误: 在 {folder} 下未找到 episode_*.parquet 文件")
        return False

    print(f"找到 {len(files)} 个 episode 文件:\n")
    print(f"{'文件':35s} {'episode':>8s} {'帧数':>6s} {'时长(s)':>8s}")
    print("-" * 60)
    for f in files:
        try:
            df = pd.read_parquet(f)
            ep = int(df.iloc[0]["episode_index"])
            nf = len(df)
            ts = df["timestamp"].values
            dur = ts[-1] - ts[0] if len(ts) > 1 else 0.0
            print(f"{os.path.basename(f):35s} {ep:8d} {nf:6d} {dur:8.2f}")
        except Exception as e:
            print(f"{os.path.basename(f):35s} {'读取失败':>8s} — {e}")
    return True


def run_analysis(folder, episode_filter=None, verbose=False, output=None):
    pattern = os.path.join(folder, "**", "episode_*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        pattern2 = os.path.join(folder, "data", "chunk-000", "episode_*.parquet")
        files = sorted(glob.glob(pattern2))
    if not files:
        print(f"错误: 在 {folder} 下未找到 episode_*.parquet 文件")
        return

    # 如果指定了 episode 编号
    if episode_filter is not None:
        filtered = []
        for f in files:
            base = os.path.basename(f)
            try:
                ep_num = int(base.replace("episode_", "").replace(".parquet", ""))
                if ep_num in episode_filter:
                    filtered.append(f)
            except ValueError:
                pass
        files = filtered
        if not files:
            print(f"错误: 未找到指定 episode {episode_filter}")
            return

    print(f"加载 FK 引擎...")
    ik = load_fk()
    print(f"分析 {len(files)} 个 episode...")

    all_results = []
    for fpath in files:
        ep_num = int(os.path.basename(fpath).replace("episode_", "").replace(".parquet", ""))
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            print(f"  ✗ episode {ep_num:3d}: 读取失败 — {e}")
            continue

        try:
            result = analyze_motion(ik, df)
            all_results.append(result)
            print_report(result, verbose=verbose)
        except Exception as e:
            print(f"  ✗ episode {ep_num:3d}: FK 计算失败 — {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    if len(all_results) > 1:
        print(f"\n\n{'='*60}")
        print(f"  汇总 ({len(all_results)} episodes)")
        print(f"{'='*60}")
        print(f"{'ep':>4s} {'帧数':>5s} {'时长':>5s} "
              f"{'Δx_l':>6s} {'Δy_l':>6s} {'Δz_l':>6s}  "
              f"{'Δx_r':>6s} {'Δy_r':>6s} {'Δz_r':>6s}  "
              f"{'ΔR_l':>7s} {'ΔP_l':>7s} {'ΔY_l':>7s}  "
              f"{'ΔR_r':>7s} {'ΔP_r':>7s} {'ΔY_r':>7s}  动作")
        print("-" * 130)
        for r in all_results:
            dl = r["left"]["delta"]
            dr = r["right"]["delta"]
            rdl = r["left"].get("rpy_delta", [0,0,0])
            rdr = r["right"].get("rpy_delta", [0,0,0])
            print(f"{r['episode_index']:4d} {r['n_frames']:5d} {r['duration_s']:5.1f} "
                  f"{dl[0]:+6.4f} {dl[1]:+6.4f} {dl[2]:+6.4f}  "
                  f"{dr[0]:+6.4f} {dr[1]:+6.4f} {dr[2]:+6.4f}  "
                  f"{rdl[0]:+7.4f} {rdl[1]:+7.4f} {rdl[2]:+7.4f}  "
                  f"{rdr[0]:+7.4f} {rdr[1]:+7.4f} {rdr[2]:+7.4f}  {r['motion_label']}")

    # 保存 JSON
    if output and all_results:
        with open(output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="批量分析 LeRobot .parquet 数据，计算末端位姿并判断动作")
    parser.add_argument("folder", nargs="?", default=None,
                        help="数据文件夹路径 (包含 episode_*.parquet)")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 JSON 文件路径")
    parser.add_argument("-e", "--episodes", nargs="+", type=int, default=None,
                        help="只处理指定的 episode 编号，如 -e 0 2 5")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出详细关节变化")
    parser.add_argument("--info", action="store_true",
                        help="仅列出 episode 基本信息，不做 FK 计算")
    args = parser.parse_args()

    folder = args.folder
    if folder is None:
        # 默认路径
        folder = "/home/ubuntu/code/data_archive/data/lerobot_groot_data/0526_test_for_fk"

    if not os.path.isdir(folder):
        print(f"错误: 文件夹不存在 — {folder}")
        sys.exit(1)

    if args.info:
        list_episodes(folder)
    else:
        run_analysis(folder, episode_filter=args.episodes,
                     verbose=args.verbose, output=args.output)


if __name__ == "__main__":
    main()
