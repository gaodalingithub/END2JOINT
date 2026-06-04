#!/usr/bin/env python3
"""
批量计算 .parquet 数据的 FK 末端位姿，按原始编号保存完整结果。

输出文件（每帧一行）包含：
  - 原始 state (16维): 左臂7 + 右臂7 + 夹爪2
  - 左/右臂末端位姿: 位置(x,y,z) + 姿态(roll,pitch,yaw)
  - 时间戳等信息

用法:
  conda activate actibot_sdk
  python example/compute_fk_batch.py /path/to/data/folder -o /path/to/output

示例:
  python example/compute_fk_batch.py data/0602_test_for_net
  python example/compute_fk_batch.py data/0525_workflow_120 -o results/fk_output
  python example/compute_fk_batch.py data/0525_workflow_120 --format csv
"""
import sys, os, json, glob, argparse
import numpy as np
import pandas as pd

_example_dir = os.path.abspath(os.path.dirname(__file__))
if _example_dir not in sys.path:
    sys.path.insert(0, _example_dir)
_project_dir = os.path.abspath(os.path.join(_example_dir, ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from actibot_fk import Arm_IK
from pinocchio.rpy import matrixToRpy

URDF_PATH = os.path.join(_project_dir,
    "actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf")

COL_JOINTS_L = [f"L_{n}" for n in ("sh_pitch","sh_roll","sh_yaw",
                                    "el_pitch","el_roll","wr_yaw","wr_pitch")]
COL_JOINTS_R = [f"R_{n}" for n in ("sh_pitch","sh_roll","sh_yaw",
                                    "el_pitch","el_roll","wr_yaw","wr_pitch")]
COL_GRIPPER  = ["gripper_L", "gripper_R"]
COL_POS_L    = ["eeL_x", "eeL_y", "eeL_z"]
COL_RPY_L    = ["eeL_roll", "eeL_pitch", "eeL_yaw"]
COL_POS_R    = ["eeR_x", "eeR_y", "eeR_z"]
COL_RPY_R    = ["eeR_roll", "eeR_pitch", "eeR_yaw"]
COL_META     = ["episode_index", "frame_index", "timestamp"]

ALL_COLS = COL_META + COL_JOINTS_L + COL_JOINTS_R + COL_GRIPPER \
           + COL_POS_L + COL_RPY_L + COL_POS_R + COL_RPY_R


def find_parquet_files(folder):
    """递归搜索 folder 下的所有 episode_*.parquet"""
    patterns = [
        os.path.join(folder, "**", "episode_*.parquet"),
        os.path.join(folder, "data", "chunk-000", "episode_*.parquet"),
    ]
    for p in patterns:
        files = sorted(glob.glob(p, recursive=True))
        if files:
            return files
    # fallback: flat search
    files = sorted(glob.glob(os.path.join(folder, "episode_*.parquet")))
    return files


def process_episode(fpath, ik):
    """
    读取单个 parquet 文件，计算每帧 FK，返回 (DataFrame, episode_index)。
    """
    df_in = pd.read_parquet(fpath)
    ep = int(df_in.iloc[0]["episode_index"])
    N = len(df_in)

    rows = []
    for i in range(N):
        s = df_in.iloc[i]["observation.state"]
        ts = float(df_in.iloc[i]["timestamp"])
        fi = int(df_in.iloc[i]["frame_index"])

        # 组装 19 维 q
        q = np.zeros(19)
        q[5:12] = s[0:7]     # left arm
        q[12:19] = s[7:14]   # right arm

        # FK
        T_l, T_r = ik.get_fk_solution(q)
        pos_l = T_l[:3, 3]
        pos_r = T_r[:3, 3]
        rpy_l = matrixToRpy(T_l[:3, :3])
        rpy_r = matrixToRpy(T_r[:3, :3])

        row = {
            "episode_index": ep,
            "frame_index": fi,
            "timestamp": ts,
        }
        # left joint angles
        for j, col in enumerate(COL_JOINTS_L):
            row[col] = float(s[j])
        # right joint angles
        for j, col in enumerate(COL_JOINTS_R):
            row[col] = float(s[7 + j])
        # gripper
        row["gripper_L"] = float(s[14])
        row["gripper_R"] = float(s[15])
        # left EE pose
        for j, col in enumerate(COL_POS_L):
            row[col] = float(pos_l[j])
        for j, col in enumerate(COL_RPY_L):
            row[col] = float(rpy_l[j])
        # right EE pose
        for j, col in enumerate(COL_POS_R):
            row[col] = float(pos_r[j])
        for j, col in enumerate(COL_RPY_R):
            row[col] = float(rpy_r[j])

        rows.append(row)

    df_out = pd.DataFrame(rows, columns=ALL_COLS)
    return df_out, ep


def main():
    parser = argparse.ArgumentParser(
        description="批量计算 FK 末端位姿并保存完整结果")
    parser.add_argument("folder", help="数据文件夹路径（含 episode_*.parquet）")
    parser.add_argument("-o", "--output", default=None,
                        help="输出文件夹路径（默认: folder 同级目录下加 _fk_results）")
    parser.add_argument("-f", "--format", choices=["parquet", "csv"], default="parquet",
                        help="输出格式（默认 parquet）")
    parser.add_argument("-e", "--episodes", nargs="+", type=int, default=None,
                        help="只处理指定编号，如 -e 0 2 5")
    parser.add_argument("--merge", action="store_true",
                        help="合并所有 episode 为一个文件")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"错误: 文件夹不存在 — {args.folder}")
        sys.exit(1)

    # 确定输出路径
    if args.output:
        out_dir = args.output
    else:
        base = os.path.basename(args.folder.rstrip("/"))
        parent = os.path.dirname(args.folder.rstrip("/"))
        out_dir = os.path.join(parent, f"{base}_fk_results")

    os.makedirs(out_dir, exist_ok=True)

    # 搜索 parquet
    files = find_parquet_files(args.folder)
    if not files:
        print(f"错误: 在 {args.folder} 下未找到 episode_*.parquet 文件")
        sys.exit(1)

    # 过滤 episode 编号
    if args.episodes is not None:
        filtered = []
        for f in files:
            try:
                ep_num = int(os.path.basename(f).replace("episode_","").replace(".parquet",""))
                if ep_num in args.episodes:
                    filtered.append(f)
            except ValueError:
                pass
        files = filtered
        if not files:
            print(f"错误: 未找到指定编号的 episode")
            sys.exit(1)

    print(f"FK 引擎加载中...")
    ik = Arm_IK(URDF_PATH)
    print(f"找到 {len(files)} 个 episode 文件，输出至: {out_dir}")

    all_dfs = []
    for fpath in files:
        ep_num = int(os.path.basename(fpath).replace("episode_","").replace(".parquet",""))
        print(f"  处理 episode {ep_num:3d} ...", end=" ", flush=True)
        try:
            df_out, ep = process_episode(fpath, ik)
            ext = ".parquet" if args.format == "parquet" else ".csv"
            out_path = os.path.join(out_dir, f"episode_{ep:06d}_fk{ext}")
            if args.format == "parquet":
                df_out.to_parquet(out_path, index=False)
            else:
                df_out.to_csv(out_path, index=False, float_format="%.6f")
            print(f"{len(df_out):3d} 帧 → {os.path.basename(out_path)}")
            all_dfs.append(df_out)
        except Exception as e:
            print(f"失败: {e}")
            import traceback
            traceback.print_exc()

    # 合并文件
    if args.merge and all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        ext = ".parquet" if args.format == "parquet" else ".csv"
        merge_path = os.path.join(out_dir, f"all_episodes_fk{ext}")
        if args.format == "parquet":
            merged.to_parquet(merge_path, index=False)
        else:
            merged.to_csv(merge_path, index=False, float_format="%.6f")
        print(f"\n合并文件: {merge_path}  ({len(merged)} 帧)")

    print(f"\n完成! 共处理 {len(all_dfs)} 个 episode，结果保存在 {out_dir}")


if __name__ == "__main__":
    main()
