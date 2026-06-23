#!/usr/bin/env python3
"""
关节状态接收测试服务端。

只接收机器人的观测数据，提取 joint state 并打印诊断信息，
不给机器人发送任何控制信号。

用法:
  # 启动测试服务端
  python server/test_joint_state_receiver.py --port 5555

  # 机器人端连接（指令任意）
  python main_actibot.py \
    --policy-host 192.168.0.223 \
    --policy-port 5555 \
    --instruction "test" \
    --open-loop-horizon 1
"""

import argparse
import datetime
import os
import sys
import time

import msgpack_numpy as mnp
import numpy as np
import zmq


# ── 从 ik_validation_server.py 复用的关节提取逻辑 ──
# （独立脚本，不依赖 ik_validation_server.py，方便单独测试）

STATE_KEYS = [
    "left_arm_joint_positions",
    "right_arm_joint_positions",
    "left_gripper_position",
    "right_gripper_position",
]


def flatten_joint_value(val):
    """将任意嵌套结构的关节值展平为一维 float64 数组。"""
    # 处理 dict 格式的编码数组
    if isinstance(val, dict):
        # 格式1: {'__ndarray_class__': True, 'as_npy': b'\x93NUMPY...'}
        # PolicyClient 使用 NPY 格式编码 numpy 数组
        as_npy = val.get('as_npy')
        if as_npy is not None and isinstance(as_npy, (bytes, bytearray)):
            import io
            arr = np.load(io.BytesIO(as_npy))
            arr = arr.squeeze()
            if arr.ndim == 1 and arr.dtype.kind in ('f', 'i', 'u'):
                return arr.astype(np.float64)
            return arr.ravel().astype(np.float64)

        # 格式2: msgpack_numpy 标准格式 {'nd': 3, 'type': 'float32', 'shape': [...], 'data': b'...'}
        for data_key in ('data', b'data'):
            raw = val.get(data_key)
            if raw is not None:
                dtype_str = val.get('type', val.get(b'type', 'float32'))
                shape = val.get('shape', val.get(b'shape', [-1]))
                if isinstance(dtype_str, bytes):
                    dtype_str = dtype_str.decode()
                if isinstance(shape, (list, tuple)):
                    arr = np.frombuffer(raw, dtype=dtype_str).reshape(shape)
                else:
                    arr = np.frombuffer(raw, dtype=dtype_str)
                arr = arr.squeeze()
                if arr.ndim == 1 and arr.dtype.kind in ('f', 'i', 'u'):
                    return arr.astype(np.float64)
                return arr.ravel().astype(np.float64)

    # numpy 数组 / 嵌套列表
    arr = np.asarray(val)
    if arr.ndim == 1 and arr.dtype.kind in ('f', 'i', 'u'):
        return arr.astype(np.float64)
    arr = np.asarray(val).squeeze()
    if arr.ndim == 1 and arr.dtype.kind in ('f', 'i', 'u'):
        return arr.astype(np.float64)
    if arr.dtype.kind == 'O' or arr.ndim > 1:
        try:
            flat = np.concatenate([np.asarray(x).ravel() for x in np.asarray(val).ravel()])
            if flat.dtype.kind in ('f', 'i', 'u'):
                return flat.astype(np.float64)
        except Exception:
            pass
    import re
    nums = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', str(val))
    if nums:
        return np.array([float(n) for n in nums], dtype=np.float64)
    return np.array([], dtype=np.float64)


def extract_joints(observation):
    """从 observation 中提取 joint_state (16D)。"""
    state = observation.get("state", {})
    if not state:
        return None, "observation 中没有 state 字段"

    results = {}
    for key in STATE_KEYS:
        raw_val = state.get(key)
        if raw_val is None:
            results[key] = (None, "字段缺失")
            continue
        flat = flatten_joint_value(raw_val)
        results[key] = (flat, f"shape={flat.shape}, dtype={flat.dtype}")

    # 检查能否拼出 16D
    try:
        left_arm = flatten_joint_value(state.get("left_arm_joint_positions", []))
        right_arm = flatten_joint_value(state.get("right_arm_joint_positions", []))
        left_grip = flatten_joint_value(state.get("left_gripper_position", []))
        right_grip = flatten_joint_value(state.get("right_gripper_position", []))

        if len(left_arm) < 7:
            return results, f"left_arm 长度不足: {len(left_arm)}"
        if len(right_arm) < 7:
            return results, f"right_arm 长度不足: {len(right_arm)}"

        joints_16 = np.concatenate([
            left_arm[-7:], right_arm[-7:],
            left_grip[-1:] if len(left_grip) > 0 else [0.0],
            right_grip[-1:] if len(right_grip) > 0 else [0.0],
        ]).astype(np.float64)
        return results, joints_16
    except Exception as e:
        return results, f"拼接失败: {e}"


def print_request_structure(req, prefix=""):
    """递归打印请求结构的前几层，用于理解数据格式。"""
    if isinstance(req, dict):
        print(f"{prefix}dict with keys: {list(req.keys())}")
        for k in list(req.keys())[:5]:  # 最多打印 5 个键
            v = req[k]
            if isinstance(v, (dict, list)):
                print_request_structure(v, prefix + f"  [{k}].")
            elif isinstance(v, np.ndarray):
                print(f"{prefix}  [{k}]: ndarray shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"{prefix}  [{k}]: {type(v).__name__}")
    elif isinstance(req, list):
        print(f"{prefix}list len={len(req)}")
        if len(req) > 0 and isinstance(req[0], (dict, list)):
            print_request_structure(req[0], prefix + "  [0].")


def main():
    parser = argparse.ArgumentParser(description="关节状态接收测试服务端")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=5555, help="绑定端口")
    parser.add_argument("--count", type=int, default=10,
                        help="接收帧数后自动退出 (0=持续运行)")
    parser.add_argument("--print-structure", action="store_true",
                        help="首次请求时打印完整结构")
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[Server] 监听 tcp://{args.host}:{args.port}")
    print(f"[Server] 等待机器人连接...")
    print(f"[Server] 启动命令: python main_actibot.py "
          f"--policy-host <本机IP> --policy-port {args.port} "
          f"--instruction test --open-loop-horizon 1")
    print()

    frame_count = 0
    structure_printed = False
    running = True

    while running:
        try:
            msg = sock.recv()
            req = mnp.unpackb(msg, raw=False)
        except Exception as e:
            print(f"[Error] recv: {e}")
            continue

        ep = req.get("endpoint", "unknown")
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if ep == "ping":
            sock.send(mnp.packb({"status": "ok", "test_server": True}))
            print(f"[{ts}] Ping ← 机器人")
            continue

        if ep != "get_action":
            sock.send(mnp.packb({"status": "ok", "test_server": True}))
            print(f"[{ts}] 跳过端点: {ep}")
            continue

        frame_count += 1
        print(f"\n{'='*60}")
        print(f"[{ts}] 第 {frame_count} 帧 get_action ← 机器人")

        # ── 打印请求顶层结构（仅第一次）──
        if args.print_structure and not structure_printed:
            print(f"\n[Structure] 请求结构:")
            print_request_structure(req)
            structure_printed = True

        # ── 尝试各种路径提取 state ──
        joint_state_data = None

        # 路径1: req.data.state
        path1 = req.get("data", {}).get("state")
        if path1:
            joint_state_data = path1
            print(f"  [路径] req.data.state ✓")
        else:
            # 路径2: req.data.observation.state
            path2 = req.get("data", {}).get("observation", {}).get("state")
            if path2:
                joint_state_data = path2
                print(f"  [路径] req.data.observation.state ✓")
            else:
                # 路径3: req.state
                path3 = req.get("state")
                if path3:
                    joint_state_data = path3
                    print(f"  [路径] req.state ✓")
                else:
                    # 路径4: req.data.observation 自身含有关节字段
                    obs = req.get("data", {}).get("observation", {})
                    if isinstance(obs, dict) and any(k in obs for k in STATE_KEYS):
                        joint_state_data = obs
                        print(f"  [路径] req.data.observation (顶层) ✓")

        # ── 提取关节 ──
        obs_wrapper = {"state": joint_state_data or {}}
        detail, result = extract_joints(obs_wrapper)

        # ── 打印每个字段的详细信息 ──
        if detail:
            for key in STATE_KEYS:
                val, info = detail.get(key, (None, "未检查"))
                if isinstance(val, np.ndarray):
                    print(f"  {key}: {info}")
                    print(f"    values: [{', '.join(f'{v:.4f}' for v in val[:5])}{'...' if len(val) > 5 else ''}]")
                else:
                    print(f"  {key}: {info}")

        # ── 打印原始 state 值，帮助诊断解码问题 ──
        if joint_state_data and isinstance(joint_state_data, dict):
            print(f"  [Raw] 原始 state 值:")
            for k in STATE_KEYS:
                raw = joint_state_data.get(k)
                if raw is None:
                    print(f"    {k}: 缺失")
                elif isinstance(raw, dict):
                    print(f"    {k}: dict keys={list(raw.keys())}")
                    for dk, dv in raw.items():
                        if isinstance(dv, (bytes, bytearray)):
                            print(f"      {dk}: bytes len={len(dv)}")
                        elif isinstance(dv, (list, tuple)):
                            print(f"      {dk}: list len={len(dv)} {dv[:3]}")
                        elif isinstance(dv, np.ndarray):
                            print(f"      {dk}: ndarray shape={dv.shape}")
                        else:
                            print(f"      {dk}: {type(dv).__name__} = {dv}")
                elif isinstance(raw, np.ndarray):
                    print(f"    {k}: ndarray shape={raw.shape}, dtype={raw.dtype}, "
                          f"val={raw.ravel()[:8]}")
                elif isinstance(raw, list):
                    print(f"    {k}: list len={len(raw)}, type(0)={type(raw[0]).__name__}")
                else:
                    print(f"    {k}: {type(raw).__name__}")

        # ── 判断是否成功 ──
        if isinstance(result, np.ndarray) and len(result) == 16:
            print(f"  ✅ 关节状态提取成功! 16D joint_state:")
            joint_names = [
                "L_sh_pitch", "L_sh_roll", "L_sh_yaw",
                "L_el_pitch", "L_el_roll", "L_wr_yaw", "L_wr_pitch",
                "R_sh_pitch", "R_sh_roll", "R_sh_yaw",
                "R_el_pitch", "R_el_roll", "R_wr_yaw", "R_wr_pitch",
                "gripper_L", "gripper_R",
            ]
            for name, v in zip(joint_names, result):
                print(f"    {name:>15s}: {v:8.4f}")
            print(f"  [范围] min={result.min():.4f}, max={result.max():.4f}")
        else:
            print(f"  ❌ 关节状态提取失败: {result}")
            # 打印原始值帮助调试
            if joint_state_data:
                print(f"  [Raw] state keys: {list(joint_state_data.keys())}")
                for k in joint_state_data:
                    v = joint_state_data[k]
                    print(f"    {k}: type={type(v).__name__}", end="")
                    if isinstance(v, dict):
                        print(f", dict_keys={list(v.keys())}", end="")
                    elif isinstance(v, np.ndarray):
                        print(f", shape={v.shape}, dtype={v.dtype}", end="")
                    print()
            else:
                print(f"  [Raw] 未找到任何 state 字段")

        # ── 响应（不发送控制信号）──
        sock.send(mnp.packb({
            "status": "test_mode_no_control",
            "frame": frame_count,
            "joint_state_extracted": isinstance(result, np.ndarray) and len(result) == 16,
        }))
        print(f"  [响应] test_mode_no_control (不发送控制信号)")

        # ── 到达计数后退出 ──
        if args.count > 0 and frame_count >= args.count:
            print(f"\n[Server] 已接收 {args.count} 帧，自动退出")
            running = False

    sock.close()
    ctx.term()
    print(f"[Server] 测试结束，共接收 {frame_count} 帧")
    print(f"[Server] 提示: 如果关节状态提取成功，说明通信链路和格式完全正确")
    print(f"[Server] 提示: 如果提取失败，检查 msgpack_numpy 版本和 PolicyClient 序列化方式")


if __name__ == "__main__":
    main()
