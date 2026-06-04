"""数据加载与预处理

样本构造:
  X = [ee_pose_t (12D), prev_joints_{t-1} (14D)]   → 26D
  y = joints_t (14D)
"""
import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config import hp, data_config

EE_COLS = data_config["col_eeL"] + data_config["col_eeR"]
JOINT_COLS = data_config["col_joints_l"] + data_config["col_joints_r"]


def load_episode_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*_fk.parquet")))
    episodes = {}
    for fpath in files:
        base = os.path.basename(fpath)
        ep = int(base.split("_")[1])
        episodes[ep] = pd.read_parquet(fpath)
    return episodes


def split_episodes(episodes, seed=hp["seed"]):
    eps = sorted(episodes.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(eps)
    n = len(eps)
    n_train = int(n * hp["train_ratio"])
    n_val = int(n * hp["val_ratio"])
    return (sorted(eps[:n_train]), sorted(eps[n_train:n_train + n_val]),
            sorted(eps[n_train + n_val:]))


def episodes_to_arrays(episodes_dict, ep_list, add_noise=False):
    """构造样本，可对 prev_joints 添加噪声模拟自回归推理误差。

    add_noise=True: 以 noise_prob 概率对 prev_joints 加高斯噪声，
    使模型学会容忍不精确的输入，降低自回归推理时的误差累积。
    """
    rng = np.random.RandomState()
    all_X, all_y = [], []
    for ep in ep_list:
        df = episodes_dict[ep]
        n = len(df)
        ee_t = df[EE_COLS].values.astype(np.float64)       # (n, 12)
        joints_t = df[JOINT_COLS].values.astype(np.float64)  # (n, 14)
        joints_prev = np.vstack([joints_t[0:1], joints_t[:-1]])  # (n, 14)

        if add_noise:
            # 随机选取部分帧，对 prev_joints 添加高斯噪声
            mask = rng.rand(n) < hp["noise_prob"]
            noise = rng.randn(*joints_prev.shape) * hp["noise_scale"]
            joints_prev[mask] = joints_prev[mask] + noise[mask]

        X = np.hstack([ee_t, joints_prev])  # (n, 26)
        all_X.append(X)
        all_y.append(joints_t)
    return np.vstack(all_X), np.vstack(all_y)


class Scaler:
    def __init__(self):
        self.X = StandardScaler()
        self.y = StandardScaler()

    def fit(self, X, y): self.X.fit(X); self.y.fit(y); return self
    def transform_X(self, X): return self.X.transform(X)
    def transform_y(self, y): return self.y.transform(y)
    def inverse_y(self, y_norm): return self.y.inverse_transform(y_norm)
    def inverse_X(self, X_norm): return self.X.inverse_transform(X_norm)


class IKDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def build_dataloaders(data_dir):
    episodes = load_episode_files(data_dir)
    train_eps, val_eps, test_eps = split_episodes(episodes)
    # 训练集加噪声模拟推理误差，验证/测试集不加
    X_train, y_train = episodes_to_arrays(episodes, train_eps, add_noise=True)
    X_val, y_val = episodes_to_arrays(episodes, val_eps, add_noise=False)
    X_test, y_test = episodes_to_arrays(episodes, test_eps, add_noise=False)
    scaler = Scaler().fit(X_train, y_train)

    def make_loader(X, y, shuffle):
        ds = IKDataset(scaler.transform_X(X), scaler.transform_y(y))
        return DataLoader(ds, batch_size=hp["batch_size"], shuffle=shuffle, pin_memory=True)

    info = {"train_eps": train_eps, "val_eps": val_eps, "test_eps": test_eps,
            "train_samples": len(X_train), "val_samples": len(X_val),
            "test_samples": len(X_test)}
    return (make_loader(X_train, y_train, True),
            make_loader(X_val, y_val, False),
            make_loader(X_test, y_test, False), scaler, info)
