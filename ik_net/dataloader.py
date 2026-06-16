"""数据加载与预处理

样本构造:
  X = [ee_pose_t(12), state_{t-1}(14)]  → 26D
  y = action_t                          → 14D
模型: action_t_pred = state_{t-1} + delta
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


def episodes_to_arrays(episodes_ee, episodes_state, ep_list):
    """构造样本。

    X = [ee_pose_t(12), state_{t-1}(14)]  → 26D    (ee + prev_joints from action/state FK)
    y = action_t                           → 14D    (control signal from action FK)
    模型: action_t_pred = state_{t-1} + delta
    """
    all_X, all_y = [], []
    for ep in ep_list:
        df_ee = episodes_ee[ep]
        df_st = episodes_state[ep]
        ee_t = df_ee[EE_COLS].values.astype(np.float64)          # (n, 12) ← action FK
        action = df_ee[JOINT_COLS].values.astype(np.float64)      # (n, 14) ← action FK
        state = df_st[JOINT_COLS].values.astype(np.float64)       # (n, 14) ← state FK
        state_prev = np.vstack([state[0:1], state[:-1]])           # (n, 14) ← state_{t-1}
        X = np.hstack([ee_t, state_prev])  # (n, 26)
        all_X.append(X)
        all_y.append(action.copy())
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
    """加载数据。

    ee_pose 和 action 来自 data_dir（action FK），
    prev_joints 来自 joint_state_dir（state FK）。
    """
    from config import paths

    episodes_ee = load_episode_files(data_dir)
    state_dir = paths.get("joint_state_dir", data_dir)
    episodes_state = load_episode_files(state_dir)
    if state_dir != data_dir:
        print(f"  ee/action: {data_dir}  |  prev_state: {state_dir}")

    # 合并额外数据到两个 dict
    extra_dirs = paths.get("extra_data_dirs", [])
    if extra_dirs:
        ep_offset = max(max(episodes_ee.keys()), max(episodes_state.keys())) + 1
        for edir in extra_dirs:
            for ep_idx, df in load_episode_files(edir).items():
                episodes_ee[ep_offset + ep_idx] = df
                episodes_state[ep_offset + ep_idx] = df
        print(f"  合并额外数据: {sum(1 for d in extra_dirs for _ in load_episode_files(d))} episodes")

    train_eps, val_eps, test_eps = split_episodes(episodes_ee)
    X_train, y_train = episodes_to_arrays(episodes_ee, episodes_state, train_eps)
    X_val, y_val = episodes_to_arrays(episodes_ee, episodes_state, val_eps)
    X_test, y_test = episodes_to_arrays(episodes_ee, episodes_state, test_eps)
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
