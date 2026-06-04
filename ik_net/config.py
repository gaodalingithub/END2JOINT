"""IK-Net 超参数配置

使用方法:
  from config import hp, paths, data_config
"""


# ═══════════════════════════════════════════════════════════════
#  超参数 (HYPERPARAMETERS) — 与模型训练相关，常修改
# ═══════════════════════════════════════════════════════════════

hp = {

    # ── 数据 ──
    "input_dim": 26,          # 输入: [eeL_xyzrpy(6), eeR_xyzrpy(6), prev_joints_L(7), prev_joints_R(7)]
    "output_dim": 14,         # 输出: [joints_L(7), joints_R(7)]

    # ── 模型架构 ──
    "hidden_dims": [400, 300, 200, 100, 50],   # 各隐藏层神经元数
    "dropout": 0.1,                             # Dropout 比率
    "use_residual": True,                  # True = 残差连接, False = 直接预测

    # ── 训练 ──
    "batch_size": 256,          # 批大小
    "learning_rate": 1e-3,      # 初始学习率
    "weight_decay": 1e-5,       # L2 正则化系数
    "num_epochs": 10000,          # 最大训练轮数
    "lr_step": 50,              # 学习率衰减间隔 (epoch)
    "lr_gamma": 0.5,            # 学习率衰减系数
    "patience": 100,             # 早停容忍轮数
    "target_joint_deg": 0.1,    # 目标关节角平均误差（度），达标即停
    "ckpt_dir": "/home/ubuntu/code/End2Joint/ik_net/results",  # 模型权重 + 归一化参数路径

    # ── 数据划分 ──
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    # ── 随机种子 ──
    "seed": 42,
}


# ═══════════════════════════════════════════════════════════════
#  路径配置 (PATHS) — 与环境相关，很少修改
# ═══════════════════════════════════════════════════════════════

paths = {
    "data_dir": "/home/ubuntu/code/End2Joint/data/0525_workflow_120_action_fk",
    "results_dir": "/home/ubuntu/code/End2Joint/ik_net/end2action_model",
    "project_dir": "/home/ubuntu/code/End2Joint",
    "urdf_path": "actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf",
}


# ═══════════════════════════════════════════════════════════════
#  数据字段名 (DATA FIELDS) — 与数据列名相关
# ═══════════════════════════════════════════════════════════════

data_config = {
    "col_eeL": ["eeL_x", "eeL_y", "eeL_z", "eeL_roll", "eeL_pitch", "eeL_yaw"],
    "col_eeR": ["eeR_x", "eeR_y", "eeR_z", "eeR_roll", "eeR_pitch", "eeR_yaw"],
    "col_joints_l": ["L_sh_pitch", "L_sh_roll", "L_sh_yaw",
                     "L_el_pitch", "L_el_roll", "L_wr_yaw", "L_wr_pitch"],
    "col_joints_r": ["R_sh_pitch", "R_sh_roll", "R_sh_yaw",
                     "R_el_pitch", "R_el_roll", "R_wr_yaw", "R_wr_pitch"],
}
