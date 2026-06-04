#!/usr/bin/env python3
"""
订阅 actibot_arm_state（+ body/lift），实时计算末端 6D 位姿并发布。

发布话题：
  /actibot/ee_left_pose   — geometry_msgs/PoseStamped  左臂末端
  /actibot/ee_right_pose  — geometry_msgs/PoseStamped  右臂末端
  /actibot/ee_poses       — std_msgs/String            文本 6D，方便 echo 查看

用法：
  conda activate actibot_sdk
  python ee_pose_node.py
"""
import sys, os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import numpy as np
from pinocchio.rpy import matrixToRpy

# 加载 FK
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from example.actibot_fk import Arm_IK

URDF_PATH = "actibot_sdk/robot_description/v3/urdf/v3_urdf_251121-2.urdf"


class EEPoseNode(Node):
    def __init__(self):
        super().__init__("ee_pose_node")

        # ── FK 引擎 ──
        self.ik = Arm_IK(URDF_PATH)
        self.NQ = self.ik.reduced_robot.model.nq  # 19

        # ── 当前关节状态缓存 ──
        # arm:  [L7, R7, gripper2] = 16 维
        # body: [waist_yaw, waist_pitch, head_yaw, head_pitch]
        # lift: [height]
        self.arm_q  = np.zeros(16)
        self.body_q = np.zeros(4)
        self.lift_q = 0.0
        self.have_arm  = False
        self.have_body = False
        self.have_lift = False

        # ── 发布器 ──
        self.pub_left  = self.create_publisher(PoseStamped, "/actibot/ee_left_pose", 10)
        self.pub_right = self.create_publisher(PoseStamped, "/actibot/ee_right_pose", 10)
        self.pub_text  = self.create_publisher(String, "/actibot/ee_poses", 10)

        # ── 订阅器（只要有 arm_state 就计算，body/lift 可选）──
        self.create_subscription(JointState, "actibot_arm_state",  self._cb_arm,  10)
        self.create_subscription(JointState, "actibot_body_state", self._cb_body, 10)
        self.create_subscription(JointState, "actibot_lift_state", self._cb_lift, 10)

        self.get_logger().info("ee_pose_node 已启动，等待话题数据...")

    # ── 回调 ──
    def _cb_arm(self, msg):
        self.arm_q = np.array(msg.position[:16])  # 14 arm + 2 gripper
        self.have_arm = True
        self._compute_and_publish()

    def _cb_body(self, msg):
        self.body_q = np.array(msg.position[:4])  # waist_yaw/pitch, head_yaw/pitch
        self.have_body = True

    def _cb_lift(self, msg):
        self.lift_q = msg.position[0] if msg.position else 0.0
        self.have_lift = True

    # ── FK 计算 + 发布 ──
    def _compute_and_publish(self):
        if not self.have_arm:
            return

        # 拼装 19 维 q
        q = np.zeros(self.NQ)
        q[0] = self.lift_q                           # up_down
        q[1] = self.body_q[0]                        # waist_yaw
        q[2] = self.body_q[1]                        # waist_pitch
        q[3] = self.body_q[2]                        # head_yaw
        q[4] = self.body_q[3]                        # head_pitch
        q[5:12]  = self.arm_q[0:7]                   # left arm  7
        q[12:19] = self.arm_q[7:14]                  # right arm 7

        # FK
        T_left, T_right = self.ik.get_fk_solution(q)

        now = self.get_clock().now().to_msg()

        # 4×4 → PoseStamped
        left_pose  = self._to_pose_stamped(T_left,  "actibot_base", now)
        right_pose = self._to_pose_stamped(T_right, "actibot_base", now)
        self.pub_left.publish(left_pose)
        self.pub_right.publish(right_pose)

        # 同时发文本 6D，方便 ros2 topic echo 查看
        rpy_l = np.degrees(matrixToRpy(T_left[:3, :3]))
        rpy_r = np.degrees(matrixToRpy(T_right[:3, :3]))
        txt = (
            f"[左] x={T_left[0,3]:.4f} y={T_left[1,3]:.4f} z={T_left[2,3]:.4f} "
            f"r={rpy_l[0]:.2f} p={rpy_l[1]:.2f} y={rpy_l[2]:.2f}   |   "
            f"[右] x={T_right[0,3]:.4f} y={T_right[1,3]:.4f} z={T_right[2,3]:.4f} "
            f"r={rpy_r[0]:.2f} p={rpy_r[1]:.2f} y={rpy_r[2]:.2f}"
        )
        self.pub_text.publish(String(data=txt))
        self.get_logger().info(txt)

    @staticmethod
    def _to_pose_stamped(T, frame_id, stamp):
        p = PoseStamped()
        p.header.frame_id = frame_id
        p.header.stamp = stamp
        p.pose.position.x = float(T[0, 3])
        p.pose.position.y = float(T[1, 3])
        p.pose.position.z = float(T[2, 3])
        # 4×4 → 四元数（PyPinocchio 的 SE3 → 原生 numpy）
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        p.pose.orientation.x = quat[0]
        p.pose.orientation.y = quat[1]
        p.pose.orientation.z = quat[2]
        p.pose.orientation.w = quat[3]
        return p


def main():
    rclpy.init()
    node = EEPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
