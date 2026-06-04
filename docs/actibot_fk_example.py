import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time

from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

import os
import sys
import threading
import utils

exit_flag = False

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)

class Arm_IK:
    def __init__(self, urdf_path, smooth=False):
        # np.set_printoptions(precision=5, suppress=True, linewidth=200)
        
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=os.path.dirname(urdf_path))
        # self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
        self.joints_to_lock = ["LeftGripperA_Joint", "LeftGripperB_Joint", "RightGripperA_Joint", "RightGripperB_Joint"]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.joints_to_lock,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            pin.Frame('ee_left',
                      self.reduced_robot.model.getJointId('LeftWrist_Joint7'),
                      pin.SE3(
                          pin.Quaternion(1, 0, 0, 0),
                          np.array([0.15, -0.02, 0.0]),
                      ),
                      pin.FrameType.OP_FRAME)
        )

        self.reduced_robot.model.addFrame(
            pin.Frame('ee_right',
                      self.reduced_robot.model.getJointId('RightWrist_Joint7'),
                      pin.SE3(
                          pin.Quaternion(1, 0, 0, 0),
                          np.array([0.15, 0.02, 0.0]),
                      ),
                      pin.FrameType.OP_FRAME)
        )
        # self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION, package_dirs=os.path.dirname(urdf_path))
        # for i in range(4, 9):
        #     for j in range(0, 3):
        #         self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        # self.geometry_data = pin.GeometryData(self.geom_model)
        self.left_gripper_id = self.reduced_robot.model.getFrameId("ee_left")
        self.right_gripper_id = self.reduced_robot.model.getFrameId("ee_right")
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth = smooth



        # Creating Casadi models and data for symbolic computing
        self.model = pin.Model(self.reduced_robot.model)
        self.data  = self.model.createData()
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_left = casadi.SX.sym("tf_left", 4, 4)
        self.cTf_right = casadi.SX.sym("tf_right", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # # Get the hand joint ID and define the error function


    def get_fk_solution(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.left_gripper_id)
        pin.updateFramePlacement(self.model, self.data, self.right_gripper_id)
        # print(self.data.oMf[self.left_gripper_id])
        # pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        # collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        T_left = np.eye(4)
        T_left[:3, :3] = self.data.oMf[self.left_gripper_id].rotation
        T_left[:3, 3] = self.data.oMf[self.left_gripper_id].translation
        T_right = np.eye(4)
        T_right[:3, :3] = self.data.oMf[self.right_gripper_id].rotation
        T_right[:3, 3] = self.data.oMf[self.right_gripper_id].translation
        return T_left, T_right

def test():

    urdf_path = os.path.join(current_dir, 'actibot-v1-urdf-0929/urdf/actibot-v1-urdf-0929.urdf')
    ik = Arm_IK(urdf_path, visualize=False)
    # test fk 
    q = np.zeros(14)
    fk_left, fk_right = ik.get_fk_solution(q)
    print(fk_left)
    print(fk_right)






