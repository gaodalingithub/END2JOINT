#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import termios
import tty
import select

class ActibotKeyboardControl(Node):
    def __init__(self):
        super().__init__("actibot_keyboard_control")
        
        # 控制发布话题
        self.publisher_ = self.create_publisher(JointState, '/actibot_arm_ctrl', 10)
        
        self.names = [
            "[左臂] 关节 1", "[左臂] 关节 2", "[左臂] 关节 3", 
            "[左臂] 关节 4", "[左臂] 关节 5", "[左臂] 关节 6", "[左臂] 关节 7",
            "[右臂] 关节 1", "[右臂] 关节 2", "[右臂] 关节 3", 
            "[右臂] 关节 4", "[右臂] 关节 5", "[右臂] 关节 6", "[右臂] 关节 7",
            "[左手] 夹爪开合", "[右手] 夹爪开合"
        ]
        
        self.num_joints = 16 
        
        # 定义安全的初始零位 (14个手臂关节 + 2个夹爪)
        self.init_positions = [
            0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,   # 左臂 7 个
            -0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, # 右臂 7 个
            0.0, 0.0                             # 左右夹爪 2 个 (默认0.0)
        ]
        
        # 将当前位置初始化为安全零位
        self.positions = list(self.init_positions)
        
        self.selected_joint = 0
        self.step_size = 0.05 
        
        self.old_settings = termios.tcgetattr(sys.stdin)
        self._print_instructions()

    def _print_instructions(self):
        print("\n=== Actibot 键盘控制程序 (16通道模式) ===")
        print("w / s : 切换选择上一个/下一个控制点 (左臂 -> 右臂 -> 夹爪)")
        print("a / d : 控制当前选定通道的数值 (a='减少', d='增加')")
        print("z : 将所有数值重置为安全的初始零位")
        print("q : 退出程序")
        print("-" * 45)
        self._print_current_status()

    def _print_current_status(self):
        print(f"\r当前选中: {self.names[self.selected_joint]} (索引: {self.selected_joint}) | 当前数值: {self.positions[self.selected_joint]:.4f}   ", end="")

    def get_key(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

    def run(self):
        try:
            tty.setcbreak(sys.stdin.fileno())
            # 启动时先发送一次初始位置，防止底层没有收到指令
            self.publish_joint_command()
            
            while rclpy.ok():
                key = self.get_key()
                if key is not None:
                    # 切换控制通道
                    if key == 'w':
                        self.selected_joint = (self.selected_joint - 1) % self.num_joints
                        print("\n")
                        self._print_current_status()
                    elif key == 's':
                        self.selected_joint = (self.selected_joint + 1) % self.num_joints
                        print("\n")
                        self._print_current_status()
                    
                    # 调节数值
                    elif key == 'a':
                        self.positions[self.selected_joint] -= self.step_size
                        self.publish_joint_command()
                        self._print_current_status()
                    elif key == 'd':
                        self.positions[self.selected_joint] += self.step_size
                        self.publish_joint_command()
                        self._print_current_status()
                    
                    # 重置与退出
                    elif key == 'z':
                        # 恢复为你的安全初始位
                        self.positions = list(self.init_positions)
                        self.publish_joint_command()
                        print("\n[系统] 已将机械臂重置为安全初始位置")
                        self._print_current_status()
                    elif key == 'q':
                        print("\n[系统] 正在退出程序...")
                        break

                rclpy.spin_once(self, timeout_sec=0.01)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def publish_joint_command(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # 底层看的是16长度的数组，保持name填充即可
        msg.name = [f"joint_{i}" for i in range(self.num_joints)] 
        msg.position = self.positions
        msg.velocity = [0.0] * self.num_joints
        msg.effort = [0.0] * self.num_joints
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ActibotKeyboardControl()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()