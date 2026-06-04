import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger
from data_msgs.srv import CaptureService

class Collector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        # Service客户端（用于VR按键控制数据采集）
        self.capture_client = self.create_client(CaptureService, '/data_tools_dataCapture/capture_service')
        
        self.subscriber_joy = self.create_subscription(Joy, "/teleop/joy", self.joy_callback, 1)
        # VR按键状态记录（用于检测上升沿）
        self.last_button_x = False
        self.last_button_b = False
        self.is_capturing = False  # 记录当前采集状态

        # 回零


    def joy_callback(self, msg: Joy):
        
        if len(msg.axes) < 4 or len(msg.buttons) < 4:
            return
        # left_thumb_x = msg.axes[0]
        # left_thumb_y = msg.axes[1]
        # right_thumb_x = msg.axes[2]
        # right_thumb_y = msg.axes[3]
        # button_A = msg.buttons[0]
        button_B = msg.buttons[1]
        button_X = msg.buttons[2]
        # button_Y = msg.buttons[3]


        current_button_x = bool(button_X)
        current_button_b = bool(button_B)
        
        # 检测按键A上升沿：开始数据采集
        if current_button_x and not self.last_button_x:
            if not self.is_capturing:
                self.get_logger().info('VR Button X pressed: Starting data capture')
                self.call_capture_service(start=True, end=False)
                self.is_capturing = True
            else:
                self.get_logger().warn('Button X pressed but already capturing, ignoring')
        
        # 检测按键B上升沿：停止数据采集
        if current_button_b and not self.last_button_b:
            if self.is_capturing:
                self.get_logger().info('VR Button B pressed: Stopping data capture')
                self.call_capture_service(start=False, end=True)
                self.is_capturing = False
            else:
                self.get_logger().warn('Button B pressed but not capturing, ignoring')
        
        # 更新按键状态
        self.last_button_x = current_button_x
        self.last_button_b = current_button_b


    
    def call_capture_service(self, start=False, end=False):
        """调用数据采集Service"""
        if not self.capture_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Capture service not available')
            return
        
        request = CaptureService.Request()
        request.start = start
        request.end = end
        request.episode_index = -1  # 自动递增
        request.dataset_dir = ''
        request.instructions = '[null]'
        
        # 异步调用Service
        future = self.capture_client.call_async(request)
        future.add_done_callback(self.service_response_callback)
    
    def service_response_callback(self, future):
        """Service响应回调"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Capture service call succeeded')
            else:
                self.get_logger().error('Capture service call failed')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = Collector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

        

if __name__ == '__main__':
    main()