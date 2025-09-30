# 파일명: wamv_keyboard_controller.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import sys, select, termios, tty

# 키보드 조작 안내 메시지
msg = """
---------------------------
WAM-V Keyboard Controller
---------------------------
Moving around:
   w
a  s  d
   x

w/x : increase/decrease forward thrust
a/d : increase/decrease turning thrust (yaw)
s   : emergency stop (all thrust to zero)

CTRL-C to quit
---------------------------
"""

# 키보드 입력에 따른 추력 변화량
THRUST_STEP = 10.0

class WamvKeyboardController(Node):
    def __init__(self):
        super().__init__('wamv_keyboard_controller')
        
        # 추진기 제어 퍼블리셔
        self.left_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)

        # 현재 추력 상태 변수
        self.forward_thrust = 0.0
        self.turn_thrust = 0.0
        
        # 터미널 설정 저장
        self.settings = termios.tcgetattr(sys.stdin)
        self.get_logger().info("Keyboard controller node has been started.")

    def getKey(self):
        """터미널에서 키 입력을 받는 함수"""
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run_controller(self):
        """메인 컨트롤 루프"""
        print(msg)
        while rclpy.ok():
            key = self.getKey()
            
            # 'w' 키: 전진 추력 증가
            if key == 'w':
                self.forward_thrust += THRUST_STEP
            # 'x' 키: 전진 추력 감소
            elif key == 'x':
                self.forward_thrust -= THRUST_STEP
            # 'a' 키: 좌회전 추력 증가 (오른쪽은 +, 왼쪽은 -)
            elif key == 'a':
                self.turn_thrust += THRUST_STEP
            # 'd' 키: 우회전 추력 증가 (오른쪽은 -, 왼쪽은 +)
            elif key == 'd':
                self.turn_thrust -= THRUST_STEP
            # 's' 키: 모든 추력 0으로 (긴급 정지)
            elif key == 's':
                self.forward_thrust = 0.0
                self.turn_thrust = 0.0
            # Ctrl+C (종료)
            elif key == '\x03':
                break
            
            # 계산된 추력을 각 추진기에 분배
            # 전진: 양쪽 같은 힘 / 회전: 양쪽 반대 힘
            left_command = self.forward_thrust - self.turn_thrust
            right_command = self.forward_thrust + self.turn_thrust
            
            # 메시지 생성 및 발행
            left_msg = Float64()
            left_msg.data = left_command
            
            right_msg = Float64()
            right_msg.data = right_command
            
            self.left_thrust_pub.publish(left_msg)
            self.right_thrust_pub.publish(right_msg)
            
            # 현재 상태 출력
            self.get_logger().info(f'Forward: {self.forward_thrust:.1f}, Turn: {self.turn_thrust:.1f} | Left Cmd: {left_command:.1f}, Right Cmd: {right_command:.1f}')

def main(args=None):
    rclpy.init(args=args)
    controller_node = WamvKeyboardController()
    
    try:
        controller_node.run_controller()
    except Exception as e:
        print(e)
    finally:
        # 종료 시 로봇 정지
        stop_msg = Float64()
        stop_msg.data = 0.0
        controller_node.left_thrust_pub.publish(stop_msg)
        controller_node.right_thrust_pub.publish(stop_msg)
        
        # 터미널 설정 복원
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, controller_node.settings)
        
        controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()