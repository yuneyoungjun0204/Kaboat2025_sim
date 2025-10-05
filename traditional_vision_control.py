#!/usr/bin/env python3
"""
전통적인 컴퓨터 비전 기반 선박 자동 주차
OpenCV를 사용한 색상 검출 + 제어 로직

장점:
- 즉시 작동 (학습 불필요)
- 100% 해석 가능
- 매우 빠름 (<5ms)
- 데이터 수집 불필요
"""

import cv2
import numpy as np


class TraditionalVisionController:
    """전통적인 비전 기반 제어"""

    def __init__(self):
        # 빨간색 HSV 범위
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

        # 제어 파라미터
        self.center_tolerance = 50  # 중앙 허용 오차 (픽셀)
        self.min_area = 500  # 최소 도형 크기

    def detect_red_shape(self, image):
        """빨간 도형 검출"""
        # BGR → HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 빨간색 마스크 (빨강은 HSV에서 0-10, 160-180 두 범위)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = mask1 | mask2

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, mask

        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < self.min_area:
            return None, mask

        # 중심점 계산
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return {
            "center": (cx, cy),
            "area": area,
            "contour": largest_contour
        }, mask

    def calculate_action(self, image, detection_result):
        """액션 계산 [전진, 좌우, 선회]"""
        h, w = image.shape[:2]
        image_center_x = w // 2
        image_center_y = h // 2

        # 도형을 못 찾으면 정지
        if detection_result is None:
            return [0.0, 0.0, 0.0], "No red shape found"

        cx, cy = detection_result["center"]
        area = detection_result["area"]

        # 1. 선회 계산 (좌우 방향)
        x_error = cx - image_center_x
        rotation = np.clip(x_error / (w / 2), -1.0, 1.0)  # -1 ~ 1

        # 2. 전진 계산 (거리 기반)
        # 면적이 작으면 멀리 있음 → 전진
        # 면적이 크면 가까움 → 정지
        target_area = w * h * 0.1  # 이미지의 10%
        area_error = (target_area - area) / target_area
        forward = np.clip(area_error, -1.0, 1.0)

        # 3. 중앙에 도착했는지 확인
        if abs(x_error) < self.center_tolerance and area > target_area * 0.8:
            status = "STOP - TARGET REACHED"
            return [0.0, 0.0, 0.0], status

        # 4. 방향 결정
        if abs(x_error) > self.center_tolerance:
            if x_error > 0:
                status = f"Turn RIGHT (error: {x_error}px)"
            else:
                status = f"Turn LEFT (error: {x_error}px)"
        else:
            if forward > 0.3:
                status = f"Move FORWARD (dist: {area:.0f})"
            elif forward < -0.3:
                status = f"Move BACK (too close)"
            else:
                status = "ALIGNED - Fine tuning"

        # 좌우 이동 (보통 선박은 사용 안 함)
        lateral = 0.0

        return [forward, lateral, rotation], status

    def process_image(self, image):
        """전체 파이프라인"""
        # 1. 빨간 도형 검출
        detection, mask = self.detect_red_shape(image)

        # 2. 액션 계산
        action, status = self.calculate_action(image, detection)

        # 3. 시각화 (디버깅용)
        debug_image = image.copy()
        h, w = image.shape[:2]

        if detection:
            cx, cy = detection["center"]
            # 도형 중심 표시
            cv2.circle(debug_image, (cx, cy), 10, (0, 255, 0), -1)
            cv2.drawContours(debug_image, [detection["contour"]], -1, (0, 255, 0), 3)

        # 이미지 중앙 표시
        cv2.line(debug_image, (w//2, 0), (w//2, h), (255, 0, 0), 2)
        cv2.line(debug_image, (0, h//2), (w, h//2), (255, 0, 0), 2)

        # 상태 표시
        cv2.putText(debug_image, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_image, f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return action, status, debug_image, mask


def test_on_images():
    """테스트 이미지로 실행"""
    import os

    controller = TraditionalVisionController()

    test_images = [
        "test_img/Screenshot from 2025-10-03 10-15-26.png",
        "test_img/Screenshot from 2025-10-03 10-19-43.png",
        "test_img/Screenshot from 2025-10-03 10-24-31.png",
        "test_img/Screenshot from 2025-10-03 10-13-15.png",
    ]

    print("=" * 70)
    print("전통적인 비전 기반 제어 테스트")
    print("=" * 70)

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"이미지 없음: {img_path}")
            continue

        image = cv2.imread(img_path)
        action, status, debug_img, mask = controller.process_image(image)

        print(f"\n{os.path.basename(img_path)}")
        print(f"  상태: {status}")
        print(f"  액션: [전진={action[0]:+.2f}, 좌우={action[1]:+.2f}, 선회={action[2]:+.2f}]")

        # 결과 이미지 저장
        output_path = f"output_{os.path.basename(img_path)}"
        cv2.imwrite(output_path, debug_img)
        print(f"  저장: {output_path}")


def create_ros_node():
    """ROS2 노드 생성"""
    code = '''#!/usr/bin/env python3
"""
전통적인 비전 기반 ROS2 노드
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class TraditionalVisionNode(Node):
    def __init__(self):
        super().__init__('traditional_vision_node')

        # Controller
        from traditional_vision_control import TraditionalVisionController
        self.controller = TraditionalVisionController()

        # CV Bridge
        self.bridge = CvBridge()

        # ROS2 토픽
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Image, '/vision/debug_image', 10)

        self.get_logger().info('Traditional Vision Node 준비 완료!')

    def image_callback(self, msg):
        try:
            # ROS Image → OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 처리
            action, status, debug_img, mask = self.controller.process_image(cv_image)

            # Twist 메시지
            twist = Twist()
            twist.linear.x = float(action[0])
            twist.linear.y = float(action[1])
            twist.angular.z = float(action[2])
            self.cmd_pub.publish(twist)

            # 디버그 이미지 발행
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.debug_pub.publish(debug_msg)

            self.get_logger().info(f'{status} | Action: {action}')

        except Exception as e:
            self.get_logger().error(f'오류: {str(e)}')


def main():
    rclpy.init()
    node = TraditionalVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
'''

    with open("traditional_vision_ros_node.py", 'w') as f:
        f.write(code)

    print("\nROS2 노드 생성: traditional_vision_ros_node.py")


if __name__ == "__main__":
    # 테스트 실행
    test_on_images()

    # ROS 노드 생성
    create_ros_node()

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)
    print("\nROS 노드 실행:")
    print("  python3 traditional_vision_ros_node.py")
