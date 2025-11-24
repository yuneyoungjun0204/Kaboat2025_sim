#!/usr/bin/env python3
"""
PX4 VehicleLocalPosition을 GPS 좌표로 변환하는 노드
- NED 좌표계를 LLA (Latitude, Longitude, Altitude)로 변환
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleLocalPosition
from std_msgs.msg import Float64MultiArray
import math
import time

class LocalPositionToGPS(Node):
    def __init__(self):
        super().__init__('local_position_to_gps')
        
        # PX4용 QoS 프로파일 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # PX4는 Best Effort 사용
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 구독 (QoS 적용)
        self.subscription = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos_profile)
        
        # GPS 발행 (QoS 적용) - Float64MultiArray로 발행 [lat, lon, alt]
        self.gps_publisher = self.create_publisher(
            Float64MultiArray,
            '/converted_gps_position',
            qos_profile)
        
        # 상태 변수
        self.last_log_time = 0.0
        self.log_interval = 1.0  # 1초마다 로그 출력
        self.last_ref_lat = None
        self.last_ref_lon = None
        self.message_count = 0
        self.valid_message_count = 0
        
        self.get_logger().info('✅ Local Position to GPS converter started')
    
    def position_callback(self, msg):
        """VehicleLocalPosition 메시지 콜백"""
        self.message_count += 1
        
        # xy_global과 z_global이 True인지 확인
        if not msg.xy_global or not msg.z_global:
            if self.message_count % 100 == 0:  # 100번마다 경고
                self.get_logger().warn(
                    f'⚠️ Position not in global frame (xy_global={msg.xy_global}, '
                    f'z_global={msg.z_global}), skipping...'
                )
            return
        
        # 기준점 유효성 검사
        ref_lat = msg.ref_lat
        ref_lon = msg.ref_lon
        ref_alt = msg.ref_alt
        
        # ref_lat, ref_lon이 유효한 범위인지 확인 (-90~90, -180~180)
        if not (-90.0 <= ref_lat <= 90.0) or not (-180.0 <= ref_lon <= 180.0):
            if self.message_count % 100 == 0:
                self.get_logger().warn(
                    f'⚠️ Invalid reference point: ref_lat={ref_lat:.6f}°, '
                    f'ref_lon={ref_lon:.6f}°, skipping...'
                )
            return
        
        # 기준점이 변경되었는지 확인
        if (self.last_ref_lat is None or self.last_ref_lon is None or
            abs(self.last_ref_lat - ref_lat) > 0.0001 or
            abs(self.last_ref_lon - ref_lon) > 0.0001):
            self.get_logger().info(
                f'📍 Reference point updated: lat={ref_lat:.8f}°, '
                f'lon={ref_lon:.8f}°, alt={ref_alt:.2f}m'
            )
            self.last_ref_lat = ref_lat
            self.last_ref_lon = ref_lon
        
        # 로컬 좌표 (NED 좌표계)
        x = msg.x  # 북쪽 방향 (미터)
        y = msg.y  # 동쪽 방향 (미터)
        z = msg.z  # 아래쪽 방향 (미터)
        
        # NaN/무한대 값 체크
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            if self.message_count % 100 == 0:
                self.get_logger().warn(
                    f'⚠️ Invalid NED coordinates: x={x}, y={y}, z={z}, skipping...'
                )
            return
        
        try:
            # GPS 좌표로 변환
            lat, lon = self.local_to_gps(x, y, ref_lat, ref_lon)
            alt = ref_alt - z  # NED에서 z는 Down이므로 빼줌
            
            # 변환된 좌표 유효성 검사
            if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
                if self.message_count % 100 == 0:
                    self.get_logger().warn(
                        f'⚠️ Invalid GPS coordinates: lat={lat:.8f}°, '
                        f'lon={lon:.8f}°, skipping...'
                    )
                return
            
            # NaN/무한대 값 체크
            if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(alt)):
                if self.message_count % 100 == 0:
                    self.get_logger().warn(
                        f'⚠️ Invalid GPS values (NaN/Inf): lat={lat}, lon={lon}, alt={alt}'
                    )
                return
            
        except Exception as e:
            self.get_logger().error(f'❌ GPS conversion error: {e}')
            return
        
        # Float64MultiArray로 GPS 좌표 발행 [lat, lon, alt]
        gps_msg = Float64MultiArray()
        gps_msg.data = [float(lat), float(lon), float(alt)]
        
        # GPS 메시지 발행
        self.gps_publisher.publish(gps_msg)
        self.valid_message_count += 1
        
        # 주기적 로그 출력 (1초마다)
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            speed = math.sqrt(msg.vx**2 + msg.vy**2) if math.isfinite(msg.vx) and math.isfinite(msg.vy) else 0.0
            self.get_logger().info(
                f'📍 GPS: lat={lat:.8f}°, lon={lon:.8f}°, alt={alt:.2f}m | '
                f'NED: x={x:.2f}m, y={y:.2f}m, z={z:.2f}m | '
                f'Speed: {speed:.2f} m/s | '
                f'Valid: {self.valid_message_count}/{self.message_count}'
            )
            self.last_log_time = current_time
    
    def local_to_gps(self, x, y, ref_lat, ref_lon):
        """
        로컬 NED 좌표를 GPS 좌표로 변환
        
        Args:
            x: 북쪽 방향 거리 (미터) - NED의 North
            y: 동쪽 방향 거리 (미터) - NED의 East
            ref_lat: 기준점 위도 (도)
            ref_lon: 기준점 경도 (도)
        
        Returns:
            (latitude, longitude) in degrees
        
        Note:
            간단한 평면 근사 공식 사용 (작은 거리에서 정확함)
            더 정확한 변환이 필요하면 pyproj 사용 권장
        """
        # 지구 반지름 (WGS84)
        EARTH_RADIUS = 6378137.0  # 미터
        
        # 위도 변환 (1도 ≈ 111,320m)
        d_lat_rad = x / EARTH_RADIUS
        lat = ref_lat + math.degrees(d_lat_rad)
        
        # 경도 변환 (위도에 따라 달라짐)
        # 1도 경도 ≈ 111,320 * cos(위도) m
        d_lon_rad = y / (EARTH_RADIUS * math.cos(math.radians(ref_lat)))
        lon = ref_lon + math.degrees(d_lon_rad)
        
        return lat, lon

def main(args=None):
    rclpy.init(args=args)
    node = LocalPositionToGPS()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()