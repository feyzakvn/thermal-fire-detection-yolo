import rclpy
from rclpy.node import Node
from edge_ai_interfaces.msg import FusedData
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
import numpy as np
import csv
import os
import cv2
from datetime import datetime
from cv_bridge import CvBridge

THERMAL_HOT       = 45.0
THERMAL_DANGEROUS = 60.0
#DEPTH_TOO_CLOSE   = 50
#DEPTH_MAX_VALID   = 300
DEPTH_TOO_CLOSE   = 30
DEPTH_MAX_VALID   = 1500

CSV_PATH    = '/workspace/logs/decision_log.csv'
BITMAP_PATH = '/workspace/logs/thermal_bitmap_latest.png'


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')
        self.bridge = CvBridge()
        self.last_bitmap_time = self.get_clock().now()

        self.fused_sub   = self.create_subscription(
            FusedData, '/fused/output_v2', self.decision_callback, 10)
        self.bitmap_pub  = self.create_publisher(Image,   '/decision/thermal_bitmap', 10)
        self.alert_pub   = self.create_publisher(String,  '/decision/alert',          10)
        self.latency_pub = self.create_publisher(Float32, '/decision/latency',        10)

        self.previous_regions = {}

        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        self.csv_file   = open(CSV_PATH, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'alert_type', 'region_id', 'detail',
            'region_pixels', 'closest_mm', 'center_x', 'center_y'
        ])
        self.get_logger().info('DecisionNode started.')

    def process_regions(self, mask, depth_array, alert_type_label, detail_prefix):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        regions = []
        for i in range(1, num_labels):
            region_mask   = (labels == i)
            region_pixels = int(np.sum(region_mask))
            if region_pixels < 5:
                continue
            depths        = depth_array[region_mask]
            valid_mask    = (depths >= DEPTH_TOO_CLOSE) & (depths <= DEPTH_MAX_VALID)
            far_mask      = depths > DEPTH_MAX_VALID
            tooclose_mask = depths < DEPTH_TOO_CLOSE
            cx = int(centroids[i][0])
            cy = int(centroids[i][1])
            if valid_mask.any():
                closest_mm = int(np.min(depths[valid_mask]))
                detail     = f'{detail_prefix} in valid range | closest={closest_mm}mm | center=({cx},{cy})'
                alert_type = alert_type_label
            elif far_mask.all():
                closest_mm = -1
                detail     = f'{detail_prefix} BUT FAR | monitoring'
                alert_type = 'WARNING'
            elif tooclose_mask.all():
                closest_mm = -1
                detail     = f'{detail_prefix} | TOO CLOSE | sensor range invalid'
                alert_type = 'WARNING'
            else:
                closest_mm = int(np.min(depths[valid_mask])) if valid_mask.any() else -1
                detail     = f'{detail_prefix} | MIXED RANGE | closest={closest_mm}mm | center=({cx},{cy})'
                alert_type = alert_type_label
            regions.append({
                'alert_type'   : alert_type,
                'detail'       : detail,
                'region_pixels': region_pixels,
                'closest_mm'   : closest_mm,
                'center_x'     : cx,
                'center_y'     : cy,
            })
        return regions

    def should_write(self, region_id, region):
        prev = self.previous_regions.get(region_id)
        if prev is None:
            return True
        if region['alert_type'] != prev['alert_type']:
            return True
        if abs(region['closest_mm'] - prev['closest_mm']) > 10:
            return True
        if (abs(region['center_x'] - prev['center_x']) > 5 or
                abs(region['center_y'] - prev['center_y']) > 5):
            return True
        return False

    def decision_callback(self, fused_msg: FusedData):

        # --- Latency: fusion → decision ---
        fusion_stamp_ns = (
            fused_msg.header.stamp.sec * 1_000_000_000 +
            fused_msg.header.stamp.nanosec)
        now_ns = self.get_clock().now().nanoseconds
        latency_ms = (now_ns - fusion_stamp_ns) / 1_000_000.0

        # --- 1. THERMAL ARRAY ---
        thermal_array = self.bridge.imgmsg_to_cv2(fused_msg.thermal, desired_encoding='32FC1')

        # --- 2. THERMAL BITMAP ---
        bitmap = np.zeros_like(thermal_array, dtype=np.uint8)
        bitmap[thermal_array > THERMAL_HOT]       = 1
        bitmap[thermal_array > THERMAL_DANGEROUS] = 2

        bitmap_color = np.zeros((bitmap.shape[0], bitmap.shape[1], 3), dtype=np.uint8)
        bitmap_color[bitmap == 0] = [0, 255, 0]
        bitmap_color[bitmap == 1] = [0, 165, 255]
        bitmap_color[bitmap == 2] = [0, 0,   255]

        bitmap_msg          = Image()
        bitmap_msg.header   = fused_msg.thermal.header
        bitmap_msg.height   = bitmap_color.shape[0]
        bitmap_msg.width    = bitmap_color.shape[1]
        bitmap_msg.encoding = 'bgr8'
        bitmap_msg.step     = bitmap_msg.width * 3
        bitmap_msg.data     = bitmap_color.tobytes()
        self.bitmap_pub.publish(bitmap_msg)

        # --- 5. HER 20 SANİYEDE BİR PNG KAYDETME ---
        current_time = self.get_clock().now()
        if (current_time - self.last_bitmap_time).nanoseconds > 20 * 1e9:
            scale_factor = 4
            bitmap_large = np.kron(bitmap_color, np.ones((scale_factor, scale_factor, 1), dtype=np.uint8))
            
            timestamp_str = datetime.now().strftime("%H%M%S")
            file_path = f'/workspace/logs/bitmap_{timestamp_str}.png'
            cv2.imwrite(file_path, bitmap_large)
            self.get_logger().info(f'Periodic analysis saved: {file_path}')
            self.last_bitmap_time = current_time

        # --- 4. DEPTH ARRAY ---
        depth_raw = self.bridge.imgmsg_to_cv2(fused_msg.depth, desired_encoding='16UC1')
        depth_array = cv2.resize(
            depth_raw,
            (bitmap.shape[1], bitmap.shape[0]),
            interpolation=cv2.INTER_NEAREST)

        # --- 5. CONNECTED COMPONENTS ---
        dangerous_regions = self.process_regions(
            bitmap == 2, depth_array, 'CRITICAL', 'DANGEROUS region')
        hot_regions = self.process_regions(
            bitmap == 1, depth_array, 'WARNING', 'HOT region')
        all_regions = dangerous_regions + hot_regions

        # --- 6. PUBLISH ALERT ---
        if not all_regions:
            alert_msg      = String()
            alert_msg.data = 'SAFE'
            self.alert_pub.publish(alert_msg)
            self.get_logger().info('SAFE')
            self.previous_regions = {}
        else:
            top_alert = 'CRITICAL' if any(
                r['alert_type'] == 'CRITICAL' for r in all_regions) else 'WARNING'
            alert_msg      = String()
            alert_msg.data = f'{top_alert} | {len(all_regions)} regions detected'
            self.alert_pub.publish(alert_msg)
            self.get_logger().info(alert_msg.data)
            
            if top_alert == 'CRITICAL':
                scale_factor = 4
                bitmap_large = np.kron(bitmap_color, np.ones((scale_factor, scale_factor, 1), dtype=np.uint8))
                timestamp_str = datetime.now().strftime("%H%M%S_CRITICAL")
                cv2.imwrite(f'/workspace/logs/bitmap_{timestamp_str}.png', bitmap_large)
                self.get_logger().info(f'ANLIK KANIT KAYDEDILDI: bitmap_{timestamp_str}.png')
            
            timestamp          = datetime.now().isoformat()
            current_region_ids = set()
            for region in all_regions:
                region_id = (f'{region["alert_type"]}_{region["center_x"]}'
                             f'_{region["center_y"]}_{region["region_pixels"]}')
                current_region_ids.add(region_id)
                if self.should_write(region_id, region):
                    self.csv_writer.writerow([
                        timestamp, region['alert_type'], region_id,
                        region['detail'], region['region_pixels'],
                        region['closest_mm'], region['center_x'], region['center_y']])
                    self.csv_file.flush()
                    self.previous_regions[region_id] = region
            self.previous_regions = {
                k: v for k, v in self.previous_regions.items()
                if k in current_region_ids}

        # --- 7. PUBLISH LATENCY ---
        latency_msg      = Float32()
        latency_msg.data = float(latency_ms)
        self.latency_pub.publish(latency_msg)
        self.get_logger().debug(f'Latency: {latency_ms:.2f}ms')

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
