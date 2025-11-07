import cv2
import numpy as np
import onnxruntime as ort
from time import time
import json
import logging
from datetime import datetime
import threading
import queue

# Watchdog and RTC
try:
    from gpiozero import LED
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: gpiozero not available. Watchdog will use software fallback.")

# RTC
try:
    import board
    import adafruit_ds3231
    RTC_AVAILABLE = True
except ImportError:
    RTC_AVAILABLE = False
    print("Warning: adafruit_ds3231 not available. Using system time.")

# AWS IoT Core
try:
    from awscrt import mqtt
    from awsiot import mqtt_connection_builder
    AWS_IOT_AVAILABLE = True
except ImportError:
    AWS_IOT_AVAILABLE = False
    print("Warning: AWS IoT SDK not available. Install with: pip install awsiotsdk")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('microplastic_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Watchdog:
    """Software and hardware watchdog implementation"""
    def __init__(self, timeout=30, gpio_pin=None):
        self.timeout = timeout
        self.last_reset = time()
        self.running = False
        self.thread = None
        
        # Hardware watchdog via GPIO (optional)
        self.hw_watchdog = None
        if gpio_pin and WATCHDOG_AVAILABLE:
            try:
                self.hw_watchdog = LED(gpio_pin)
                logger.info(f"Hardware watchdog initialized on GPIO {gpio_pin}")
            except Exception as e:
                logger.warning(f"Could not initialize hardware watchdog: {e}")
    
    def start(self):
        """Start watchdog monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        logger.info("Watchdog started")
    
    def stop(self):
        """Stop watchdog monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Watchdog stopped")
    
    def reset(self):
        """Reset watchdog timer"""
        self.last_reset = time()
        if self.hw_watchdog:
            self.hw_watchdog.toggle()
    
    def _monitor(self):
        """Monitor thread to check for timeouts"""
        while self.running:
            elapsed = time() - self.last_reset
            if elapsed > self.timeout:
                logger.critical("Watchdog timeout! System may be frozen.")
                # In production, this could trigger a system reboot
                # os.system('sudo reboot')
            threading.Event().wait(1)


class RTCManager:
    """Real-Time Clock manager"""
    def __init__(self):
        self.rtc = None
        if RTC_AVAILABLE:
            try:
                i2c = board.I2C()
                self.rtc = adafruit_ds3231.DS3231(i2c)
                logger.info("RTC initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize RTC: {e}")
    
    def get_time(self):
        """Get current time from RTC or system"""
        if self.rtc:
            try:
                t = self.rtc.datetime
                return datetime(t.tm_year, t.tm_mon, t.tm_mday, 
                              t.tm_hour, t.tm_min, t.tm_sec)
            except Exception as e:
                logger.warning(f"Error reading RTC: {e}")
        return datetime.now()
    
    def sync_system_time(self):
        """Sync system time with RTC"""
        if self.rtc:
            try:
                rtc_time = self.get_time()
                # This requires root privileges
                # os.system(f'sudo date -s "{rtc_time.strftime("%Y-%m-%d %H:%M:%S")}"')
                logger.info(f"System time synced with RTC: {rtc_time}")
                return True
            except Exception as e:
                logger.error(f"Could not sync system time: {e}")
        return False


class IoTConnector:
    """AWS IoT Core connector"""
    def __init__(self, config):
        self.config = config
        self.connection = None
        self.connected = False
        self.message_queue = queue.Queue(maxsize=100)
        self.publish_thread = None
        
        if AWS_IOT_AVAILABLE:
            self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize AWS IoT connection"""
        try:
            self.connection = mqtt_connection_builder.mtls_from_path(
                endpoint=self.config['endpoint'],
                cert_filepath=self.config['cert_path'],
                pri_key_filepath=self.config['key_path'],
                ca_filepath=self.config['ca_path'],
                client_id=self.config['client_id'],
                clean_session=False,
                keep_alive_secs=30
            )
            
            # Connect
            connect_future = self.connection.connect()
            connect_future.result()
            self.connected = True
            logger.info("Connected to AWS IoT Core")
            
            # Start publish thread
            self.publish_thread = threading.Thread(target=self._publish_worker, daemon=True)
            self.publish_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to connect to AWS IoT: {e}")
            self.connected = False
    
    def _publish_worker(self):
        """Background thread to publish messages"""
        while True:
            try:
                topic, message = self.message_queue.get(timeout=1)
                if self.connected:
                    self.connection.publish(
                        topic=topic,
                        payload=json.dumps(message),
                        qos=mqtt.QoS.AT_LEAST_ONCE
                    )
                    logger.debug(f"Published to {topic}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error publishing message: {e}")
    
    def publish(self, topic, message):
        """Queue message for publishing"""
        try:
            self.message_queue.put_nowait((topic, message))
        except queue.Full:
            logger.warning("Message queue full, dropping message")
    
    def disconnect(self):
        """Disconnect from AWS IoT"""
        if self.connection and self.connected:
            disconnect_future = self.connection.disconnect()
            disconnect_future.result()
            self.connected = False
            logger.info("Disconnected from AWS IoT Core")


class MicroplasticDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Class names
        self.classes = ['PS', 'PS_with_dust', 'PHA', 'PHA_with_dust', 'mixed', 'none']
        
        # Colors for visualization (BGR)
        self.colors = [
            (255, 0, 0),    # PS - Blue
            (255, 128, 0),  # PS_with_dust - Light Blue
            (0, 255, 0),    # PHA - Green
            (0, 255, 128),  # PHA_with_dust - Light Green
            (0, 165, 255),  # mixed - Orange
            (0, 0, 255),    # none - Red
        ]
    
    def preprocess(self, image):
        # Resize and normalize
        img_height, img_width = image.shape[:2]
        input_height, input_width = self.input_shape[2], self.input_shape[3]
        
        # Resize image
        img_resized = cv2.resize(image, (input_width, input_height))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, img_height, img_width
    
    def postprocess(self, outputs, orig_height, orig_width):
        # Extract predictions
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Filter by confidence
        scores = np.max(predictions[4:], axis=0)
        class_ids = np.argmax(predictions[4:], axis=0)
        mask = scores > self.conf_threshold
        
        boxes = predictions[:4, mask].T
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # Convert boxes from xywh to xyxy
        boxes[:, 0] -= boxes[:, 2] / 2  # x1
        boxes[:, 1] -= boxes[:, 3] / 2  # y1
        boxes[:, 2] += boxes[:, 0]  # x2
        boxes[:, 3] += boxes[:, 1]  # y2
        
        # Scale boxes to original image size
        scale_x = orig_width / self.input_shape[3]
        scale_y = orig_height / self.input_shape[2]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        # Apply NMS
        indices = self.nms(boxes, scores, self.iou_threshold)
        
        return boxes[indices], scores[indices], class_ids[indices]
    
    def nms(self, boxes, scores, iou_threshold):
        # Simple NMS implementation
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            current_box = boxes[current]
            rest_boxes = boxes[indices[1:]]
            
            ious = self.compute_iou(current_box, rest_boxes)
            indices = indices[1:][ious < iou_threshold]
        
        return np.array(keep)
    
    def compute_iou(self, box, boxes):
        # Compute IoU between one box and multiple boxes
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        return intersection / union
    
    def detect(self, image):
        # Preprocess
        input_data, orig_h, orig_w = self.preprocess(image)
        
        # Inference
        start_time = time()
        outputs = self.session.run(None, {self.input_name: input_data})
        inference_time = time() - start_time
        
        # Postprocess
        boxes, scores, class_ids = self.postprocess(outputs, orig_h, orig_w)
        
        return boxes, scores, class_ids, inference_time
    
    def draw_detections(self, image, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            color = self.colors[class_id]
            class_name = self.classes[class_id]
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{class_name}: {score:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image


class MicroplasticIoTSystem:
    """Integrated system with detection, watchdog, RTC, and IoT"""
    def __init__(self, model_path, iot_config, device_id):
        self.detector = MicroplasticDetector(model_path)
        self.watchdog = Watchdog(timeout=30, gpio_pin=None)  # Set GPIO pin if using hardware
        self.rtc = RTCManager()
        self.iot = IoTConnector(iot_config) if iot_config else None
        self.device_id = device_id
        
        # Statistics
        self.detection_count = 0
        self.session_start = time()
        
        logger.info("Microplastic IoT System initialized")
    
    def start(self):
        """Start the system"""
        self.watchdog.start()
        if self.rtc:
            self.rtc.sync_system_time()
        logger.info("System started")
    
    def stop(self):
        """Stop the system"""
        self.watchdog.stop()
        if self.iot:
            self.iot.disconnect()
        logger.info("System stopped")
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.watchdog.reset()  # Reset watchdog on each frame
        
        # Detect microplastics
        boxes, scores, class_ids, inference_time = self.detector.detect(frame)
        
        # Update statistics
        self.detection_count += len(boxes)
        
        # Prepare IoT message
        if self.iot and len(boxes) > 0:
            detections = []
            for box, score, class_id in zip(boxes, scores, class_ids):
                detections.append({
                    'class': self.detector.classes[class_id],
                    'confidence': float(score),
                    'bbox': box.tolist()
                })
            
            message = {
                'device_id': self.device_id,
                'timestamp': self.rtc.get_time().isoformat(),
                'detection_count': len(boxes),
                'detections': detections,
                'inference_time_ms': inference_time * 1000,
                'session_detections': self.detection_count
            }
            
            # Publish to IoT platform
            self.iot.publish(f'microplastic/{self.device_id}/detections', message)
        
        # Send periodic heartbeat
        if int(time() - self.session_start) % 60 == 0:  # Every minute
            heartbeat = {
                'device_id': self.device_id,
                'timestamp': self.rtc.get_time().isoformat(),
                'status': 'online',
                'uptime_seconds': int(time() - self.session_start),
                'total_detections': self.detection_count
            }
            if self.iot:
                self.iot.publish(f'microplastic/{self.device_id}/heartbeat', heartbeat)
        
        # Draw detections
        result_frame = self.detector.draw_detections(frame.copy(), boxes, scores, class_ids)
        
        return result_frame, boxes, scores, class_ids, inference_time


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # IoT Configuration (AWS IoT Core)
    iot_config = {
        'endpoint': 'your-endpoint.iot.us-east-1.amazonaws.com',
        'cert_path': '/path/to/certificate.pem.crt',
        'key_path': '/path/to/private.pem.key',
        'ca_path': '/path/to/AmazonRootCA1.pem',
        'client_id': 'microplastic-detector-001'
    }
    
    # Initialize system
    system = MicroplasticIoTSystem(
        model_path='best.onnx',
        iot_config=iot_config,
        device_id='RPI-MP-001'
    )
    
    system.start()
    
    try:
        # Example 1: Image detection
        image = cv2.imread('test_image.jpg')
        result_frame, boxes, scores, class_ids, inference_time = system.process_frame(image)
        
        print(f"Inference time: {inference_time*1000:.2f}ms")
        print(f"Detections: {len(boxes)}")
        
        cv2.imwrite('result.jpg', result_frame)
        
        # Example 2: Real-time video detection
        cap = cv2.VideoCapture(0)  # USB camera or Pi Camera
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result_frame, boxes, scores, class_ids, inference_time = system.process_frame(frame)
            
            # Display FPS and info
            fps = 1 / inference_time if inference_time > 0 else 0
            cv2.putText(result_frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f'Detections: {len(boxes)}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f'Total: {system.detection_count}', (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Microplastic Detection', result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    finally:
        system.stop()
