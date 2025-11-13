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

# Raspberry Pi Camera (libcamera)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. CSI camera support disabled.")

# AWS IoT Core
try:
    from awscrt import mqtt
    from awsiot import mqtt_connection_builder
    AWS_IOT_AVAILABLE = True
except ImportError:
    AWS_IOT_AVAILABLE = False
    print("Warning: AWS IoT SDK not available.")

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


class CMOSCameraController:
    """
    CMOS Camera Controller with support for:
    - Raspberry Pi Camera Module (CSI interface)
    - USB Cameras (UVC compliant)
    - RTSP Network Cameras
    """
    
    def __init__(self, camera_config):
        self.config = camera_config
        self.camera = None
        self.camera_type = camera_config.get('type', 'usb')
        self.width = camera_config.get('width', 1920)
        self.height = camera_config.get('height', 1080)
        self.fps = camera_config.get('fps', 30)
        self.is_streaming = False
        
        # Camera settings
        self.exposure_mode = camera_config.get('exposure_mode', 'auto')
        self.exposure_time = camera_config.get('exposure_time', None)
        self.gain = camera_config.get('gain', 1.0)
        self.white_balance = camera_config.get('white_balance', 'auto')
        self.brightness = camera_config.get('brightness', 0)
        self.contrast = camera_config.get('contrast', 1.0)
        self.saturation = camera_config.get('saturation', 1.0)
        self.sharpness = camera_config.get('sharpness', 1.0)
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera based on type"""
        try:
            if self.camera_type == 'csi' and PICAMERA2_AVAILABLE:
                self._init_csi_camera()
            elif self.camera_type == 'usb':
                self._init_usb_camera()
            elif self.camera_type == 'rtsp':
                self._init_rtsp_camera()
            else:
                raise ValueError(f"Unsupported camera type: {self.camera_type}")
            
            logger.info(f"Camera initialized: {self.camera_type} at {self.width}x{self.height}@{self.fps}fps")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise
    
    def _init_csi_camera(self):
        """Initialize Raspberry Pi CSI Camera (CMOS sensor)"""
        self.camera = Picamera2()
        
        # Configure camera
        config = self.camera.create_preview_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888"
            },
            controls={
                "FrameRate": self.fps
            }
        )
        
        self.camera.configure(config)
        
        # Set camera controls
        controls = {}
        
        if self.exposure_mode == 'manual' and self.exposure_time:
            controls["ExposureTime"] = int(self.exposure_time)
            controls["AnalogueGain"] = self.gain
        
        if self.white_balance == 'manual':
            controls["AwbEnable"] = False
            controls["ColourGains"] = (1.5, 1.5)  # Red, Blue gains
        
        controls["Brightness"] = self.brightness
        controls["Contrast"] = self.contrast
        controls["Saturation"] = self.saturation
        controls["Sharpness"] = self.sharpness
        
        if controls:
            self.camera.set_controls(controls)
        
        self.camera.start()
        logger.info("CSI camera (CMOS) initialized with libcamera")
    
    def _init_usb_camera(self):
        """Initialize USB Camera (UVC CMOS sensor)"""
        camera_index = self.config.get('device_index', 0)
        self.camera = cv2.VideoCapture(camera_index)
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Set exposure if supported
        if self.exposure_mode == 'manual':
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
            if self.exposure_time:
                self.camera.set(cv2.CAP_PROP_EXPOSURE, self.exposure_time)
        else:
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto mode
        
        # Set other properties
        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
        self.camera.set(cv2.CAP_PROP_CONTRAST, self.contrast)
        self.camera.set(cv2.CAP_PROP_SATURATION, self.saturation)
        self.camera.set(cv2.CAP_PROP_GAIN, self.gain)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open USB camera at index {camera_index}")
        
        logger.info("USB camera (CMOS) initialized")
    
    def _init_rtsp_camera(self):
        """Initialize RTSP network camera"""
        rtsp_url = self.config.get('rtsp_url')
        if not rtsp_url:
            raise ValueError("RTSP URL not provided")
        
        self.camera = cv2.VideoCapture(rtsp_url)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to connect to RTSP camera: {rtsp_url}")
        
        logger.info(f"RTSP camera initialized: {rtsp_url}")
    
    def capture_frame(self):
        """Capture a single frame from the camera"""
        if not self.camera:
            raise RuntimeError("Camera not initialized")
        
        try:
            if self.camera_type == 'csi' and PICAMERA2_AVAILABLE:
                frame = self.camera.capture_array()
                return True, frame
            else:
                ret, frame = self.camera.read()
                return ret, frame
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return False, None
    
    def start_stream(self):
        """Start camera stream"""
        self.is_streaming = True
        logger.info("Camera stream started")
    
    def stop_stream(self):
        """Stop camera stream"""
        self.is_streaming = False
        logger.info("Camera stream stopped")
    
    def get_camera_info(self):
        """Get camera sensor information"""
        info = {
            'type': self.camera_type,
            'resolution': f"{self.width}x{self.height}",
            'fps': self.fps,
            'sensor_type': 'CMOS',
            'exposure_mode': self.exposure_mode,
            'white_balance': self.white_balance
        }
        
        if self.camera_type == 'usb' and self.camera:
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            info['actual_resolution'] = f"{int(actual_width)}x{int(actual_height)}"
            info['actual_fps'] = actual_fps
        
        return info
    
    def set_exposure(self, mode='auto', exposure_time=None, gain=1.0):
        """Set camera exposure settings"""
        self.exposure_mode = mode
        self.exposure_time = exposure_time
        self.gain = gain
        
        if self.camera_type == 'csi' and PICAMERA2_AVAILABLE:
            controls = {}
            if mode == 'manual' and exposure_time:
                controls["ExposureTime"] = int(exposure_time)
                controls["AnalogueGain"] = gain
            else:
                controls["AeEnable"] = True
            self.camera.set_controls(controls)
        elif self.camera_type == 'usb':
            if mode == 'manual':
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                if exposure_time:
                    self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure_time)
                self.camera.set(cv2.CAP_PROP_GAIN, gain)
            else:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        
        logger.info(f"Exposure set to {mode}")
    
    def set_white_balance(self, mode='auto', red_gain=1.0, blue_gain=1.0):
        """Set white balance settings"""
        self.white_balance = mode
        
        if self.camera_type == 'csi' and PICAMERA2_AVAILABLE:
            if mode == 'manual':
                self.camera.set_controls({
                    "AwbEnable": False,
                    "ColourGains": (red_gain, blue_gain)
                })
            else:
                self.camera.set_controls({"AwbEnable": True})
        
        logger.info(f"White balance set to {mode}")
    
    def adjust_brightness(self, value):
        """Adjust camera brightness (-1.0 to 1.0)"""
        self.brightness = value
        
        if self.camera_type == 'csi' and PICAMERA2_AVAILABLE:
            self.camera.set_controls({"Brightness": value})
        elif self.camera_type == 'usb':
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, value)
        
        logger.info(f"Brightness adjusted to {value}")
    
    def adjust_contrast(self, value):
        """Adjust camera contrast (0.0 to 2.0)"""
        self.contrast = value
        
        if self.camera_type == 'csi' and PICAMERA2_AVAILABLE:
            self.camera.set_controls({"Contrast": value})
        elif self.camera_type == 'usb':
            self.camera.set(cv2.CAP_PROP_CONTRAST, value)
        
        logger.info(f"Contrast adjusted to {value}")
    
    def release(self):
        """Release camera resources"""
        if self.camera:
            if self.camera_type == 'csi' and PICAMERA2_AVAILABLE:
                self.camera.stop()
                self.camera.close()
            else:
                self.camera.release()
            logger.info("Camera released")


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
            
            connect_future = self.connection.connect()
            connect_future.result()
            self.connected = True
            logger.info("Connected to AWS IoT Core")
            
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
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.classes = ['PS', 'PS_with_dust', 'PHA', 'PHA_with_dust', 'mixed', 'none']
        
        self.colors = [
            (255, 0, 0), (255, 128, 0), (0, 255, 0),
            (0, 255, 128), (0, 165, 255), (0, 0, 255)
        ]
    
    def preprocess(self, image):
        img_height, img_width = image.shape[:2]
        input_height, input_width = self.input_shape[2], self.input_shape[3]
        
        img_resized = cv2.resize(image, (input_width, input_height))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, img_height, img_width
    
    def postprocess(self, outputs, orig_height, orig_width):
        predictions = outputs[0][0]
        
        scores = np.max(predictions[4:], axis=0)
        class_ids = np.argmax(predictions[4:], axis=0)
        mask = scores > self.conf_threshold
        
        boxes = predictions[:4, mask].T
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        
        scale_x = orig_width / self.input_shape[3]
        scale_y = orig_height / self.input_shape[2]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        indices = self.nms(boxes, scores, self.iou_threshold)
        
        return boxes[indices], scores[indices], class_ids[indices]
    
    def nms(self, boxes, scores, iou_threshold):
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
        input_data, orig_h, orig_w = self.preprocess(image)
        
        start_time = time()
        outputs = self.session.run(None, {self.input_name: input_data})
        inference_time = time() - start_time
        
        boxes, scores, class_ids = self.postprocess(outputs, orig_h, orig_w)
        
        return boxes, scores, class_ids, inference_time
    
    def draw_detections(self, image, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            color = self.colors[class_id]
            class_name = self.classes[class_id]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f'{class_name}: {score:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image


class MicroplasticIoTSystem:
    """Integrated system with CMOS camera, detection, watchdog, RTC, and IoT"""
    def __init__(self, model_path, camera_config, iot_config, device_id):
        self.camera = CMOSCameraController(camera_config)
        self.detector = MicroplasticDetector(model_path)
        self.watchdog = Watchdog(timeout=30, gpio_pin=None)
        self.rtc = RTCManager()
        self.iot = IoTConnector(iot_config) if iot_config else None
        self.device_id = device_id
        
        self.detection_count = 0
        self.session_start = time()
        self.last_heartbeat = time()
        
        logger.info("Microplastic IoT System with CMOS camera initialized")
        logger.info(f"Camera info: {self.camera.get_camera_info()}")
    
    def start(self):
        """Start the system"""
        self.watchdog.start()
        self.camera.start_stream()
        if self.rtc:
            self.rtc.sync_system_time()
        logger.info("System started")
    
    def stop(self):
        """Stop the system"""
        self.watchdog.stop()
        self.camera.stop_stream()
        self.camera.release()
        if self.iot:
            self.iot.disconnect()
        logger.info("System stopped")
    
    def process_frame(self, frame=None):
        """Process a frame (capture from camera if not provided)"""
        self.watchdog.reset()
        
        # Capture frame from camera if not provided
        if frame is None:
            ret, frame = self.camera.capture_frame()
            if not ret or frame is None:
                logger.warning("Failed to capture frame from camera")
                return None, [], [], [], 0
        
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
                'session_detections': self.detection_count,
                'camera_info': self.camera.get_camera_info()
            }
            
            self.iot.publish(f'microplastic/{self.device_id}/detections', message)
        
        # Send periodic heartbeat (every 60 seconds)
        if time() - self.last_heartbeat >= 60:
            heartbeat = {
                'device_id': self.device_id,
                'timestamp': self.rtc.get_time().isoformat(),
                'status': 'online',
                'uptime_seconds': int(time() - self.session_start),
                'total_detections': self.detection_count,
                'camera_info': self.camera.get_camera_info()
            }
            if self.iot:
                self.iot.publish(f'microplastic/{self.device_id}/heartbeat', heartbeat)
            self.last_heartbeat = time()
        
        # Draw detections
        result_frame = self.detector.draw_detections(frame.copy(), boxes, scores, class_ids)
        
        return result_frame, boxes, scores, class_ids, inference_time


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Camera Configuration
    camera_config = {
        'type': 'csi',  # Options: 'csi', 'usb', 'rtsp'
        'width': 1920,
        'height': 1080,
        'fps': 30,
        'exposure_mode': 'auto',  # 'auto' or 'manual'
        'exposure_time': None,  # microseconds (for manual mode)
        'gain': 1.0,
        'white_balance': 'auto',  # 'auto' or 'manual'
        'brightness': 0,
        'contrast': 1.0,
        'saturation': 1.0,
        'sharpness': 1.0,
        # For USB camera
        'device_index': 0,
        # For RTSP camera
        'rtsp_url': 'rtsp://192.168.1.100:554/stream'
    }
    
    # IoT Configuration
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
        camera_config=camera_config,
        iot_config=iot_config,
        device_id='RPI-MP-CMOS-001'
    )
    
    system.start()
    
    try:
        # Real-time detection from CMOS camera
        frame_count = 0
        
        while True:
            # Process frame from camera
            result_frame, boxes, scores, class_ids, inference_time = system.process_frame()
            
            if result_frame is None:
                continue
            
            # Calculate FPS
            fps = 1 / inference_time if inference_time > 0 else 0
            
            # Display information
            cv2.putText(result_frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f'Detections: {len(boxes)}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f'Total: {system.detection_count}', (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f'Camera: CMOS {camera_config["type"].upper()}', (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Microplastic Detection - CMOS Camera', result_frame)
            
            frame_count += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                # Toggle exposure mode
                if system.camera.exposure_mode == 'auto':
                    system.camera.set_exposure('manual', exposure_time=10000, gain=2.0)
                else:
                    system.camera.set_exposure('auto')
            elif key == ord('b'):
                # Adjust brightness
                new_brightness = (system.camera.brightness + 0.1) % 2.0 -
