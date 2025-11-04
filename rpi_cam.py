import cv2
import numpy as np
import onnxruntime as ort
from time import time

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

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Initialize detector
    detector = MicroplasticDetector(
        model_path='best.onnx',
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Example 1: Image detection
    image = cv2.imread('test_image.jpg')
    boxes, scores, class_ids, inference_time = detector.detect(image)
    
    print(f"Inference time: {inference_time*1000:.2f}ms")
    print(f"Detections: {len(boxes)}")
    
    # Draw and save results
    result_image = detector.draw_detections(image.copy(), boxes, scores, class_ids)
    cv2.imwrite('result.jpg', result_image)
    
    # Example 2: Webcam/Pi Camera detection
    cap = cv2.VideoCapture(0)  # Use 0 for USB camera or adjust for Pi Camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        boxes, scores, class_ids, inference_time = detector.detect(frame)
        
        # Draw results
        frame = detector.draw_detections(frame, boxes, scores, class_ids)
        
        # Display FPS
        fps = 1 / inference_time if inference_time > 0 else 0
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Microplastic Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
