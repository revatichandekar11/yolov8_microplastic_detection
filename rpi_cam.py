import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# Initialize webcam, 0 is the default for the first connected camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Load the YOLO model
model = YOLO('best.pt')

# Load the class names from coco.txt file
with open("coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

count = 0
while True:
    # Capture frame-by-frame
    ret, im = cap.read()
   
    if not ret:
        print("Failed to grab frame")
        break

    # Create a grayscale version for display
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for compatibility with colored overlays

    count += 1
    if count % 3 != 0:
        continue

    # Run YOLO model on the original color frame
    results = model.predict(im)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Draw bounding boxes and labels on the color image
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        # Safe access to class list
        if 0 <= d < len(class_list):
            c = class_list[d]
        else:
            c = "Unknown"

        # Draw bounding box and label on the original color frame
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Colored bounding box
        cvzone.putTextRect(im, f'{c}', (x1, y1), 1, 1, colorT=(255, 255, 255))  # Colored label
   
    # Combine grayscale image with the color overlays
    combined_image = cv2.addWeighted(im_gray, 0.7, im, 0.3, 0)  # 0.7 for grayscale background, 0.3 for colored overlay

    # Display the combined image
    cv2.imshow("Camera", combined_image)
   
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()