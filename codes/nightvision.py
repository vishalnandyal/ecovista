import cv2
from ultralytics import YOLO
import math

cap = cv2.VideoCapture('/home/sevengods/Downloads/1.mp4') # put 0 for webcam

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 1, (frame_width, frame_height))

model = YOLO("/home/sevengods/Documents/WMS&LD/Litter_detection/yolov8_model/runs/detect/train2/weights/best.pt")
classnames = ["Person", "Trash", "Litter", "person_littered"]
colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 0, 255)]  # Define colors for each class

if len(classnames) != len(colors):
    raise ValueError("Number of classes must match the number of colors")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Simulate night vision effect
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a darker intensity to the grayscale frame
    dark_frame = cv2.addWeighted(gray_frame, 0.5, gray_frame, 0, 50)
    
    # Convert the darker grayscale frame back to BGR
    processed_frame = cv2.cvtColor(dark_frame, cv2.COLOR_GRAY2BGR)

    # Detect objects
    results = model(processed_frame, stream=True)
    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)
            color = colors[i % len(colors)]  # Use modulus to cycle through colors if more boxes than colors
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 3)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classnames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(processed_frame, (x1, y1), c2, color, -1, cv2.LINE_AA)
            cv2.putText(processed_frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    out.write(processed_frame)
    cv2.imshow("Waste Detection", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

