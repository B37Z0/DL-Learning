from ultralytics import YOLO
import cv2

model = YOLO("../yolo_weights/yolo26s.pt")

# Run on webcam
cap = cv2.VideoCapture(1) # 0 for integrated / 1 for external
cap.set(3, 640)
cap.set(4, 480)


## Plotting with YOLO ##
while cap.isOpened():
    # Ignore success var since this is live video
    _, frame = cap.read()

    # YOLO inference
    results = model(frame, verbose=False)

    # Display inference
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Inference', annotated_frame)

    # Break if `q` pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


## Plotting with cv2 ##
# # COCO class names
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]

# while True:
#     success, img = cap.read()
#     results = model(img, stream=True, verbose=False)

#     for r in results:
#         boxes = r.boxes

#         for box in boxes:
#             # box coords
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

#             # classification
#             cls = int(box.cls[0])
#             confidence = box.conf[0]
#             label = f"{classNames[cls]}: {confidence*100:.1f}%"

#             # object details
#             org = [x1, y1]
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             fontScale = 0.75
#             color = (0, 0, 255)
#             thickness = 1

#             cv2.putText(img, label, org, font, fontScale, color, thickness)

#     cv2.imshow('Webcam', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()