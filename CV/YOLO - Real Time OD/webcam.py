import cv2

# Check for webcam 
# 0: Integrated webcam
# >1: External webcams
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        print(f"Camera found at index {i}")
    cap.release()
