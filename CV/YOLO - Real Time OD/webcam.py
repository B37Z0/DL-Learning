import cv2

# Check for webcam 
# 0: Integrated webcam
# >1: External webcams
cam_indices = []

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        cam_indices.append(i)
    cap.release()

print("Webcams found at indices:", cam_indices)