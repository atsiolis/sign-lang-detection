import cv2

def list_connected_cameras(max_cameras=10):
    available_cameras = []
    
    for i in range(max_cameras):
        capture = cv2.VideoCapture(i)
        if capture.isOpened():
            available_cameras.append(i)
            capture.release()

    return available_cameras

cameras = list_connected_cameras()
if cameras:
    print(f"Available cameras: {cameras}")
else:
    print("No cameras found.")
