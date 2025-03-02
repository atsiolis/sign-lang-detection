import cv2

# Try different camera indices and backends
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use 0, 1, or try CAP_V4L2 on Linux

if not cap.isOpened():
    print("Error: External camera not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    cv2.imshow("Live Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
