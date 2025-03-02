import cv2

# Open a connection to the external camera (1)
capture = cv2.VideoCapture(1)  # Change 1 to 0 if you're using the default camera

# Check if the camera opened successfully
if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Display the frame
    cv2.imshow("Live Camera Feed", frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()
