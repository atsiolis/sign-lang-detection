import cv2
import mediapipe as mp

# Initialize MediaPipe Hands with lower accuracy for speed
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, 
                       min_tracking_confidence=0.2)

mp_drawing = mp.solutions.drawing_utils

# Open a connection to the camera (fallback to default camera if external not found)
capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not capture.isOpened():
    print("Warning: External camera not found. Switching to default camera.")
    capture = cv2.VideoCapture(0)

# Reduce resolution for better performance
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_counter = 0

def draw_landmarks_on_image(image, detection_result):
    #Draws hand landmarks if detected.
    if not detection_result.multi_hand_landmarks:
        return image  # Return original if no hands detected

    for hand_landmarks in detection_result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return image

while True:
    ret, frame = capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    if frame_counter % 1 != 0:  # Process every 3rd frame to reduce CPU usage
        cv2.imshow("Hand Landmarks", frame)
        continue
    
    # Convert to grayscale first, then back to RGB for MediaPipe (faster than direct RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    # Draw landmarks and display
    annotated_image = draw_landmarks_on_image(frame, results)
    cv2.imshow("Hand Landmarks", annotated_image)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
