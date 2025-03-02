import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the pre-trained ASL model
model = tf.keras.models.load_model("./model/trained_model.h5")  # Replace with your model path

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Open Camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# Define ASL labels (modify based on your model)
asl_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Delete", "Space"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 21 landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Flatten (x, y, z)

            # Convert landmarks to NumPy and reshape for the model
            landmarks_array = np.array(landmarks).reshape(1, -1)  # Shape (1, 63)

            # Predict ASL letter
            prediction = model.predict(landmarks_array)
            predicted_label = asl_labels[np.argmax(prediction)]

            # Display prediction
            cv2.putText(frame, predicted_label, (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Draw Hand Landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
