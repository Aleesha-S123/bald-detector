import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("bald_or_not_model.h5")

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        try:
            # Resize to model input size (224x224)
            resized_face = cv2.resize(face_img, (224, 224))

            # Normalize and reshape
            normalized = resized_face.astype("float32") / 255.0
            reshaped = np.reshape(normalized, (1, 224, 224, 3))

            # Predict
            prediction = model.predict(reshaped, verbose=0)[0][0]
            label = "Bald" if prediction > 0.5 else "Not Bald"
            color = (0, 0, 255) if label == "Bald" else (0, 255, 0)

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({prediction:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print("Face processing error:", e)

    # Show the frame
    cv2.imshow("Bald Detector - Webcam", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()