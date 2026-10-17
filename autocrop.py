import os
import cv2
import numpy as np
import tensorflow as tf

# 1. Load model safely
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "asl_model3.keras")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

class_names = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "del",
    "nothing",
    "space",
]

# 2. Initialize Webcam
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
print(
    "Webcam started! Press 'f' to toggle camera mirror flip, and 'q' to quit."
)

# Strict static box coordinates
x1, y1, x2, y2 = 220, 140, 420, 340

# Flip state tracker (True = mirrored, False = normal)
is_flipped = True

while True:
  ret, frame = cap.read()
  if not ret:
    break

  # Apply horizontal flip based on the toggle state
  if is_flipped:
    frame = cv2.flip(frame, 1)

  # Draw static green box
  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
  cv2.putText(
      frame,
      "Keep Hand Inside Box (Press 'f' to flip)",
      (x1, y1 - 10),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0, 255, 0),
      2,
  )

  # Crop static region
  roi = frame[y1:y2, x1:x2]

  # Preprocess for MobileNetV2
  roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
  img = cv2.resize(roi_rgb, (128, 128))
  img = np.expand_dims(img, axis=0)

  # Predict
  predictions = model.predict(img, verbose=0)[0]
  best_idx = np.argmax(predictions)
  confidence = predictions[best_idx] * 100
  predicted_class = class_names[best_idx]

  if confidence >= 70.0:
    display_text = f"Sign: {predicted_class} ({confidence:.1f}%)"
    text_color = (0, 255, 0)
  else:
    display_text = "Show Hand Here..."
    text_color = (0, 0, 255)

  # Render text
  cv2.putText(
      frame,
      display_text,
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      text_color,
      2,
      cv2.LINE_AA,
  )
  cv2.imshow("Static Box ASL Detection", frame)

  # Key controls: Press 'q' to quit, Press 'f' to flip camera view
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break
  elif key == ord("f"):
    is_flipped = not is_flipped
    print(f"Camera mirror flip toggled -> Mirrored: {is_flipped}")

cap.release()
cv2.destroyAllWindows()