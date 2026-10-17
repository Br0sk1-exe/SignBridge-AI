🤟 SignBridge: Real-Time ASL Alphabet Detection
A lightweight, real-time American Sign Language (ASL) alphabet recognition system powered by Deep Learning (MobileNetV2) and OpenCV. This application processes live webcam feeds through a optimized static Region of Interest (ROI) to classify 29 different ASL signs instantly with high confidence.

✨ Features
Transfer Learning Architecture: Built using a fine-tuned MobileNetV2 model trained on the Kaggle ASL Alphabet dataset.

Optimized Real-Time Inference: Utilizes a clean, stable static ROI bounding box approach to eliminate complex tracking overhead and background noise.

Interactive Controls:

Press f to toggle the camera mirror flip live.

Press q to safely exit the application.

High Confidence Output: Displays real-time prediction labels and confidence scores directly on the video stream.

🛠️ Tech Stack
Language: Python

Deep Learning: TensorFlow / Keras

Computer Vision: OpenCV (cv2)

Numerical Computing: NumPy

📁 Project Structure
Plaintext
SignBridge/
│
├── asl_model_3.keras       # Fine-tuned MobileNetV2 model weights
├── static_app.py           # Main real-time inference script
└── README.md               # Project documentation
🚀 Getting Started Locally
Follow these steps to set up and run the project on your machine.

1. Prerequisites
Make sure you have Python 3.8+ installed along with pip.

2. Clone the Repository
Bash
git clone https://github.com/YOUR_USERNAME/SignBridge.git
cd SignBridge
3. Install Dependencies
Install the required Python packages using pip:

Bash
pip install tensorflow opencv-python numpy
4. Run the Application
Ensure your fine-tuned model file (asl_model_3.keras) is in the root directory, then run the inference script:

Bash
python static_app.py
🕹️ Usage Guide
Launch the script. A window titled "Static Box ASL Detection" will open with a green bounding box.

Position your hand inside the green box.

Form any ASL alphabet sign (A–Z, space, del, or nothing).

Watch the real-time prediction and confidence percentage update at the top of the screen!

Hotkeys:

f: Flip camera mirror view.

q: Quit the application.

🧠 Model Training Details
Dataset: Kaggle ASL Alphabet Dataset (70,000+ images across 29 classes).

Base Model: MobileNetV2 (pretrained on ImageNet).

Approach: Feature extraction followed by fine-tuning custom classification layers to adapt the model to localized hand sign geometry.

📄 License
This project is open-source and available under the MIT License.
