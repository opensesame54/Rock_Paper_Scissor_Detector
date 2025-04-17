# Rock Paper Scissors Detector
## A real-time Rock-Paper-Scissors game using YOLOv8 for hand gesture detection via webcam.

Built a computer vision model that lets you play Rock-Paper-Scissors using just your hand and a webcam.

### Steps:

1. **Dataset Collection:** Used a custom Rock-Paper-Scissors hand gesture dataset from Roboflow Universe.
2. **Model Training:** Trained a YOLOv8 object detection model on Google Colab using a T4 GPU for 50 epochs. Achieved ~96% precision.
3. **Deployment:** Downloaded the trained model and implemented real-time gesture detection using OpenCV and my local webcam.
