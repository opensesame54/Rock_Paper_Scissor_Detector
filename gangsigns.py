import cv2
from ultralytics import YOLO
import argparse

# Load the trained YOLOv8 model
model = YOLO("Detector.pt")  # Change to your actual model path

# Rock-Paper-Scissors rules
WIN_RULES = {
    "rock": "scissors",
    "scissors": "paper",
    "paper": "rock"
}

# Initialize scores
score_zone1 = 0
score_zone2 = 0

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 RPS with zones & score")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int,
        help="Webcam resolution: width height"
    )
    return parser.parse_args()

args = parse_arguments()
width, height = args.webcam_resolution

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Frame cooldown to avoid score incrementing every frame
cooldown_frames = 60
cooldown_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, conf=0.5)
    r = results[0]
    annotated_frame = results[0].plot()

    # Draw zones
    mid_x = width // 2
    cv2.line(annotated_frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)
    cv2.putText(annotated_frame, "Zone 1", (mid_x // 2 - 50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, "Zone 2", (mid_x + mid_x // 2 - 50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Detection logic
    zone_1 = None
    zone_2 = None

    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2

        # Draw bounding boxes and labels
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        if cx < mid_x:
            zone_1 = cls_name.lower()
        else:
            zone_2 = cls_name.lower()

    # Determine winner if both zones have a prediction and cooldown allows
    if zone_1 and zone_2 and cooldown_counter == 0:
        if zone_1 == zone_2:
            result_text = "Draw"
        elif WIN_RULES[zone_1] == zone_2:
            result_text = "Zone 1 Wins!"
            score_zone1 += 1
        else:
            result_text = "Zone 2 Wins!"
            score_zone2 += 1

        # Display result
        cv2.putText(annotated_frame, result_text, (width // 2 - 200, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cooldown_counter = cooldown_frames

    # Cooldown logic
    if cooldown_counter > 0:
        cooldown_counter -= 1

    # Display scores
    cv2.putText(annotated_frame, f"Score - Zone 1: {score_zone1}", (30, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Score - Zone 2: {score_zone2}", (width - 350, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the final annotated frame
    cv2.imshow("YOLOv8 Rock Paper Scissors", annotated_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
