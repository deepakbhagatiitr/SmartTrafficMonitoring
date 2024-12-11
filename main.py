import cv2
from ultralytics import YOLO
import torch
import os

# Ensure correct device usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLO model
model = YOLO("yolov8n.pt").to(device)  # Load model to CUDA/CPU

# Define input and output directories
input_dir = "images"        # Folder containing input images
output_dir = "detections"   # Folder to save processed images
os.makedirs(output_dir, exist_ok=True)

# Helper function: Calculate IoU (Intersection Over Union)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Process all images in the input directory
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)

    # Read the image
    frame = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if frame is None:
        print(f"Error loading {image_name}. Skipping...")
        continue

    print(f"Processing {image_name}...")

    # Run YOLO detection
    results = model.predict(source=frame, save=False, conf=0.5, show=False)

    two_wheelers = []
    persons = []

    # Collect motorcycles, bicycles, and persons
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)  # Class ID
            label = model.names[class_id]  # Class label

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score

            if label in ["motorcycle", "bicycle"]:
                two_wheelers.append((label, (x1, y1, x2, y2, conf)))

            if label == "person":
                persons.append((x1, y1, x2, y2))

    # Match persons with two-wheelers
    for tw_label, (tx1, ty1, tx2, ty2, t_conf) in two_wheelers:
        # Draw two-wheeler bounding box
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
        cv2.putText(frame, f"{tw_label} {t_conf:.2f}", (tx1, ty1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for px1, py1, px2, py2 in persons:
            # Check if the person is riding the two-wheeler using IoU
            iou_score = calculate_iou((tx1, ty1, tx2, ty2), (px1, py1, px2, py2))
            
            if iou_score > 0.3:  # Person is likely riding the vehicle
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                cv2.putText(frame, "Rider", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the result image
    output_image_path = os.path.join(output_dir, f"result_{image_name}")
    cv2.imwrite(output_image_path, frame)
    print(f"Saved: {output_image_path}")

print("Processing complete!")
