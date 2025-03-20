import cv2
import json
import os
from ultralytics.models.yolo import YOLO  # Using cloned Ultralytics repo
from paddleocr import PaddleOCR  # OCR module

class CustomModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Load YOLO model
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize OCR

    def infer(self, image_path, output_dir="output"):
        results = self.model(image_path)  # Run YOLO detection

        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        cropped_dir = os.path.join(output_dir, "cropped_plates")
        os.makedirs(cropped_dir, exist_ok=True)

        img = cv2.imread(image_path)
        detections = []

        for i, result in enumerate(results):
            for j, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                # Ensure we correctly extract the bounding box
                if box.shape[0] == 4:  # If only x1, y1, x2, y2 are available
                    x1, y1, x2, y2 = map(int, box[:4])
                    confidence = None  # No confidence score
                elif box.shape[0] >= 5:  # If confidence score exists
                    x1, y1, x2, y2, confidence = map(float, box[:5])
                    confidence = round(confidence, 2)  # Round for readability

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if confidence is not None:
                    cv2.putText(img, f"{confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop and save the detected plate
                cropped_plate = img[y1:y2, x1:x2]
                cropped_plate_path = os.path.join(cropped_dir, f"plate_{i}_{j}.png")
                cv2.imwrite(cropped_plate_path, cropped_plate)
                print(f"Saved cropped plate: {cropped_plate_path}")

                # Perform OCR on the cropped plate
                ocr_result = self.ocr.ocr(cropped_plate_path, cls=True)
                plate_text = ""
                if ocr_result and ocr_result[0]:
                    plate_text = " ".join([word[1][0] for word in ocr_result[0]])

                print(f"OCR Result: {plate_text}")

                # Save detection details
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": confidence if confidence is not None else "N/A",
                    "cropped_image": cropped_plate_path,
                    "plate_text": plate_text
                })

        # Save annotated image
        output_image = os.path.join(output_dir, "output.png")
        cv2.imwrite(output_image, img)
        print(f"Output image saved at {output_image}")

        # Save results as JSON
        output_json = os.path.join(output_dir, "results.json")
        with open(output_json, "w") as json_file:
            json.dump(detections, json_file, indent=4)
        print(f"Results saved at {output_json}")

        return detections
