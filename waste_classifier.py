
import cv2
from ultralytics import YOLO


# ==========================================
# BIOMEDICAL WASTE CLASSIFIER
# ==========================================

class WasteClassifier:

    def __init__(self, model_path, confidence=0.25):

        self.model = YOLO(model_path)
        self.confidence = confidence

        # BGR colours for OpenCV
        self.colors = {
            "YELLOW": (0, 255, 255),
            "RED": (0, 0, 255),
            "WHITE": (255, 255, 255),
            "BLUE": (255, 0, 0),
            "MANUAL REVIEW": (128, 128, 128)
        }

        # Initial prototype mapping.
        # Context-dependent items require verification.
        self.mapping = {
            "Body Tissue or organ": "YELLOW",
            "Gauze": "MANUAL REVIEW",
            "Glass_equipment": "MANUAL REVIEW",
            "Gloves": "MANUAL REVIEW",
            "Mask": "MANUAL REVIEW",
            "Metal_equipment": "MANUAL REVIEW",
            "Paper_equipment": "MANUAL REVIEW",
            "Plastic_equipment": "MANUAL REVIEW",
            "Syringe": "MANUAL REVIEW",
            "Syringe_needle": "WHITE",
            "organic_waste": "MANUAL REVIEW",
            "tweezers": "MANUAL REVIEW"
        }

    # ======================================
    # DETECT AND CLASSIFY
    # ======================================

    def predict(self, image_path):

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Cannot read image: {image_path}"
            )

        results = self.model.predict(
            source=image,
            conf=self.confidence,
            verbose=False
        )

        detections = []
        output = image.copy()

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                score = float(box.conf[0])

                class_name = self.model.names[class_id]

                bin_name = self.mapping.get(
                    class_name, "MANUAL REVIEW"
                )

                color = self.colors[bin_name]

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                # Draw bounding box
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                label = (
                    f"{bin_name} | "
                    f"{class_name} | {score:.2f}"
                )

                # Label background
                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    2
                )

                label_y = max(y1, text_h + 10)

                cv2.rectangle(
                    output,
                    (x1, label_y - text_h - 10),
                    (x1 + text_w + 8, label_y + baseline),
                    color,
                    -1
                )

                # Black text for visibility
                cv2.putText(
                    output,
                    label,
                    (x1 + 4, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

                detections.append({
                    "class_name": class_name,
                    "confidence": round(score, 4),
                    "bin": bin_name,
                    "bbox": [x1, y1, x2, y2]
                })

        return detections, output
