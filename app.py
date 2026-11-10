
from flask import Flask, request, jsonify
from flask_cors import CORS
from waste_classifier import WasteClassifier
import os
import cv2
import base64
import tempfile

app = Flask(__name__)
CORS(app)

# Locate the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best.pt"
)

# Initialize the waste classifier
classifier = WasteClassifier(
    model_path=MODEL_PATH,
    confidence=0.25
)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Medical Waste Detection API is running",
        "status": "success"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as temp:
            image_path = temp.name
            file.save(image_path)

        detections, output_image = classifier.predict(image_path)

        success, buffer = cv2.imencode(".jpg", output_image)

        if not success:
            return jsonify({
                "error": "Image encoding failed"
            }), 500

        image_base64 = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

        return jsonify({
            "success": True,
            "detections": detections,
            "annotated_image": image_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if "image_path" in locals() and os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
