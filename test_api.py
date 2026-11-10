
import requests
import base64
import os

API_URL = "http://127.0.0.1:5000/predict"

IMAGE_PATH = "test_images/test1.jpg"

# Check whether image exists
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Send image to Flask API
with open(IMAGE_PATH, "rb") as image:
    response = requests.post(
        API_URL,
        files={"image": ("test1.jpg", image, "image/jpeg")}
    )

print("Status Code:", response.status_code)

result = response.json()

if response.status_code == 200 and result.get("success"):

    print("\nDetection Results:")

    for detection in result["detections"]:
        print("Class:", detection["class_name"])
        print("Confidence:", detection["confidence"])
        print("Bin:", detection["bin"])
        print("Bounding Box:", detection["bbox"])
        print("-------------------------")

    # Save annotated image
    image_data = base64.b64decode(result["annotated_image"])

    output_path = "outputs/test1_prediction.jpg"

    with open(output_path, "wb") as output:
        output.write(image_data)

    print("\nAnnotated image saved at:", output_path)

else:
    print("API Error:", result)
