# Smart Medical Waste Detection and Segregation System

## Project Overview

An AI-powered medical waste detection and segregation system using YOLO11n and Flask.

The system detects medical waste objects from uploaded images, identifies their categories, predicts a suitable waste-bin recommendation, and returns annotated images with bounding boxes.

## Technologies Used

- Python 3.10
- YOLO11n (Ultralytics)
- PyTorch
- OpenCV
- Flask REST API
- NumPy

## Project Structure

```text
medical-waste-detection-api/
│
├── models/
│   ├── best.pt
│   └── best.onnx
│
├── test_images/
├── outputs/
│
├── app.py
├── waste_classifier.py
├── test_api.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Supported Waste Classes

1. Body Tissue or organ
2. Gauze
3. Glass_equipment
4. Gloves
5. Mask
6. Metal_equipment
7. Paper_equipment
8. Plastic_equipment
9. Syringe
10. Syringe_needle
11. organic_waste
12. tweezers

## Waste Bin Categories

| Category | Description |
|---|---|
| YELLOW | Prescribed yellow-category biomedical waste |
| RED | Prescribed contaminated recyclable plastic waste |
| WHITE | Prescribed sharps waste |
| BLUE | Prescribed glassware and metallic implants |
| MANUAL REVIEW | Requires additional verification |

Bin recommendations are preliminary. Final segregation must follow applicable biomedical waste regulations and facility procedures.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd medical-waste-detection-api
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Flask API

```bash
python app.py
```

The API will run at:

```text
http://127.0.0.1:5000
```

## API Endpoints

### Health Check

**GET /**

Checks whether the API is running.

### Waste Detection

**POST /predict**

Accepts an image using multipart form-data.

Form field:

```text
image: uploaded image file
```

Returns:

- Detected waste class
- Confidence score
- Bounding box coordinates
- Suggested bin category
- Annotated image encoded in Base64

## Testing

Run:

```bash
python test_api.py
```

The annotated output is saved in:

```text
outputs/test1_prediction.jpg
```

## Model Information

- Architecture: YOLO11n
- Task: Object Detection
- Number of classes: 12
- Input image size: 640 × 640
- Model weights: `models/best.pt`

## Project Scope

The ML/DL module provides image-based medical waste detection, bounding-box localization, and preliminary waste-bin recommendations.

The Flask API is designed to support integration with an IoT camera system and a mobile application.

## Disclaimer

This project is an academic prototype. Its predictions must not replace trained personnel or official biomedical waste segregation procedures.
