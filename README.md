
# Deep Fake Image Detection

AI-powered image classification application that predicts
whether an uploaded image is REAL or AI-generated (FAKE).
The application includes a React frontend and a FastAPI backend.

## Technology Stack

- Python
- PyTorch
- EfficientNet-B0
- FastAPI
- Torchvision
- Pillow
- Uvicorn
- React
- Vite

## Model Details

The API supports two models:

- **EfficientNet-B0**: Custom-trained binary classifier. It resizes RGB
  images to 224 x 224 and predicts REAL or FAKE. It was trained on CIFAKE.
- **Smaug**: Pretrained image detector downloaded by the backend on first
  startup and cached under `models/smaug/`.

## Training Results

The following metrics are for EfficientNet-B0:

- Training accuracy: 99.88%
- Validation accuracy: 98.87%
- Validation loss: 0.0437

These metrics are from the training and validation run.
They do not establish performance on unseen generators.

## Project Structure

```text
deep_fake_image_detection/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model_loader.py
│   └── predict.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── README.md
├── models/
│   └── deepfake_model.pth
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate environment

Windows Command Prompt:

```cmd
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run API

From the project root, activate the virtual environment and start the API:

```bash
uvicorn app.main:app --reload
```

The first startup downloads the Smaug model if it is not already cached.

## Run Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the frontend URL printed by Vite (typically http://localhost:5173).
The frontend connects to the API at http://127.0.0.1:8000.

## API Documentation

Open:

http://127.0.0.1:8000/docs

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | API information |
| GET | /health | Health check |
| POST | /predict | Predict image class |

`POST /predict` accepts a multipart image field named `file` and an optional
`model_name` query parameter (`smaug` by default, or `efficientnet`). JPG/JPEG
and PNG files up to 10 MB are supported.

## Prediction Response

```json
{
  "filename": "sample.jpg",
  "model_used": "smaug",
  "prediction": "FAKE",
  "confidence": 0.987,
  "class_scores": {
    "FAKE": 0.987,
    "REAL": 0.013
  }
}
```

The values are illustrative, not an actual prediction.

## Backend Logs

The backend uses Python's standard logging and writes logs to the backend
process output. Logs include application and model startup/shutdown, prediction
requests and results, and rejected uploads. Uploaded image contents and
filenames are not included in these application log messages. When running
locally, check the terminal running Uvicorn for these logs.

## Limitations

The model was trained on CIFAKE. Performance on unseen
AI image generators, edited images and real-world images
must be evaluated separately.

Detector confidence scores are not calibrated probabilities and
should not be interpreted as proof of authenticity.
