# ImageGuard – Deepfake Detection Frontend

A React-based user interface for detecting AI-generated and potentially manipulated images using two deep learning models:

- **Smaug** – Pretrained high-resolution image detector.
- **EfficientNet-B0** – Custom-trained model using the CIFAKE dataset.

The frontend communicates with a FastAPI backend to upload images and display prediction results.

## 1. Features

- Upload images using file selection or drag and drop.
- Preview uploaded images before analysis.
- Support JPG and PNG formats.
- Maximum upload size: 10 MB.
- Select between Smaug and EfficientNet-B0.
- Display REAL or FAKE predictions.
- Display detector scores and confidence.
- Responsive interface for desktop and mobile.
- Real-time communication with FastAPI.

## 2. Technology Stack

| Technology | Purpose |
|---|---|
| React | Frontend UI |
| Vite | Development server and build tool |
| JavaScript | Application logic |
| CSS | UI styling |
| Lucide React | Icons |
| FastAPI | Backend API |
| Python | Model inference |

## 3. Project Structure

```text
frontend/
│
├── public/
│
├── src/
│   ├── App.jsx
│   ├── App.css
│   └── main.jsx
│
├── index.html
├── package.json
├── package-lock.json
└── README.md
```

## 4. Prerequisites

Install the following:

- Node.js (LTS version recommended)
- npm
- FastAPI backend running locally

Verify your installations:

```bash
node --version
npm --version
```

## 5. Installation

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

This installs the frontend dependencies, including the icon library.

## 6. Backend Configuration

The frontend currently connects to the FastAPI backend using:

```javascript
const API_URL = "http://127.0.0.1:8000";
```

Ensure the backend is running at this address.

The backend enables CORS for both local Vite development origins:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 7. Start the Application

### Step 1: Start FastAPI

Open a terminal in the project root:

```bash
venv\Scripts\activate.bat
```

Start the backend:

```bash
uvicorn app.main:app --reload
```

Backend API:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

### Step 2: Start React

Open another terminal:

```bash
cd frontend
npm run dev
```

Frontend URL:

http://localhost:5173

Vite opens this URL in the default browser automatically when the development
server starts.

## 8. Model Selection

The application supports two detection models.

| Model | Description |
|---|---|
| Smaug | Pretrained detector designed for AI-generated image detection |
| EfficientNet-B0 | Custom-trained two-class image classifier |

Select the required model from the dropdown before uploading an image.

### API requests

EfficientNet:

```http
POST /predict?model_name=efficientnet
```

Smaug:

```http
POST /predict?model_name=smaug
```

Both requests use multipart form data with the image field named `file`.

## 9. Prediction Response

Example:

```json
{
  "filename": "sample.png",
  "model_used": "smaug",
  "prediction": "FAKE",
  "confidence": 0.9721,
  "class_scores": {
    "FAKE": 0.9721,
    "REAL": 0.0279
  }
}
```

The frontend displays:

- Predicted class
- Detector score
- AI-generated score
- Real score
- Selected model
- Uploaded filename

Note: Detector scores are not necessarily calibrated probabilities and do not establish image authenticity.

## 10. Build for Production

Create an optimized frontend build:

```bash
npm run build
```

The production files will be generated in:

```text
frontend/dist/
```

For local production-build testing:

```bash
npm run preview
```

## 11. Troubleshooting

### API connection failed

Check that FastAPI is running:

```bash
uvicorn app.main:app --reload
```

### CORS error

Verify that FastAPI allows the frontend origin:

```text
http://localhost:5173
http://127.0.0.1:5173
```

### Image upload failed

Check:

- File format is JPG or PNG.
- Image size is below 10 MB.
- Backend upload validation is enabled.

### Model prediction failed

Check the FastAPI terminal for the complete traceback. Verify that the selected model has loaded successfully and its inference interface matches the prediction implementation.

### Backend logs

The backend writes standard Python logs to the terminal running Uvicorn. They
include startup and model-loading events, prediction requests and results, and
upload validation rejections. Application log messages do not include image
contents or uploaded filenames.

### npm dependency issues

Try:

```bash
npm install
```

If necessary, reinstall dependencies:

```bash
npm ci
```

## 12. Important Notes

- The frontend does not contain model weights.
- Model inference is performed by the FastAPI backend.
- Smaug weights are downloaded and stored by the backend.
- EfficientNet-B0 weights are maintained separately in the backend's `models` directory.
- The backend must be running before image analysis can be performed.
- Predictions are automated assessments, not proof that an image is real or fake.

## 13. Future Enhancements

- Compare predictions from both models simultaneously.
- Display image metadata and resolution.
- Add prediction history.
- Support batch image uploads.
- Add Grad-CAM visual explanations.
- Add authentication and user management.
- Deploy frontend and backend separately.

---

**Project:** ImageGuard – Deepfake Image Detection

**Frontend:** React + Vite

**Backend:** FastAPI + PyTorch