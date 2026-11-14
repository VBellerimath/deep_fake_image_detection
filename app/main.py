import logging
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, UnidentifiedImageError

from app.model_loader import load_all_models
from app.predict import predict_efficientnet, predict_smaug

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==================================================
# CONFIGURATION
# ==================================================

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
}

AVAILABLE_MODELS = {
    "efficientnet",
    "smaug",
}


# ==================================================
# APPLICATION LIFESPAN
# ==================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all detection models once during application startup.
    """

    logger.info("Starting Deepfake Detection API.")

    app.state.models = load_all_models()

    logger.info("Deepfake Detection API started; all models are ready.")

    yield

    logger.info("Deepfake Detection API stopped.")


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Deep Fake Image Detection API",
    description=(
        "Upload an image to detect whether it is REAL "
        "or AI-generated (FAKE) using EfficientNet or Smaug."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ==================================================
# CORS CONFIGURATION
# ==================================================

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


# ==================================================
# IMAGE VALIDATION
# ==================================================


async def validate_uploaded_image(file: UploadFile) -> Image.Image:
    """
    Validate uploaded image format, size and image contents.
    Returns a PIL RGB image.
    """

    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("Rejected image upload with unsupported content type.")
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported.",
        )

    # Read maximum allowed size + 1 byte
    image_bytes = await file.read(MAX_FILE_SIZE + 1)

    # Check empty file
    if not image_bytes:
        logger.warning("Rejected empty image upload.")
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # Check file size
    if len(image_bytes) > MAX_FILE_SIZE:
        logger.warning("Rejected image upload exceeding the maximum file size.")
        raise HTTPException(
            status_code=413,
            detail="Image size must be less than 10 MB.",
        )

    try:
        # Verify image structure
        with Image.open(BytesIO(image_bytes)) as test_image:
            test_image.verify()

        # Reopen and convert to RGB
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        logger.warning("Rejected invalid or corrupted image upload.")
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image.",
        )


# ==================================================
# MODEL PREDICTION
# ==================================================


def run_prediction(image: Image.Image, model_name: str) -> dict:
    """
    Execute prediction using the selected detection model.
    """

    if model_name == "efficientnet":

        model_data = app.state.models["efficientnet"]

        return predict_efficientnet(
            image=image,
            model=model_data["model"],
            idx_to_class=model_data["idx_to_class"],
            device=model_data["device"],
        )

    if model_name == "smaug":

        smaug_model = app.state.models["smaug"]

        return predict_smaug(
            image=image,
            model=smaug_model,
        )

    raise HTTPException(
        status_code=400,
        detail="Invalid model selection.",
    )


# ==================================================
# API ENDPOINTS
# ==================================================


@app.get("/")
async def home():
    """
    API welcome endpoint.
    """

    return {
        "message": "Deep Fake Image Detection API",
        "status": "running",
        "available_models": sorted(AVAILABLE_MODELS),
        "documentation": "/docs",
    }


@app.get("/health")
async def health():
    """
    Check API and model loading status.
    """

    models_loaded = hasattr(app.state, "models")

    return {
        "status": "healthy" if models_loaded else "starting",
        "model_loaded": models_loaded,
        "loaded_models": (list(app.state.models.keys()) if models_loaded else []),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query(
        default="smaug",
        description="Choose efficientnet or smaug",
    ),
):
    """
    Upload an image and predict REAL or FAKE.
    """

    try:
        # Validate selected model
        if model_name not in AVAILABLE_MODELS:
            logger.warning("Rejected prediction request with an invalid model.")
            raise HTTPException(
                status_code=400,
                detail="Invalid model. Choose efficientnet or smaug.",
            )

        logger.info("Received prediction request using model '%s'.", model_name)

        # Validate uploaded image
        image = await validate_uploaded_image(file)

        # Execute prediction
        result = run_prediction(
            image=image,
            model_name=model_name,
        )
        logger.info(
            "Prediction completed using model '%s': %s.",
            model_name,
            result["prediction"],
        )

        # Return prediction response
        return {
            "filename": file.filename,
            "model_used": model_name,
            **result,
        }

    finally:
        await file.close()
