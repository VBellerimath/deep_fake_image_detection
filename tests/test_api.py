from io import BytesIO
import logging

import pytest
from PIL import Image
from fastapi.testclient import TestClient

import app.main as main

# --------------------------------------------------
# Test image generator
# --------------------------------------------------


def create_test_image():

    image = Image.new("RGB", (512, 512), color="blue")

    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")

    return image_bytes.getvalue()


# --------------------------------------------------
# Mock model predictions
# --------------------------------------------------


@pytest.fixture
def client(monkeypatch):

    fake_models = {
        "efficientnet": {
            "model": object(),
            "idx_to_class": {0: "FAKE", 1: "REAL"},
            "device": "cpu",
        },
        "smaug": object(),
    }

    # Avoid downloading or loading actual models
    monkeypatch.setattr(main, "load_all_models", lambda: fake_models)

    # Mock EfficientNet prediction
    monkeypatch.setattr(
        main,
        "predict_efficientnet",
        lambda **kwargs: {
            "prediction": "REAL",
            "confidence": 0.95,
            "class_scores": {"FAKE": 0.05, "REAL": 0.95},
        },
    )

    # Mock Smaug prediction
    monkeypatch.setattr(
        main,
        "predict_smaug",
        lambda **kwargs: {
            "prediction": "FAKE",
            "confidence": 0.97,
            "class_scores": {"FAKE": 0.97, "REAL": 0.03},
        },
    )

    with TestClient(main.app) as test_client:
        yield test_client


# --------------------------------------------------
# 1. Home endpoint
# --------------------------------------------------


def test_home_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"
    assert "efficientnet" in data["available_models"]
    assert "smaug" in data["available_models"]


# --------------------------------------------------
# 2. Health endpoint
# --------------------------------------------------


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert set(data["loaded_models"]) == {"efficientnet", "smaug"}


# --------------------------------------------------
# 3. Smaug prediction
# --------------------------------------------------


def test_smaug_prediction(client, caplog):

    image_bytes = create_test_image()

    with caplog.at_level(logging.INFO, logger="app.main"):
        response = client.post(
            "/predict?model_name=smaug",
            files={"file": ("test.png", image_bytes, "image/png")},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "FAKE"
    assert data["model_used"] == "smaug"
    assert data["confidence"] == 0.97
    assert "Prediction completed using model 'smaug': FAKE." in caplog.text


# --------------------------------------------------
# 4. EfficientNet prediction
# --------------------------------------------------


def test_efficientnet_prediction(client):

    image_bytes = create_test_image()

    response = client.post(
        "/predict?model_name=efficientnet",
        files={"file": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "REAL"
    assert data["model_used"] == "efficientnet"
    assert data["confidence"] == 0.95


# --------------------------------------------------
# 5. Default model prediction
# --------------------------------------------------


def test_default_model_is_smaug(client):

    image_bytes = create_test_image()

    response = client.post(
        "/predict", files={"file": ("test.png", image_bytes, "image/png")}
    )

    assert response.status_code == 200

    assert response.json()["model_used"] == "smaug"


# --------------------------------------------------
# 6. Invalid model
# --------------------------------------------------


def test_invalid_model(client):

    image_bytes = create_test_image()

    response = client.post(
        "/predict?model_name=unknown",
        files={"file": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 400


# --------------------------------------------------
# 7. Unsupported file format
# --------------------------------------------------


def test_unsupported_file_format(client, caplog):

    with caplog.at_level(logging.WARNING, logger="app.main"):
        response = client.post(
            "/predict",
            files={"file": ("test.pdf", b"sample content", "application/pdf")},
        )

    assert response.status_code == 400
    assert "unsupported content type" in caplog.text


# --------------------------------------------------
# 8. Empty image
# --------------------------------------------------


def test_empty_image(client):

    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})

    assert response.status_code == 400


# --------------------------------------------------
# 9. Corrupted image
# --------------------------------------------------


def test_corrupted_image(client):

    response = client.post(
        "/predict",
        files={"file": ("corrupted.png", b"this is not a valid image", "image/png")},
    )

    assert response.status_code == 400


# --------------------------------------------------
# 10. Image larger than 10 MB
# --------------------------------------------------


def test_image_size_limit(client):

    large_image = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/predict", files={"file": ("large.png", large_image, "image/png")}
    )

    assert response.status_code == 413
