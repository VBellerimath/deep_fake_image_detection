import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

EFFICIENTNET_PATH = MODELS_DIR / "deepfake_model.pth"
SMAUG_PATH = MODELS_DIR / "smaug"

SMAUG_REPO_ID = "abacusai/Smaug-AI-Image-Detector"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 1. Load existing EfficientNet-B0 model
# --------------------------------------------------


def load_efficientnet():

    logger.info("Loading EfficientNet-B0 from %s.", EFFICIENTNET_PATH)

    if not EFFICIENTNET_PATH.exists():
        raise FileNotFoundError(f"EfficientNet model not found: {EFFICIENTNET_PATH}")

    checkpoint = torch.load(EFFICIENTNET_PATH, map_location=DEVICE, weights_only=True)

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(in_features, 2)

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)
    model.eval()

    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {index: label for label, index in class_to_idx.items()}

    logger.info("EfficientNet-B0 loaded successfully on %s.", DEVICE)

    return {"model": model, "idx_to_class": idx_to_class, "device": DEVICE}


# --------------------------------------------------
# 2. Load Smaug model
# --------------------------------------------------


def load_smaug():

    logger.info("Loading Smaug model from repository '%s'.", SMAUG_REPO_ID)

    SMAUG_PATH.mkdir(parents=True, exist_ok=True)

    # Downloads missing files and reuses existing files
    snapshot_download(repo_id=SMAUG_REPO_ID, local_dir=str(SMAUG_PATH))

    if str(SMAUG_PATH) not in sys.path:
        sys.path.insert(0, str(SMAUG_PATH))

    from smaug_image_detect import SmaugImageDetect

    detector = SmaugImageDetect.from_pretrained(str(SMAUG_PATH))

    logger.info("Smaug model loaded successfully.")

    return detector


# --------------------------------------------------
# 3. Load both models
# --------------------------------------------------


def load_all_models():

    logger.info("Initializing deepfake detection models.")

    models_dict = {}

    models_dict["efficientnet"] = load_efficientnet()

    models_dict["smaug"] = load_smaug()

    logger.info("All deepfake detection models loaded successfully.")

    return models_dict
