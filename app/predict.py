import torch
from torchvision import transforms

# --------------------------------------------------
# EfficientNet preprocessing
# --------------------------------------------------

efficientnet_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# --------------------------------------------------
# 1. EfficientNet prediction
# --------------------------------------------------


def predict_efficientnet(image, model, idx_to_class, device):

    image = image.convert("RGB")

    image_tensor = efficientnet_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        predicted_index = torch.argmax(probabilities, dim=1).item()

    prediction = idx_to_class[predicted_index]

    class_scores = {
        idx_to_class[index]: float(probabilities[0][index].item())
        for index in range(len(idx_to_class))
    }

    return {
        "prediction": prediction,
        "confidence": round(class_scores[prediction], 4),
        "class_scores": {
            label: round(score, 4) for label, score in class_scores.items()
        },
    }


# --------------------------------------------------
# 2. Smaug prediction
# --------------------------------------------------


def predict_smaug(image, model):

    image = image.convert("RGB")

    # Smaug inference
    score = float(model(image))

    prediction = "FAKE" if score >= 0.5 else "REAL"

    return {
        "prediction": prediction,
        "confidence": round(score if prediction == "FAKE" else 1 - score, 4),
        "class_scores": {"FAKE": round(score, 4), "REAL": round(1 - score, 4)},
    }
