import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==============================================================================
# CLASS DEFINITIONS & DESCRIPTIONS
# ==============================================================================
CLASS_CONFIG = {
    "waterlogged_roads": {
        "display_name": "Waterlogged Road",
        "title": "WATERLOGGED ROAD DETECTED",
        "attention_level": "HIGH",
        "escalation_message": "Visible waterlogging may affect road access. Further assessment recommended.",
        "icon": "🌊"
    },
    "potholes": {
        "display_name": "Pothole",
        "title": "POTHOLE DETECTED",
        "attention_level": "ATTENTION REQUIRED",
        "escalation_message": "Potential road access and vehicle damage issue detected. Further assessment recommended.",
        "icon": "⚠️"
    },
    "open_manholes": {
        "display_name": "Open Manhole",
        "title": "OPEN MANHOLE DETECTED",
        "attention_level": "CRITICAL ATTENTION",
        "escalation_message": "Severe road access and safety hazard detected. Prompt field verification required.",
        "icon": "⛔"
    },
    "broken_edges": {
        "display_name": "Broken Road Edge",
        "title": "BROKEN ROAD EDGE DETECTED",
        "attention_level": "ATTENTION REQUIRED",
        "escalation_message": "Shoulder erosion or edge collapse detected along road boundary. Caution recommended.",
        "icon": "⚠️"
    },
    "cracks": {
        "display_name": "Road Cracks",
        "title": "ROAD CRACK DETECTED",
        "attention_level": "MODERATE ATTENTION",
        "escalation_message": "Surface fracture patterns detected. Structural assessment recommended.",
        "icon": "⚡"
    },
    "construction_zones": {
        "display_name": "Construction Zone",
        "title": "CONSTRUCTION ZONE DETECTED",
        "attention_level": "ATTENTION REQUIRED",
        "escalation_message": "Active road repair, barrier, or work zone detected. Reduced speed recommended.",
        "icon": "🚧"
    }
}

CONFIDENCE_THRESHOLD = 0.50  # Below 50% triggers "NO CLEAR HAZARD IDENTIFIED"

# ==============================================================================
# MODEL LOADER
# ==============================================================================
def load_model(model_path="model/flooded_road_model.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run 'python model/train.py' first.")

    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract classes
    if isinstance(checkpoint, dict) and "classes" in checkpoint:
        classes = checkpoint["classes"]
        state_dict = checkpoint["model_state_dict"]
    else:
        # Fallback for raw state_dict
        classes = sorted(list(CLASS_CONFIG.keys()))
        state_dict = checkpoint

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, len(classes))
    )
    
    # Try loading with strict=False in case dropout wrapper changed
    try:
        model.load_state_dict(state_dict)
    except Exception:
        # Fallback if checkpoint was saved directly on Linear layer
        model.fc = nn.Linear(model.fc[1].in_features if isinstance(model.fc, nn.Sequential) else model.fc.in_features, len(classes))
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model, classes, device

# ==============================================================================
# PREPROCESSING TRANSFORM
# ==============================================================================

predict_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
# PREDICT FUNCTION
# ==============================================================================
def assess_road_image(image_path, model_path="model/flooded_road_model.pth"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    model, classes, device = load_model(model_path)
    
    image = Image.open(image_path).convert("RGB")
    tensor = predict_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        max_prob, pred_idx = torch.max(probabilities, dim=0)

    confidence = max_prob.item() * 100.0
    predicted_class = classes[pred_idx.item()]
    
    # All class probabilities
    all_probs = {
        classes[i]: probabilities[i].item() * 100.0
        for i in range(len(classes))
    }

    # Evaluate against confidence threshold
    if max_prob.item() < CONFIDENCE_THRESHOLD:
        status = "NO CLEAR HAZARD IDENTIFIED"
        escalation_title = "NO CLEAR HAZARD IDENTIFIED"
        escalation_message = "No obvious target road hazard was confidently identified from this image. This does not certify road safety."
        attention_level = "INFORMATIONAL"
        detected_condition = "Uncertain / No Target Hazard Confidently Identified"
        icon = "✓"
    else:
        status = "ATTENTION REQUIRED"
        cfg = CLASS_CONFIG.get(predicted_class, {
            "display_name": predicted_class.replace("_", " ").title(),
            "title": f"{predicted_class.upper()} DETECTED",
            "attention_level": "ATTENTION REQUIRED",
            "escalation_message": "Potential road access issue detected. Further assessment recommended.",
            "icon": "⚠️"
        })
        detected_condition = cfg["display_name"]
        escalation_title = cfg["title"]
        escalation_message = cfg["escalation_message"]
        attention_level = cfg["attention_level"]
        icon = cfg["icon"]

    return {
        "status": status,
        "condition": detected_condition,
        "condition_key": predicted_class,
        "confidence": confidence,
        "attention_level": attention_level,
        "escalation_title": escalation_title,
        "escalation_message": escalation_message,
        "icon": icon,
        "all_probabilities": all_probs,
        "disclaimer": (
            "Safety Notice: This AI assessment is based strictly on visual pattern analysis "
            "of the uploaded image. It does not measure water depth, guarantee passability, "
            "or certify road safety."
        )
    }

# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model/predict.py <path_to_image>")
        sys.exit(1)

    image_file = sys.argv[1]
    res = assess_road_image(image_file)

    print("\n=======================================================")
    print("           FLOODED ROAD AI - ROAD ASSESSMENT          ")
    print("=======================================================")
    print(f"Status:             {res['icon']} {res['status']}")
    print(f"Detected Condition: {res['condition']}")
    print(f"Model Confidence:   {res['confidence']:.2f}%")
    print(f"Attention Level:    {res['attention_level']}")
    print(f"Escalation Notice:  {res['escalation_title']}")
    print(f"Recommendation:     {res['escalation_message']}")
    print("-------------------------------------------------------")
    print("All Class Probabilities:")
    for cls_name, prob in sorted(res['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cls_name:20s}: {prob:5.2f}%")
    print("-------------------------------------------------------")
    print(f"NOTE: {res['disclaimer']}")
    print("=======================================================\n")