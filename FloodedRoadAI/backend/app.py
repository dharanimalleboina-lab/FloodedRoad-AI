import os
import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# ==============================================================================
# FLASK CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "IRDID", "images")
MODEL_PATH = os.path.join(BASE_DIR, "model", "flooded_road_model.pth")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# ==============================================================================
# TAXONOMY & SAFETY ESCALATION POLICIES
# ==============================================================================
HAZARD_PROFILES = {
    "waterlogged_roads": {
        "label": "Waterlogged Road",
        "badge": "ATTENTION REQUIRED",
        "attention_level": "HIGH",
        "escalation_title": "WATERLOGGED ROAD DETECTED",
        "escalation_message": "Visible waterlogging may affect road access. Further assessment recommended.",
        "recommendation": "Surface water detected across roadway. Stagnant or flowing water can obscure deep potholes or create hydroplaning hazards. Exercise caution; seek verified alternative access routes.",
        "icon": "🌊"
    },
    "potholes": {
        "label": "Pothole",
        "badge": "ATTENTION REQUIRED",
        "attention_level": "HIGH",
        "escalation_title": "POTHOLE DETECTED",
        "escalation_message": "Potential road access and vehicle damage issue detected. Further assessment recommended.",
        "recommendation": "Severe depression or structural cavity detected in road surface. Risk of tire puncture, suspension damage, and vehicle loss of control.",
        "icon": "🕳️"
    },
    "open_manholes": {
        "label": "Open Manhole",
        "badge": "ATTENTION REQUIRED",
        "attention_level": "CRITICAL",
        "escalation_title": "OPEN MANHOLE DETECTED",
        "escalation_message": "Critical road surface hazard detected. Immediate field escalation and marking required.",
        "recommendation": "Uncovered or dislodged utility opening detected. Extreme hazard for vehicles, two-wheelers, and pedestrians, especially in low visibility or wet weather.",
        "icon": "⛔"
    },
    "broken_edges": {
        "label": "Broken Road Edge",
        "badge": "ATTENTION REQUIRED",
        "attention_level": "MODERATE",
        "escalation_title": "BROKEN ROAD EDGE DETECTED",
        "escalation_message": "Shoulder erosion or edge collapse detected along road boundary. Caution recommended.",
        "recommendation": "Road margin integrity is compromised. Avoid driving on soft or collapsed shoulders, particularly during saturated soil conditions.",
        "icon": "⚠️"
    },
    "cracks": {
        "label": "Road Cracks",
        "badge": "ATTENTION REQUIRED",
        "attention_level": "MODERATE",
        "escalation_title": "ROAD CRACK DETECTED",
        "escalation_message": "Surface fracture patterns detected. Structural assessment recommended.",
        "recommendation": "Longitudinal, transverse, or fatigue cracking visible. While passable at low speeds, cracks accelerate water infiltration and rapid pothole formation.",
        "icon": "⚡"
    },
    "construction_zones": {
        "label": "Construction Zone",
        "badge": "ATTENTION REQUIRED",
        "attention_level": "MODERATE",
        "escalation_title": "CONSTRUCTION ZONE DETECTED",
        "escalation_message": "Active road repair, debris, or barricade zone detected. Caution recommended.",
        "recommendation": "Road work, unpaved gravel, or temporary barriers present. Traffic flow likely constricted; expect sudden stops and uneven grade.",
        "icon": "🚧"
    }
}

CONFIDENCE_THRESHOLD = 0.50  # Softmax threshold below which road is non-confidently classified

# Curated demonstration sequence of prerecorded camera frames from stored dataset
CAMERA_FRAMES = [
    {
        "id": 1,
        "frame_code": "CAM01-FR01",
        "category": "waterlogged_roads",
        "filename": "img0601.jpg",
        "label_hint": "Waterlogged Road",
        "timestamp": "10:14:02 AM",
        "url": "/api/sample-image/waterlogged_roads/img0601.jpg"
    },
    {
        "id": 2,
        "frame_code": "CAM01-FR02",
        "category": "potholes",
        "filename": "img0001.jpg",
        "label_hint": "Pothole",
        "timestamp": "10:14:05 AM",
        "url": "/api/sample-image/potholes/img0001.jpg"
    },
    {
        "id": 3,
        "frame_code": "CAM01-FR03",
        "category": "open_manholes",
        "filename": "img0301.jpg",
        "label_hint": "Open Manhole",
        "timestamp": "10:14:08 AM",
        "url": "/api/sample-image/open_manholes/img0301.jpg"
    },
    {
        "id": 4,
        "frame_code": "CAM01-FR04",
        "category": "broken_edges",
        "filename": "img0801.jpg",
        "label_hint": "Broken Road Edge",
        "timestamp": "10:14:11 AM",
        "url": "/api/sample-image/broken_edges/img0801.jpg"
    },
    {
        "id": 5,
        "frame_code": "CAM01-FR05",
        "category": "cracks",
        "filename": "img0201.jpg",
        "label_hint": "Road Cracks",
        "timestamp": "10:14:14 AM",
        "url": "/api/sample-image/cracks/img0201.jpg"
    },
    {
        "id": 6,
        "frame_code": "CAM01-FR06",
        "category": "construction_zones",
        "filename": "img1001.jpg",
        "label_hint": "Construction Zone",
        "timestamp": "10:14:17 AM",
        "url": "/api/sample-image/construction_zones/img1001.jpg"
    },
    {
        "id": 7,
        "frame_code": "CAM01-FR07",
        "category": "waterlogged_roads",
        "filename": "img0602.jpg",
        "label_hint": "Waterlogged Road",
        "timestamp": "10:14:20 AM",
        "url": "/api/sample-image/waterlogged_roads/img0602.jpg"
    },
    {
        "id": 8,
        "frame_code": "CAM01-FR08",
        "category": "potholes",
        "filename": "img0002.jpg",
        "label_hint": "Pothole",
        "timestamp": "10:14:23 AM",
        "url": "/api/sample-image/potholes/img0002.jpg"
    },
    {
        "id": 9,
        "frame_code": "CAM01-FR09",
        "category": "open_manholes",
        "filename": "img0302.jpg",
        "label_hint": "Open Manhole",
        "timestamp": "10:14:26 AM",
        "url": "/api/sample-image/open_manholes/img0302.jpg"
    },
    {
        "id": 10,
        "frame_code": "CAM01-FR10",
        "category": "broken_edges",
        "filename": "img0802.jpg",
        "label_hint": "Broken Road Edge",
        "timestamp": "10:14:29 AM",
        "url": "/api/sample-image/broken_edges/img0802.jpg"
    },
    {
        "id": 11,
        "frame_code": "CAM01-FR11",
        "category": "cracks",
        "filename": "img0202.jpg",
        "label_hint": "Road Cracks",
        "timestamp": "10:14:32 AM",
        "url": "/api/sample-image/cracks/img0202.jpg"
    },
    {
        "id": 12,
        "frame_code": "CAM01-FR12",
        "category": "construction_zones",
        "filename": "img1002.jpg",
        "label_hint": "Construction Zone",
        "timestamp": "10:14:35 AM",
        "url": "/api/sample-image/construction_zones/img1002.jpg"
    }
]

# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Flask backend initializing on device: {device}")

predict_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = None
model_classes = []

def init_model():
    global model, model_classes
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model weights not found at {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "classes" in checkpoint:
        model_classes = checkpoint["classes"]
        state_dict = checkpoint["model_state_dict"]
    else:
        model_classes = sorted(list(HAZARD_PROFILES.keys()))
        state_dict = checkpoint

    loaded_model = models.resnet18(weights=None)
    loaded_model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(loaded_model.fc.in_features, len(model_classes))
    )
    
    try:
        loaded_model.load_state_dict(state_dict)
    except Exception:
        # Fallback if checkpoint was direct Linear
        loaded_model.fc = nn.Linear(512, len(model_classes))
        loaded_model.load_state_dict(state_dict)

    loaded_model = loaded_model.to(device)
    loaded_model.eval()
    model = loaded_model
    print(f"Model successfully loaded with {len(model_classes)} classes: {model_classes}")

init_model()

def evaluate_pil_image(image: Image.Image):
    """Core evaluation function shared across all screening methods."""
    global model, model_classes
    if model is None:
        init_model()
        if model is None:
            raise RuntimeError("Model could not be initialized")

    tensor = predict_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        max_prob, pred_idx = torch.max(probabilities, dim=0)

    confidence = round(max_prob.item() * 100.0, 2)
    predicted_key = model_classes[pred_idx.item()]

    # Assemble all probabilities sorted descending
    probs_list = []
    for i, cls_name in enumerate(model_classes):
        profile = HAZARD_PROFILES.get(cls_name, {})
        probs_list.append({
            "key": cls_name,
            "label": profile.get("label", cls_name.replace("_", " ").title()),
            "score": round(probabilities[i].item() * 100.0, 2)
        })
    probs_list.sort(key=lambda x: x["score"], reverse=True)

    # Apply strict confidence threshold logic
    if max_prob.item() < CONFIDENCE_THRESHOLD:
        status = "NO CLEAR HAZARD IDENTIFIED"
        condition = "No Clear Hazard Identified"
        attention_level = "INFORMATIONAL"
        escalation_title = "NO CLEAR HAZARD IDENTIFIED"
        escalation_message = "No obvious target road hazard was confidently identified from this image. This does not certify road safety."
        recommendation = "Visual features did not meet confidence threshold for the 6 monitored hazard types. Never assume road clearance; always remain alert for unmonitored hazards, blind spots, or changing conditions."
        badge = "NO CLEAR HAZARD IDENTIFIED"
        icon = "✓"
    else:
        profile = HAZARD_PROFILES.get(predicted_key, {})
        status = "ATTENTION REQUIRED"
        condition = profile.get("label", predicted_key.replace("_", " ").title())
        attention_level = profile.get("attention_level", "HIGH")
        escalation_title = profile.get("escalation_title", f"{condition.upper()} DETECTED")
        escalation_message = profile.get("escalation_message", "Potential road access issue detected. Further assessment recommended.")
        recommendation = profile.get("recommendation", "Further field assessment recommended.")
        badge = profile.get("badge", "ATTENTION REQUIRED")
        icon = profile.get("icon", "🚨")

    return {
        "status": status,
        "badge": badge,
        "condition": condition,
        "condition_key": predicted_key,
        "confidence": confidence,
        "attention_level": attention_level,
        "escalation_title": escalation_title,
        "escalation_message": escalation_message,
        "recommendation": recommendation,
        "icon": icon,
        "probabilities": probs_list,
        "disclaimer": (
            "Safety Notice: This AI assessment is based strictly on visual pattern analysis "
            "of the uploaded image. It does not measure water depth, guarantee passability, "
            "or certify road safety."
        )
    }

# ==============================================================================
# API ROUTES
# ==============================================================================
@app.route("/api/camera/frames", methods=["GET"])
def get_camera_frames():
    """Returns the sequence of prerecorded camera frames for automatic screening demonstration."""
    return jsonify({
        "camera_id": "ROAD-CAM-01",
        "location": "Sector A-4 Arterial Corridor",
        "mode": "Prerecorded camera frames — prototype demonstration",
        "is_live_cctv": False,
        "disclaimer": "Demonstration using stored dataset images. Not a live CCTV feed.",
        "frames": CAMERA_FRAMES
    })

@app.route("/api/camera/screen-frame", methods=["POST"])
def screen_camera_frame():
    """Screens a specific prerecorded frame by ID or category/filename via PyTorch inference."""
    data = request.get_json(silent=True) or {}
    frame_id = data.get("id") or data.get("frame_id")
    category = data.get("category")
    filename = data.get("filename")

    target_frame = None
    if frame_id is not None:
        try:
            f_id = int(frame_id)
            target_frame = next((f for f in CAMERA_FRAMES if f["id"] == f_id), None)
        except (ValueError, TypeError):
            pass

    if target_frame is None and category and filename:
        safe_cat = os.path.basename(category)
        safe_file = os.path.basename(filename)
        img_path = os.path.join(DATASET_DIR, safe_cat, safe_file)
    elif target_frame is not None:
        img_path = os.path.join(DATASET_DIR, target_frame["category"], target_frame["filename"])
    else:
        # Default to first frame
        target_frame = CAMERA_FRAMES[0]
        img_path = os.path.join(DATASET_DIR, target_frame["category"], target_frame["filename"])

    if not os.path.exists(img_path):
        return jsonify({"error": f"Frame image not found at: {img_path}"}), 404

    try:
        image = Image.open(img_path).convert("RGB")
        result = evaluate_pil_image(image)
        if target_frame:
            result["frame_info"] = {
                "id": target_frame["id"],
                "frame_code": target_frame["frame_code"],
                "timestamp": target_frame["timestamp"],
                "label_hint": target_frame["label_hint"],
                "mode": "Prerecorded camera frames — prototype demonstration"
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Fallback manual image screening route (multipart file upload)."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded in request form field 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty file uploaded"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        result = evaluate_pil_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

@app.route("/api/samples", methods=["GET"])
def get_sample_images():
    """Returns sample benchmark images."""
    return jsonify({"samples": CAMERA_FRAMES})

@app.route("/api/sample-image/<category>/<filename>", methods=["GET"])
def serve_sample_image(category, filename):
    safe_cat = os.path.basename(category)
    safe_file = os.path.basename(filename)
    img_path = os.path.join(DATASET_DIR, safe_cat, safe_file)
    if not os.path.exists(img_path):
        return jsonify({"error": "Sample image not found"}), 404
    return send_file(img_path, mimetype="image/jpeg")

# ==============================================================================
# STATIC FRONTEND SERVING
# ==============================================================================
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>", methods=["GET"])
def static_proxy(path):
    file_path = os.path.join(FRONTEND_DIR, path)
    if os.path.exists(file_path):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")

# ==============================================================================
# SERVER ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)