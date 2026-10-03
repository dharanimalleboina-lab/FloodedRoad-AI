# FloodedRoadAI — Automated Road Camera Screening
**JIGNASA 2026 • Track B: CNN & Computer Vision**  
**Problem Statement:** Flooded Road Image Assessment

---

## 🌊 Overview

FloodedRoadAI is an AI-powered visual road condition assessment system built to evaluate compromised or flooded roadways during severe weather events and infrastructure failures.

For the hackathon demonstration, the system features an **automated road camera screening pipeline ("Road Camera 01")** that screens prerecorded camera frames sequentially without requiring manual image uploads from operators.

### Monitored Road Hazards (6 IRDID Classes):
1. **Waterlogged Roads** (Submerged or partially submerged road surfaces)
2. **Potholes** (Surface depressions causing vehicle damage & access disruption)
3. **Open Manholes** (Missing or displaced utility covers)
4. **Broken Road Edges** (Shoulder erosion and pavement collapse)
5. **Road Cracks** (Structural fatigue and moisture ingress)
6. **Construction Zones** (Barricades, gravel, and active work sectors)

---

## 🚀 Key Features

* **Automated Screening Demo:** Simulates an automated road inspection camera feed ("Road Camera 01"). Operators can click **Start Demo** to automatically screen frames sequentially on a loop, or **Stop Demo** to pause and inspect any frame.
* **Prerecorded Frames Disclosure:** Explicitly identifies that stored dataset images are used for demonstration purposes (not live CCTV).
* **Real-Time PyTorch Inference:** Every single camera frame is evaluated through the fine-tuned ResNet-18 model on Apple Silicon MPS with full 6-class probability distribution.
* **Uncertainty & Rejection Threshold:** If model confidence is below 50%, the system flags **`NO CLEAR HAZARD IDENTIFIED`** with the explicit notice: *"No obvious target road hazard was confidently identified from this image. This does not certify road safety."*
* **Safety Boundaries:** Explicitly avoids claiming water depth, GPS coordinates, vehicle passability, or safety certification.

---

## 💻 Quick Start Instructions (macOS)

Open your Terminal in the project directory:

```bash
cd /Users/tanmayee/Desktop/FloodedRoadAI
source venv/bin/activate
```

### Start the Application (Backend + Frontend)

```bash
python backend/app.py
```

The server will start on:  
👉 **`http://127.0.0.1:5000`**

Open your web browser (Safari or Google Chrome) and navigate to:
```
http://127.0.0.1:5000
```

---

## 🎥 Running the Hackathon Demonstration

1. When the page loads, you will see the **Road Camera 01** viewport displaying the first camera frame.
2. Click the blue **▶ Start Demo** button:
   * The camera feed begins automated continuous screening.
   * A scanning laser overlay sweeps across each frame.
   * Every 2.8 seconds, the feed advances to the next frame and automatically sends it to the ResNet-18 model.
   * The right panel updates in real time with the actual model confidence, condition, safety escalation, and 6-class distribution.
3. Click **⏸ Stop Demo** at any time to pause the demonstration.
4. Use **◀ Prev** and **Next ▶** buttons or click any frame in the **Prerecorded Camera Frame Queue** to jump directly to a specific road hazard during judge questions.

---

## ⚠️ Limitations & Demonstration Boundaries

* **Prerecorded Demonstration:** The system demonstrates automated screening using stored dataset frames, not a live streaming camera network.
* **No Safety Certification:** The model cannot certify that a road is completely safe.
* **No Depth Measurement:** Visual image assessment cannot measure water depth or submerged road cavities.
* **Field Verification Required:** Always defer to municipal authorities and emergency service responders.
