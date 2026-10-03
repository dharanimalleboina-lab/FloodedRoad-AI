/* ==============================================================================
   FLOODEDROADAI — AUTOMATED ROAD CAMERA SCREENING DEMO JAVASCRIPT
   JIGNASA 2026 • Track B: CNN & Computer Vision
============================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Camera & Controls
    const cameraViewport = document.getElementById("cameraViewport");
    const cameraFrameImg = document.getElementById("cameraFrameImg");
    const scannerLine = document.getElementById("scannerLine");
    const liveDot = document.getElementById("liveDot");
    const hudRecText = document.getElementById("hudRecText");
    const hudTimestamp = document.getElementById("hudTimestamp");
    const hudFrameCode = document.getElementById("hudFrameCode");
    const hudHint = document.getElementById("hudHint");

    const btnStartDemo = document.getElementById("btnStartDemo");
    const btnStopDemo = document.getElementById("btnStopDemo");
    const btnPrevFrame = document.getElementById("btnPrevFrame");
    const btnNextFrame = document.getElementById("btnNextFrame");
    const frameCounterPill = document.getElementById("frameCounterPill");
    const frameChipsContainer = document.getElementById("frameChips");

    const systemStatus = document.getElementById("systemStatus");
    const statusText = document.getElementById("statusText");

    // DOM Elements - AI Assessment Output
    const stateEmpty = document.getElementById("stateEmpty");
    const stateLoading = document.getElementById("stateLoading");
    const stateResult = document.getElementById("stateResult");

    const statusBanner = document.getElementById("statusBanner");
    const statusIcon = document.getElementById("statusIcon");
    const statusKicker = document.getElementById("statusKicker");
    const statusTitle = document.getElementById("statusTitle");

    const hazardName = document.getElementById("hazardName");
    const levelPill = document.getElementById("levelPill");
    const confValue = document.getElementById("confValue");
    const confidenceBar = document.getElementById("confidenceBar");

    const recommendationCallout = document.getElementById("recommendationCallout");
    const calloutEscalation = document.getElementById("calloutEscalation");
    const calloutBody = document.getElementById("calloutBody");
    const breakdownList = document.getElementById("breakdownList");

    // Demo state tracking
    let cameraFrames = [];
    let currentFrameIndex = 0;
    let demoIntervalTimer = null;
    let isDemoRunning = false;
    const FRAME_INTERVAL_MS = 2800; // 2.8 seconds per frame for smooth demo pacing

    // API Base URL helper
    const apiBase = window.location.origin.includes("5000") ? "" : "http://127.0.0.1:5000";

    // ==========================================================================
    // INITIALIZATION & LOAD CAMERA FRAMES PLAYLIST
    // ==========================================================================
    async function initCameraDemo() {
        try {
            const resp = await fetch(`${apiBase}/api/camera/frames`);
            if (!resp.ok) throw new Error("Could not load camera frames");
            const data = await resp.json();

            cameraFrames = data.frames || [];
            if (cameraFrames.length > 0) {
                renderFrameChips();
                // Load first frame in paused state
                loadFrame(0, false);
            }
            statusText.textContent = "AI MODEL ACTIVE (RESNET-18)";
        } catch (err) {
            console.error("Camera demo init failed:", err);
            statusText.textContent = "BACKEND DISCONNECTED";
            systemStatus.style.color = "#f87171";
            systemStatus.style.background = "rgba(239, 68, 68, 0.15)";
            systemStatus.style.borderColor = "rgba(239, 68, 68, 0.35)";
        }
    }

    function renderFrameChips() {
        frameChipsContainer.innerHTML = "";
        cameraFrames.forEach((frame, idx) => {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = `frame-chip ${idx === 0 ? "active" : ""}`;
            chip.setAttribute("data-index", idx);

            const iconMap = {
                "waterlogged_roads": "🌊",
                "potholes": "🕳️",
                "open_manholes": "⛔",
                "broken_edges": "⚠️",
                "cracks": "⚡",
                "construction_zones": "🚧"
            };
            const icon = iconMap[frame.category] || "📷";

            chip.innerHTML = `<span>${icon}</span> <span>Frame ${idx + 1}: ${frame.label_hint}</span>`;
            chip.addEventListener("click", () => {
                if (isDemoRunning) {
                    stopDemo();
                }
                loadFrame(idx, true);
            });

            frameChipsContainer.appendChild(chip);
        });
    }

    // ==========================================================================
    // FRAME LOADING & AI SCREENING
    // ==========================================================================
    async function loadFrame(index, runInference = true) {
        if (cameraFrames.length === 0) return;

        currentFrameIndex = (index + cameraFrames.length) % cameraFrames.length;
        const frame = cameraFrames[currentFrameIndex];

        // Update image src
        cameraFrameImg.src = `${apiBase}${frame.url}`;

        // Update HUD
        hudTimestamp.textContent = frame.timestamp || "10:14:00 AM";
        hudFrameCode.textContent = `FRAME ${String(frame.id).padStart(2, '0')} / ${cameraFrames.length} (${frame.frame_code})`;
        hudHint.textContent = frame.label_hint;
        frameCounterPill.textContent = `${frame.id} / ${cameraFrames.length}`;

        // Update active chip
        document.querySelectorAll(".frame-chip").forEach((chip, i) => {
            chip.classList.toggle("active", i === currentFrameIndex);
        });

        if (runInference) {
            await screenCurrentFrame(frame);
        }
    }

    async function screenCurrentFrame(frame) {
        // Visual indicator on camera viewport
        cameraViewport.classList.add("screening-active");
        scannerLine.style.display = "block";

        try {
            const resp = await fetch(`${apiBase}/api/camera/screen-frame`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    id: frame.id,
                    category: frame.category,
                    filename: frame.filename
                })
            });

            if (!resp.ok) {
                throw new Error(`Screening failed: ${resp.status}`);
            }

            const data = await resp.json();
            renderAssessmentResult(data);
        } catch (err) {
            console.error("Screening error:", err);
        } finally {
            setTimeout(() => {
                scannerLine.style.display = "none";
            }, 600);
        }
    }

    // ==========================================================================
    // RENDER AI ASSESSMENT RESULTS
    // ==========================================================================
    function renderAssessmentResult(data) {
        showState("result");

        const isAttention = data.status === "ATTENTION REQUIRED";
        const isSafeOrUncertain = data.status === "NO CLEAR HAZARD IDENTIFIED";

        // Status Banner Styling
        statusBanner.className = "status-banner " + (isSafeOrUncertain ? "banner-safe" : "");
        statusIcon.textContent = isAttention ? (data.icon || "🚨") : "✓";
        statusKicker.textContent = isAttention ? "SAFETY ESCALATION" : "UNCERTAINTY DISCLOSURE";
        statusTitle.textContent = data.status;

        // Condition Title & Attention Pill
        hazardName.textContent = data.escalation_title || data.condition.toUpperCase();
        levelPill.textContent = (data.attention_level || "ATTENTION REQUIRED") + " ATTENTION";
        levelPill.className = "level-pill " + (
            isSafeOrUncertain ? "level-safe" : 
            (data.attention_level === "MODERATE" ? "level-mod" : "")
        );

        // Confidence Meter
        const confNum = Number(data.confidence) || 0;
        confValue.textContent = `${confNum.toFixed(2)}%`;
        confidenceBar.style.width = "0%";
        setTimeout(() => {
            confidenceBar.style.width = `${Math.min(100, Math.max(5, confNum))}%`;
        }, 80);

        // Escalation / Recommendation Callout
        recommendationCallout.className = "recommendation-callout " + (isSafeOrUncertain ? "callout-safe" : "");
        calloutEscalation.textContent = data.escalation_message || "Assessment recorded.";
        calloutBody.textContent = data.recommendation || "";

        // Render 6-Class Probabilities Distribution
        renderBreakdownList(data.probabilities || [], data.condition_key);
    }

    function renderBreakdownList(probabilities, topKey) {
        breakdownList.innerHTML = "";

        probabilities.forEach((item, index) => {
            const row = document.createElement("div");
            const isTop = index === 0;
            row.className = `breakdown-row ${isTop ? "top-class" : ""}`;

            row.innerHTML = `
                <span class="breakdown-label" title="${item.label}">${item.label}</span>
                <div class="breakdown-bar-bg">
                    <div class="breakdown-bar-fill" style="width: 0%;"></div>
                </div>
                <span class="breakdown-val">${item.score.toFixed(1)}%</span>
            `;

            breakdownList.appendChild(row);

            // Animate bar width
            setTimeout(() => {
                const fill = row.querySelector(".breakdown-bar-fill");
                if (fill) {
                    fill.style.width = `${Math.max(2, item.score)}%`;
                }
            }, 80 + (index * 30));
        });
    }

    function showState(state) {
        stateEmpty.style.display = state === "empty" ? "flex" : "none";
        stateLoading.style.display = state === "loading" ? "flex" : "none";
        stateResult.style.display = state === "result" ? "flex" : "none";
    }

    // ==========================================================================
    // DEMO CONTROLS (START / STOP / PREV / NEXT)
    // ==========================================================================
    function startDemo() {
        if (isDemoRunning) return;
        isDemoRunning = true;

        btnStartDemo.disabled = true;
        btnStopDemo.disabled = false;

        liveDot.className = "live-dot active";
        liveDot.textContent = "● AUTOMATIC SCREENING ACTIVE";
        hudRecText.textContent = "CAM-01 • ACTIVE STREAM";

        // Screen current frame immediately
        loadFrame(currentFrameIndex, true);

        // Start cycle
        demoIntervalTimer = setInterval(() => {
            const nextIdx = (currentFrameIndex + 1) % cameraFrames.length;
            loadFrame(nextIdx, true);
        }, FRAME_INTERVAL_MS);
    }

    function stopDemo() {
        isDemoRunning = false;
        if (demoIntervalTimer) {
            clearInterval(demoIntervalTimer);
            demoIntervalTimer = null;
        }

        btnStartDemo.disabled = false;
        btnStopDemo.disabled = true;

        liveDot.className = "live-dot";
        liveDot.textContent = "● DEMO PAUSED";
        hudRecText.textContent = "CAM-01 • PAUSED";
        cameraViewport.classList.remove("screening-active");
    }

    btnStartDemo.addEventListener("click", startDemo);
    btnStopDemo.addEventListener("click", stopDemo);

    btnPrevFrame.addEventListener("click", () => {
        if (isDemoRunning) stopDemo();
        const prevIdx = (currentFrameIndex - 1 + cameraFrames.length) % cameraFrames.length;
        loadFrame(prevIdx, true);
    });

    btnNextFrame.addEventListener("click", () => {
        if (isDemoRunning) stopDemo();
        const nextIdx = (currentFrameIndex + 1) % cameraFrames.length;
        loadFrame(nextIdx, true);
    });

    // Start initialization
    initCameraDemo();
});