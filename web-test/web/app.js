import {
  FilesetResolver,
  PoseLandmarker
} from "./vendor/mediapipe/vision_bundle.mjs";

const MODEL_PATHS = Object.freeze({
  lite: "./models/pose_landmarker_lite.task",
  heavy: "./models/pose_landmarker_heavy.task"
});

const MODEL_LABELS = Object.freeze({
  lite: "Lite",
  heavy: "Heavy"
});

const WASM_PATH = "./vendor/mediapipe/wasm";
const TARGET_INFERENCE_FPS = 30;
const INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;

const IDX = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28
};

const POSE_CONNECTIONS = [
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [11, 23], [12, 24],
  [23, 24],
  [23, 25], [25, 27],
  [24, 26], [26, 28]
];

const elements = {
  video: document.querySelector("#camera"),
  canvas: document.querySelector("#overlay"),
  status: document.querySelector("#status"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  resetButton: document.querySelector("#resetButton"),
  modelSwitchButton: document.querySelector("#modelSwitchButton"),
  currentModel: document.querySelector("#currentModel"),
  poseState: document.querySelector("#poseState"),
  phase: document.querySelector("#phase"),
  repCount: document.querySelector("#repCount"),
  leftKnee: document.querySelector("#leftKnee"),
  rightKnee: document.querySelector("#rightKnee"),
  latency: document.querySelector("#latency"),
  fps: document.querySelector("#fps")
};

const canvasContext = elements.canvas.getContext("2d");

const appState = {
  visionFileset: null,
  poseLandmarker: null,
  currentModel: "lite",
  isSwitchingModel: false,
  stream: null,
  isRunning: false,
  animationFrameId: null,
  lastInferenceAt: 0,
  processedFrames: 0,
  fpsWindowStartedAt: performance.now(),
  repCount: 0,
  phase: "standing"
};

window.addEventListener("load", async () => {
  await registerServiceWorker();

  elements.startButton.addEventListener("click", startCameraTest);
  elements.stopButton.addEventListener("click", stopCameraTest);
  elements.resetButton.addEventListener("click", resetReps);
  elements.modelSwitchButton.addEventListener("click", () => {
    const nextModel = appState.currentModel === "lite" ? "heavy" : "lite";
    void switchPoseModel(nextModel);
  });

  window.addEventListener("resize", resizeCanvasToVideo);
  window.addEventListener("beforeunload", releaseResources);

  updateModelUi();
  setStatus("Ready. Press Start camera.");
});

async function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }

  try {
    await navigator.serviceWorker.register("./service-worker.js");
  } catch (error) {
    console.warn("Service worker registration failed:", error);
  }
}

async function startCameraTest() {
  if (appState.isRunning || appState.isSwitchingModel) {
    return;
  }

  try {
    elements.startButton.disabled = true;

    if (!appState.poseLandmarker) {
      setStatus(`Loading ${MODEL_LABELS[appState.currentModel]} model…`);
      appState.poseLandmarker = await createPoseLandmarker(
        appState.currentModel
      );
    }

    setStatus("Opening camera…");
    appState.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 }
      }
    });

    elements.video.srcObject = appState.stream;
    await elements.video.play();

    resizeCanvasToVideo();
    resetFpsWindow();

    appState.isRunning = true;
    elements.stopButton.disabled = false;
    setStatus(
      `Running ${MODEL_LABELS[appState.currentModel]} pose detection locally.`
    );

    appState.animationFrameId = requestAnimationFrame(processFrameLoop);
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${error.message}`);
    elements.startButton.disabled = false;
    elements.stopButton.disabled = true;
  }
}

async function createPoseLandmarker(modelName) {
  const modelPath = MODEL_PATHS[modelName];

  if (!modelPath) {
    throw new Error(`Unknown model: ${modelName}`);
  }

  if (!appState.visionFileset) {
    appState.visionFileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  }

  try {
    return await PoseLandmarker.createFromOptions(
      appState.visionFileset,
      {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        numPoses: 1,
        outputSegmentationMasks: false,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      }
    );
  } catch (error) {
    throw new Error(
      `Could not load the ${MODEL_LABELS[modelName]} model from ${modelPath}. ` +
      `Confirm that the model file exists and is included in the offline cache. ` +
      `Original error: ${error.message}`
    );
  }
}

async function switchPoseModel(nextModel) {
  if (
    appState.isSwitchingModel ||
    nextModel === appState.currentModel ||
    !MODEL_PATHS[nextModel]
  ) {
    return;
  }

  const previousModel = appState.currentModel;
  const oldLandmarker = appState.poseLandmarker;

  appState.isSwitchingModel = true;
  appState.poseLandmarker = null;
  elements.modelSwitchButton.disabled = true;
  elements.startButton.disabled = true;
  elements.currentModel.textContent = `Loading ${MODEL_LABELS[nextModel]}…`;
  setStatus(`Switching to ${MODEL_LABELS[nextModel]} model…`);

  try {
    oldLandmarker?.close?.();

    appState.poseLandmarker = await createPoseLandmarker(nextModel);
    appState.currentModel = nextModel;
    appState.lastInferenceAt = 0;
    resetFpsWindow();

    setStatus(
      appState.isRunning
        ? `Running ${MODEL_LABELS[nextModel]} pose detection locally.`
        : `${MODEL_LABELS[nextModel]} model ready. Press Start camera.`
    );
  } catch (error) {
    console.error("Failed to switch pose model:", error);
    setStatus(`Model switch failed: ${error.message}`);

    try {
      appState.poseLandmarker = await createPoseLandmarker(previousModel);
      appState.currentModel = previousModel;
      setStatus(
        `Restored ${MODEL_LABELS[previousModel]} model after switch failure.`
      );
    } catch (restoreError) {
      console.error("Failed to restore previous model:", restoreError);
      appState.poseLandmarker = null;
      setStatus("No pose model is currently available.");
    }
  } finally {
    appState.isSwitchingModel = false;
    elements.modelSwitchButton.disabled = false;
    elements.startButton.disabled = appState.isRunning;
    updateModelUi();
  }
}

function processFrameLoop(nowMs) {
  if (!appState.isRunning) {
    return;
  }

  if (
    !appState.isSwitchingModel &&
    appState.poseLandmarker &&
    elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    nowMs - appState.lastInferenceAt >= INFERENCE_INTERVAL_MS
  ) {
    appState.lastInferenceAt = nowMs;
    runPoseDetection(nowMs);
  }

  appState.animationFrameId = requestAnimationFrame(processFrameLoop);
}

function runPoseDetection(timestampMs) {
  if (!appState.poseLandmarker || appState.isSwitchingModel) {
    return;
  }

  try {
    const startedAt = performance.now();
    const result = appState.poseLandmarker.detectForVideo(
      elements.video,
      timestampMs
    );
    const latencyMs = performance.now() - startedAt;

    elements.latency.textContent = `${latencyMs.toFixed(1)} ms`;
    updateFps();
    handlePoseResult(result);
  } catch (error) {
    console.error(error);
    setStatus(`Detection error: ${error.message}`);
  }
}

function handlePoseResult(result) {
  clearCanvas();

  const landmarks = result.landmarks?.[0];
  if (!landmarks) {
    elements.poseState.textContent = "not found";
    elements.leftKnee.textContent = "—";
    elements.rightKnee.textContent = "—";
    return;
  }

  elements.poseState.textContent = "found";
  drawPose(landmarks);

  const leftKnee = angleDegrees(
    landmarks[IDX.leftHip],
    landmarks[IDX.leftKnee],
    landmarks[IDX.leftAnkle]
  );

  const rightKnee = angleDegrees(
    landmarks[IDX.rightHip],
    landmarks[IDX.rightKnee],
    landmarks[IDX.rightAnkle]
  );

  elements.leftKnee.textContent = formatAngle(leftKnee);
  elements.rightKnee.textContent = formatAngle(rightKnee);

  updateSimpleSquatCounter(leftKnee, rightKnee);
}

function updateSimpleSquatCounter(leftKnee, rightKnee) {
  const validAngles = [leftKnee, rightKnee].filter(
    angle => angle !== null
  );

  if (validAngles.length === 0) {
    return;
  }

  const averageKnee =
    validAngles.reduce((sum, angle) => sum + angle, 0) /
    validAngles.length;

  if (appState.phase === "standing" && averageKnee < 125) {
    appState.phase = "down";
  } else if (appState.phase === "down" && averageKnee > 160) {
    appState.phase = "standing";
    appState.repCount += 1;
  }

  elements.phase.textContent = appState.phase;
  elements.repCount.textContent = String(appState.repCount);
}

function angleDegrees(a, b, c) {
  if (!isVisible(a) || !isVisible(b) || !isVisible(c)) {
    return null;
  }

  const baX = a.x - b.x;
  const baY = a.y - b.y;
  const bcX = c.x - b.x;
  const bcY = c.y - b.y;

  const dot = baX * bcX + baY * bcY;
  const magBA = Math.hypot(baX, baY);
  const magBC = Math.hypot(bcX, bcY);

  if (magBA === 0 || magBC === 0) {
    return null;
  }

  const cosine = clamp(dot / (magBA * magBC), -1, 1);
  return Math.acos(cosine) * 180 / Math.PI;
}

function isVisible(landmark) {
  if (!landmark) {
    return false;
  }

  const visibility = landmark.visibility ?? landmark.presence ?? 1;
  return visibility >= 0.5;
}

function drawPose(landmarks) {
  const width = elements.canvas.width;
  const height = elements.canvas.height;

  canvasContext.lineWidth = 4;
  canvasContext.strokeStyle = "#00e676";
  canvasContext.fillStyle = "#00e676";

  for (const [startIndex, endIndex] of POSE_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];

    if (!isVisible(start) || !isVisible(end)) {
      continue;
    }

    canvasContext.beginPath();
    canvasContext.moveTo(start.x * width, start.y * height);
    canvasContext.lineTo(end.x * width, end.y * height);
    canvasContext.stroke();
  }

  for (const landmark of landmarks) {
    if (!isVisible(landmark)) {
      continue;
    }

    canvasContext.beginPath();
    canvasContext.arc(
      landmark.x * width,
      landmark.y * height,
      5,
      0,
      2 * Math.PI
    );
    canvasContext.fill();
  }
}

function resizeCanvasToVideo() {
  const videoWidth = elements.video.videoWidth || 640;
  const videoHeight = elements.video.videoHeight || 480;

  elements.canvas.width = videoWidth;
  elements.canvas.height = videoHeight;
}

function clearCanvas() {
  canvasContext.clearRect(
    0,
    0,
    elements.canvas.width,
    elements.canvas.height
  );
}

function updateFps() {
  appState.processedFrames += 1;
  const now = performance.now();
  const elapsed = now - appState.fpsWindowStartedAt;

  if (elapsed >= 1000) {
    const fps = appState.processedFrames * 1000 / elapsed;
    elements.fps.textContent = fps.toFixed(1);
    appState.processedFrames = 0;
    appState.fpsWindowStartedAt = now;
  }
}

function resetFpsWindow() {
  appState.processedFrames = 0;
  appState.fpsWindowStartedAt = performance.now();
  elements.fps.textContent = "—";
  elements.latency.textContent = "—";
}

function resetReps() {
  appState.repCount = 0;
  appState.phase = "standing";
  elements.repCount.textContent = "0";
  elements.phase.textContent = "standing";
}

function stopCameraTest() {
  appState.isRunning = false;

  if (appState.animationFrameId !== null) {
    cancelAnimationFrame(appState.animationFrameId);
    appState.animationFrameId = null;
  }

  if (appState.stream) {
    for (const track of appState.stream.getTracks()) {
      track.stop();
    }
    appState.stream = null;
  }

  elements.video.srcObject = null;
  clearCanvas();
  elements.poseState.textContent = "—";
  elements.leftKnee.textContent = "—";
  elements.rightKnee.textContent = "—";
  elements.startButton.disabled = false;
  elements.stopButton.disabled = true;
  setStatus("Stopped.");
}

function releaseResources() {
  stopCameraTest();
  appState.poseLandmarker?.close?.();
  appState.poseLandmarker = null;
}

function updateModelUi() {
  const currentLabel = MODEL_LABELS[appState.currentModel];
  const nextModel = appState.currentModel === "lite" ? "heavy" : "lite";

  elements.currentModel.textContent = currentLabel;
  elements.modelSwitchButton.textContent =
    `Switch to ${MODEL_LABELS[nextModel]}`;
}

function formatAngle(angle) {
  return angle === null ? "—" : `${angle.toFixed(1)}°`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function setStatus(message) {
  elements.status.textContent = message;
}
