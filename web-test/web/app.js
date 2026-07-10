import {
  FilesetResolver,
  PoseLandmarker
} from "./vendor/mediapipe/vision_bundle.mjs";

const MODEL_PATH = "./models/pose_landmarker_lite.task";
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
  poseLandmarker: null,
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
  setStatus("Ready. Press Start camera.");

  elements.startButton.addEventListener("click", startCameraTest);
  elements.stopButton.addEventListener("click", stopCameraTest);
  elements.resetButton.addEventListener("click", resetReps);
  window.addEventListener("resize", resizeCanvasToVideo);
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
  try {
    elements.startButton.disabled = true;
    setStatus("Loading pose model…");

    if (!appState.poseLandmarker) {
      await checkModelExists();
      appState.poseLandmarker = await createPoseLandmarker();
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

    appState.isRunning = true;
    elements.stopButton.disabled = false;
    setStatus("Running pose detection locally.");
    appState.animationFrameId = requestAnimationFrame(processFrameLoop);
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${error.message}`);
    elements.startButton.disabled = false;
    elements.stopButton.disabled = true;
  }
}

async function checkModelExists() {
  const response = await fetch(MODEL_PATH, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(
      `Model not found at ${MODEL_PATH}. Copy pose_landmarker_lite.task into web/models/.`
    );
  }
}

async function createPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_PATH,
      delegate: "CPU"
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false
  });
}

function processFrameLoop(nowMs) {
  if (!appState.isRunning) {
    return;
  }

  if (
    elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    nowMs - appState.lastInferenceAt >= INFERENCE_INTERVAL_MS
  ) {
    appState.lastInferenceAt = nowMs;
    runPoseDetection(nowMs);
  }

  appState.animationFrameId = requestAnimationFrame(processFrameLoop);
}

function runPoseDetection(timestampMs) {
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
  const validAngles = [leftKnee, rightKnee].filter(angle => angle !== null);
  if (validAngles.length === 0) {
    return;
  }

  const averageKnee = validAngles.reduce((sum, angle) => sum + angle, 0) / validAngles.length;

  // Extremely simple test logic:
  // standing: knee relatively straight
  // down: knee flexed
  // one rep: user goes down, then comes back up
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
  canvasContext.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
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
  elements.startButton.disabled = false;
  elements.stopButton.disabled = true;
  setStatus("Stopped.");
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
