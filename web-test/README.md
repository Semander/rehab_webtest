# REHAB-AI Pose Test PWA

This is a minimal offline-capable web app for testing MediaPipe Pose Landmarker on a phone.

It does four things:

1. Opens the phone camera.
2. Runs `pose_landmarker_lite.task` locally in the browser using MediaPipe Tasks Vision.
3. Draws a simple skeleton overlay.
4. Displays knee angles, rough latency/FPS, and a very basic squat rep counter.

No backend is included.

## 1. Copy this folder into your repo

Recommended location:

```text
REHAB-AI/web-test/
```

## 2. Install dependencies

From this folder:

```bash
npm install
```

## 3. Copy the Lite model

From the root of your existing REHAB-AI repo, copy the model into the web app:

```bash
cp models/pretrained/pose_landmarker_lite.task web-test/web/models/pose_landmarker_lite.task
```

If you put this starter somewhere else, the important final path is:

```text
web/models/pose_landmarker_lite.task
```

## 4. Run the app

```bash
npm run dev
```

Vite will print a local HTTPS address. Open that address on the phone.

Camera access on a phone generally requires HTTPS. The included Vite config uses a development SSL certificate. The browser may show a warning the first time.

## 5. Test offline behavior

After the app has loaded once:

1. Press **Start camera** and confirm landmarks appear.
2. Stop the app.
3. Turn on airplane mode.
4. Reopen the installed page or refresh the already loaded page.
5. Press **Start camera** again.

If the service worker cached everything, the app should still load and run.

## 6. Where to add project logic

For now, the exercise logic is inside `web/app.js` in these functions:

```js
angleDegrees(...)
updateSimpleSquatCounter(...)
handlePoseResult(...)
```

For the next step, move these into a separate `pose_analyzer.js` file and keep MediaPipe/camera code separate from rehabilitation scoring.

## Notes

The current rep counter is intentionally simple. It is only for basic functionality testing, not final evaluation.
