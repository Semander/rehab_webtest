import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const projectRoot = path.resolve(path.dirname(__filename), "..");
const packageRoot = path.join(projectRoot, "node_modules", "@mediapipe", "tasks-vision");
const webVendorRoot = path.join(projectRoot, "web", "vendor", "mediapipe");

const sourceBundle = path.join(packageRoot, "vision_bundle.mjs");
const sourceWasm = path.join(packageRoot, "wasm");
const targetBundle = path.join(webVendorRoot, "vision_bundle.mjs");
const targetWasm = path.join(webVendorRoot, "wasm");

if (!fs.existsSync(sourceBundle) || !fs.existsSync(sourceWasm)) {
  console.error("MediaPipe package files were not found.");
  console.error("Run: npm install");
  process.exit(1);
}

fs.mkdirSync(webVendorRoot, { recursive: true });
fs.copyFileSync(sourceBundle, targetBundle);
fs.rmSync(targetWasm, { recursive: true, force: true });
fs.cpSync(sourceWasm, targetWasm, { recursive: true });

console.log("Copied MediaPipe web files into web/vendor/mediapipe/.");
