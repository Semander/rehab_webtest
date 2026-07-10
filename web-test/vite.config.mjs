import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  root: "web",
  plugins: [basicSsl()],
  server: {
    host: "0.0.0.0",
    https: true
  },
  preview: {
    host: "0.0.0.0",
    https: true
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true
  }
});
