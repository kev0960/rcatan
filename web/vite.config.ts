import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/games": "http://127.0.0.1:8000",
      "/checkpoints": "http://127.0.0.1:8000",
    },
  },
});
