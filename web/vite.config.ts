import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Listen on every interface so other devices (e.g. over Tailscale) can
    // open the UI, not just this machine.
    host: true,
    // Vite rejects requests whose Host header isn't localhost or an IP.
    // Allow Tailscale MagicDNS names (<machine>.<tailnet>.ts.net).
    allowedHosts: [".ts.net"],
    proxy: {
      "/games": "http://127.0.0.1:8000",
      "/checkpoints": "http://127.0.0.1:8000",
      "/training": "http://127.0.0.1:8000",
    },
  },
});
