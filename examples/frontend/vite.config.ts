import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The backend has no CORS middleware, and it does not need any: in
    // development the browser only ever talks to Vite, which forwards /api to
    // the backend. A deployed frontend wants the same arrangement from a real
    // reverse proxy, so the browser sees one origin either way.
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
