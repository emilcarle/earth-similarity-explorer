import { defineConfig } from "vite";

export default defineConfig({
  server: {
    // The dev server forwards /api/... to the local search API, so the browser
    // sees one address and needs no cross-origin permission (CORS)
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
