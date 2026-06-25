import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Build outputs to frontend/dist, which FastAPI serves as static files.
// In dev (`npm run dev`), proxy the API routes to the running uvicorn server.
export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    proxy: {
      '/detect': 'http://127.0.0.1:8000',
      '/outputs': 'http://127.0.0.1:8000',
      '/uploads': 'http://127.0.0.1:8000',
    },
  },
})
