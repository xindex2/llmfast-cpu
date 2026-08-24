import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// Two entry points out of one project: index.html is the operator console served at
// /admin/ui, app.html is the customer console served at /. They share the design tokens,
// the API client and the playground, so a fix to either lands in both.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'),
        app: resolve(__dirname, 'app.html'),
      },
    },
  },
  server: {
    proxy: {
      '/v1': 'http://localhost:8080',
      '/admin': 'http://localhost:8080',
      '/auth': 'http://localhost:8080',
      '/account': 'http://localhost:8080',
    },
  },
})
