import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  return {
    plugins: [vue()],
    base: '/',
    define: {
      __API_BASE_URL__: JSON.stringify(env.VITE_API_BASE_URL),
    }
  };
})
