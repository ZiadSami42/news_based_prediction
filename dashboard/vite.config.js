import { defineConfig } from 'vite';

export default defineConfig({
  // Set the base to the repository name for GitHub Pages
  base: '/news_based_prediction/',
  build: {
    outDir: 'dist',
  },
});
