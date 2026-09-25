import { defineConfig } from 'vite';

/** Build separado: só as ilhas, para montar sobre o HTML gerado pelo Swift. */
export default defineConfig({
  // Em modo lib o Vite não fixa NODE_ENV: sem isto o React de desenvolvimento
  // vai junto, com todos os avisos, e o bundle passa de 900 KB.
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  build: {
    minify: 'esbuild',
    target: 'es2020',
    outDir: '../swift/dist',
    emptyOutDir: false,
    lib: {
      entry: 'src/islands/index.ts',
      formats: ['es'],
      fileName: () => 'islands.js',
    },
    rollupOptions: { output: { assetFileNames: 'islands.[ext]' } },
  },
});
