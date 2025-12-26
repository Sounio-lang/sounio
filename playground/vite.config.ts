import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
    plugins: [wasm()],
    base: './',
    root: '.',
    publicDir: 'public',
    build: {
        outDir: 'dist',
        emptyOutDir: true,
        target: 'es2022',
        sourcemap: true,
        rollupOptions: {
            input: {
                main: './index.html',
            },
            output: {
                entryFileNames: '[name].js',
                chunkFileNames: '[name]-[hash].js',
                assetFileNames: '[name]-[hash][extname]',
            },
        },
    },
    server: {
        port: 3000,
        open: true,
        cors: true,
        headers: {
            // Required for SharedArrayBuffer (needed for some WASM features)
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
        },
    },
    preview: {
        port: 3000,
    },
    optimizeDeps: {
        exclude: ['sounio_compiler'],
    },
    resolve: {
        alias: {
            '@': '/src',
        },
    },
});
