// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';
import solidJs from '@astrojs/solid-js';
import tailwindcss from '@tailwindcss/vite';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkSioAsRust from './src/remark/remark-sio-as-rust.mjs';

// https://astro.build/config
export default defineConfig({
  site: 'https://www.souniolang.org',

  integrations: [
    react({ include: ['**/components/common/**', '**/components/docs/**', '**/components/home/CodeExamples.tsx', '**/components/playground/**', '**/components/science-simulators/**', '**/components/about/**'] }),
    solidJs({ include: ['**/components/home/MascotHeroWebGL.tsx'] }),
    sitemap({
      i18n: {
        defaultLocale: 'en',
        locales: {
          en: 'en',
          pt: 'pt-BR',
          el: 'el',
          zh: 'zh-CN',
          ja: 'ja',
          es: 'es',
        },
      },
    }),
    mdx({
      remarkPlugins: [remarkSioAsRust],
      syntaxHighlight: 'shiki',
      shikiConfig: {
        theme: 'github-dark',
      },
    }),
  ],

  markdown: {
    remarkPlugins: [remarkSioAsRust],
    syntaxHighlight: 'shiki',
    shikiConfig: {
      theme: 'github-dark',
    },
  },

  vite: {
    plugins: [tailwindcss()],
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'pt', 'el', 'zh', 'ja', 'es'],
    routing: {
      prefixDefaultLocale: false,
    },
  },

  // Enable view transitions for smooth page navigation
  prefetch: true,

  // /platform permanently redirects to /language#compiler.
  // Hash fragments are client-side only; the 308 targets /language.
  // The stub page at platform.astro adds the #compiler fragment via JS.
  redirects: {
    '/platform': { status: 308, destination: '/language' },
    '/pt/platform': { status: 308, destination: '/pt/language' },
    '/el/platform': { status: 308, destination: '/el/language' },
    '/zh/platform': { status: 308, destination: '/zh/language' },
    '/ja/platform': { status: 308, destination: '/ja/language' },
    '/es/platform': { status: 308, destination: '/es/language' },
  },
});
