// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import remarkSioAsRust from './src/remark/remark-sio-as-rust.mjs';

// https://astro.build/config
export default defineConfig({
  site: 'https://www.souniolang.org',

  // Astro v7 defaults to 'jsx' whitespace stripping; keep v6 HTML-aware
  // compression for behavior parity during the redesign lane.
  compressHTML: true,

  integrations: [
    react(),
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
          'zh-hk': 'zh-HK',
        },
      },
    }),
    // MDX inherits markdown.processor (remark-sio-as-rust + shiki) from the
    // top-level markdown config.
    mdx(),
  ],

  markdown: {
    // Astro v7 defaults to the Sätteri processor; stay on the unified
    // remark/rehype pipeline so remark-sio-as-rust keeps working.
    processor: unified({
      remarkPlugins: [remarkSioAsRust],
    }),
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
    locales: ['en', 'pt', 'el', 'zh', 'ja', 'es', 'zh-hk'],
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
    '/zh-hk/platform': { status: 308, destination: '/zh-hk/language' },
  },
});
