// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkSioAsRust from './src/remark/remark-sio-as-rust.mjs';

// https://astro.build/config
export default defineConfig({
  site: 'https://www.souniolang.org',

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
});
