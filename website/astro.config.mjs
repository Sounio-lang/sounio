// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import mdx from '@astrojs/mdx';

// https://astro.build/config
export default defineConfig({
  site: 'https://souniolang.org',

  integrations: [
    react(),
    mdx({
      syntaxHighlight: 'shiki',
      shikiConfig: {
        theme: 'github-dark',
      },
    }),
  ],

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
