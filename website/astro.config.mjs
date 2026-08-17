// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Real Shiki grammar for Sounio (`sio` code fences). Previously ```sio blocks
// were relabeled as ```rust before highlighting (see git history of
// src/remark/remark-sio-as-rust.mjs) — that tokenized every &!, ++, `var`
// and `with`-clause on the site using Rust's keywords, on a language whose
// own CLAUDE.md opens by listing those as compile errors. This is a
// targeted grammar, not a full port: it covers the tokens that actively
// mislead plus a minimal keyword/comment/string/number baseline. See
// src/shiki/sounio.tmLanguage.json for scope.
const sounioGrammar = JSON.parse(
  readFileSync(fileURLToPath(new URL('./src/shiki/sounio.tmLanguage.json', import.meta.url)), 'utf-8')
);

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
          'zh-hk': 'zh-HK',
        },
      },
    }),
    mdx({
      syntaxHighlight: 'shiki',
      shikiConfig: {
        theme: 'github-dark',
        langs: [sounioGrammar],
      },
    }),
  ],

  markdown: {
    syntaxHighlight: 'shiki',
    shikiConfig: {
      theme: 'github-dark',
      langs: [sounioGrammar],
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
