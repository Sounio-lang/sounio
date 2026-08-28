// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import githubDarkTheme from '@shikijs/themes/github-dark';

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

// Sounio-specific token colors, bound to the design-system tokens committed
// in src/styles/global.css (#1764 "Epistemic Instrumentarium") — e.g. the
// Mut effect color for both `&!` and `var`, the linear/affine ownership
// color, and one dedicated color per algebraic effect name (IO, Mut, Div,
// Panic, Alloc, Async, GPU, Prob, Observe). See src/shiki/sounio-theme.json
// for the full scope->token mapping; every foreground value there is a
// var() reference, no color is invented in this codebase for this purpose.
const sounioTokenColors = JSON.parse(
  readFileSync(fileURLToPath(new URL('./src/shiki/sounio-theme.json', import.meta.url)), 'utf-8')
).tokenColors;

// Base theme stays github-dark for every language except `sio` (zero visual
// change to existing JS/Python/bash/etc. code blocks). Two additions apply
// site-wide: the sio-specific tokenColors appended above, and the editor
// background/foreground rebound to design-system vars so code blocks follow
// the site's light/dark/system theme contract instead of being pinned dark
// (previously Shiki's inline style always won over the `pre {}` rule in
// global.css, so every code block ignored the light-mode toggle).
const sounioSiteTheme = {
  ...githubDarkTheme,
  colors: {
    ...githubDarkTheme.colors,
    'editor.background': 'var(--color-code-bg)',
    'editor.foreground': 'var(--color-text-primary)',
  },
  tokenColors: [...(githubDarkTheme.tokenColors ?? []), ...sounioTokenColors],
};

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
        theme: sounioSiteTheme,
        langs: [sounioGrammar],
      },
    }),
  ],

  markdown: {
    syntaxHighlight: 'shiki',
    shikiConfig: {
      theme: sounioSiteTheme,
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
