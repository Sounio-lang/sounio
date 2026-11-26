The website moves to Astro 7 and @astrojs/mdx 8. Astro 7's default Markdown
processor no longer runs remark/rehype plugins, so `@astrojs/markdown-remark`
is now a direct dependency and the existing `rehypeDemoteH1` keeps running.
