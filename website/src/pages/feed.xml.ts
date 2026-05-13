import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { getCollection } from 'astro:content';

export async function GET(context: APIContext) {
  const blog = await getCollection('blog');

  return rss({
    title: 'Sounio Language',
    description: 'Research updates, release notes, and engineering writing from the Sounio project.',
    site: context.site!,
    items: blog.map((post) => ({
      title: post.data.title,
      pubDate: post.data.publishedAt,
      description: post.data.description,
      link: `/insights/${post.id.replace(/\.mdx$/, '')}/`,
    })),
    customData: '<language>en</language>',
  });
}
