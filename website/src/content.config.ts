import { defineCollection } from 'astro:content';
import { file, glob } from 'astro/loaders';
import { z } from 'astro/zod';

const registryPackageSchema = z.object({
  name: z.string(),
  version: z.string(),
  description: z.string(),
  repository: z.optional(z.url()),
  documentation: z.optional(z.url()),
  keywords: z.array(z.string()).default([]),
  author: z.string().optional(),
  license: z.string().default('MIT'),
  downloads: z.number().optional(),
  featured: z.boolean().default(false),
});

const docs = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/docs' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    weight: z.number().default(0),
    draft: z.boolean().default(false),
    category: z.string().optional(),
    topic_id: z.string(),
    authority: z.string(),
    audience: z.string(),
    last_validated: z.string(),
    validated_by: z.string(),
    source_of_truth: z.string(),
  }),
});

const showcases = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/showcases' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    domain: z.string(),
    icon: z.string().optional(),
    color: z.string().optional(),
    draft: z.boolean().default(false),
    topic_id: z.string(),
    authority: z.string(),
    audience: z.string(),
    last_validated: z.string(),
    validated_by: z.string(),
    source_of_truth: z.string(),
  }),
});

const tutorials = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/tutorials' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    series: z.string(),
    order: z.number(),
    estimatedMinutes: z.number().optional(),
    prerequisites: z.array(z.string()).optional(),
    draft: z.boolean().default(false),
    topic_id: z.string(),
    authority: z.string(),
    audience: z.string(),
    last_validated: z.string(),
    validated_by: z.string(),
    source_of_truth: z.string(),
  }),
});

const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    publishedAt: z.coerce.date(),
    updatedAt: z.coerce.date().optional(),
    author: z.string().default('Sounio Team'),
    tags: z.array(z.string()).default([]),
    featured: z.boolean().default(false),
    draft: z.boolean().default(false),
    image: z.string().optional(),
    topic_id: z.string(),
    authority: z.string(),
    audience: z.string(),
    last_validated: z.string(),
    validated_by: z.string(),
    source_of_truth: z.string(),
  }),
});

const registry = defineCollection({
  loader: file('src/content/registry/packages.json', {
    parser: (text: string) => {
      const parsed = JSON.parse(text) as { packages: unknown[] };
      return [{ id: 'packages', packages: parsed.packages }];
    },
  }),
  schema: z.object({
    packages: z.array(registryPackageSchema),
  }),
});

export const collections = { docs, showcases, tutorials, blog, registry };
