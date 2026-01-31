import { defineCollection, z } from 'astro:content';

// Supported locales
const locales = ['en', 'pt', 'el', 'zh', 'ja', 'es'] as const;

// Documentation collection
const docsCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    weight: z.number().default(0),
    draft: z.boolean().default(false),
    category: z.string().optional(),
  }),
});

// Showcases collection (domain applications)
const showcasesCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    domain: z.string(),
    icon: z.string().optional(),
    color: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

// Tutorials collection (interactive learning)
const tutorialsCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    series: z.string(), // e.g., "getting-started", "epistemic-computing"
    order: z.number(),
    estimatedMinutes: z.number().optional(),
    prerequisites: z.array(z.string()).optional(),
    draft: z.boolean().default(false),
  }),
});

// Blog collection
const blogCollection = defineCollection({
  type: 'content',
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
  }),
});

// Package registry collection
const registryCollection = defineCollection({
  type: 'data',
  schema: z.object({
    packages: z.array(
      z.object({
        name: z.string(),
        version: z.string(),
        description: z.string(),
        repository: z.string().url().optional(),
        documentation: z.string().url().optional(),
        keywords: z.array(z.string()).default([]),
        author: z.string().optional(),
        license: z.string().default('MIT'),
        downloads: z.number().optional(),
        featured: z.boolean().default(false),
      })
    ),
  }),
});

export const collections = {
  docs: docsCollection,
  showcases: showcasesCollection,
  tutorials: tutorialsCollection,
  blog: blogCollection,
  registry: registryCollection,
};
