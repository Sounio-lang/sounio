/**
 * Compare Astro collection entry ids after Content Layer migration.
 * Glob loaders often omit `.md`/`.mdx`; legacy routes still construct candidates with extensions.
 */
export function contentIdsMatch(a: string, b: string): boolean {
  const strip = (s: string) => s.replace(/\.mdx$/i, '').replace(/\.md$/i, '');
  return strip(a) === strip(b);
}
