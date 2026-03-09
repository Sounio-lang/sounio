const localePrefixRe = /^\/(?:en|pt|el|zh|ja|es)(?=\/|$)/;

export function isDocsPath(pathname: string): boolean {
  const normalized = pathname.replace(localePrefixRe, '') || '/';
  return normalized === '/learn' || normalized.startsWith('/learn/');
}
