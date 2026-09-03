import type { ReactNode } from 'react';

export function Badge({
  children,
  variant = 'default',
  className,
}: {
  children: ReactNode;
  variant?: 'default' | 'warning';
  className?: string;
}) {
  const c =
    variant === 'warning' ? 'bg-yellow-100 text-yellow-900' : 'bg-gray-100 text-gray-800';
  return <span className={`inline-flex px-2 py-1 rounded text-sm ${c} ${className ?? ''}`}>{children}</span>;
}
