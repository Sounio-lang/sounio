import type { ButtonHTMLAttributes, ReactNode } from 'react';

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'outline';
  children: ReactNode;
};

export function Button({ variant = 'primary', className = '', type = 'button', ...rest }: ButtonProps) {
  const base = 'px-4 py-2 rounded font-medium transition-colors disabled:opacity-50';
  const styles =
    variant === 'secondary'
      ? 'bg-gray-200 text-gray-900 hover:bg-gray-300'
      : variant === 'outline'
        ? 'border border-gray-300 bg-white hover:bg-gray-50'
        : 'bg-blue-600 text-white hover:bg-blue-700';
  return <button type={type} className={`${base} ${styles} ${className}`} {...rest} />;
}
