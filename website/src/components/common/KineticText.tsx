import type { CSSProperties } from 'react';

interface Props {
  text: string;
  className?: string;
}

export function KineticText({ text, className = 'section-heading' }: Props) {
  const letters = Array.from(text);

  return (
    <h2 className={`${className} kinetic-text flex flex-wrap`} aria-label={text}>
      {letters.map((letter, index) => (
        <span
          key={index}
          className="kinetic-text-letter inline-block whitespace-pre"
          aria-hidden="true"
          style={{ ['--kinetic-index' as '--kinetic-index']: index } as CSSProperties}
        >
          {letter === ' ' ? '\u00A0' : letter}
        </span>
      ))}
    </h2>
  );
}
