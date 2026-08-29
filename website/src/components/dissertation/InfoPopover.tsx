import { useEffect, useRef, useState } from 'react';

interface InfoPopoverProps {
  /** Body content rendered in the popover. Plain text is fine. */
  children: React.ReactNode;
  /** Optional aria label for the trigger button. */
  ariaLabel?: string;
}

/**
 * Tiny info-circle trigger with a click-to-open tooltip. Designed for the
 * dissertation viewer's pharmacologist audience — copy lives at the call
 * site, this just handles positioning + dismissal.
 */
export function InfoPopover({ children, ariaLabel = 'More information' }: InfoPopoverProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-label={ariaLabel}
        className="w-4 h-4 rounded-full bg-white/15 text-[0.65rem] font-bold text-white/85 leading-4 text-center hover:bg-white/25"
      >
        i
      </button>
      {open && (
        <div
          role="tooltip"
          className="absolute right-0 mt-1.5 z-30 w-64 text-[0.75rem] leading-snug p-2.5 rounded bg-[#13192b] border border-white/15 shadow-xl text-white"
        >
          {children}
        </div>
      )}
    </div>
  );
}
