import type { Tour } from './tours';

interface TourControlsProps {
  tours: Tour[];
  activeTourId: string | null;
  /** Current narration line (lives outside this component so the viewer
   *  receives `onKeyframe` from CameraDirector and forwards a string). */
  narration: string | null;
  /** Seconds elapsed in the current tour; 0 if none active. */
  elapsedSec: number;
  onStart: (id: string) => void;
  onStop: () => void;
}

function formatProgress(elapsed: number, duration: number) {
  const pct = Math.min(100, (elapsed / duration) * 100);
  return { pct, label: `${elapsed.toFixed(1)} / ${duration.toFixed(0)} s` };
}

export function TourControls({ tours, activeTourId, narration, elapsedSec, onStart, onStop }: TourControlsProps) {
  const active = tours.find((t) => t.id === activeTourId) ?? null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-wide opacity-70">Guided tours</span>
        {active && (
          <button
            type="button"
            onClick={onStop}
            className="text-[0.7rem] underline opacity-70 hover:opacity-100"
          >
            stop
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {tours.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => onStart(t.id)}
            disabled={active?.id === t.id}
            className={
              'text-[0.72rem] px-2 py-1 rounded border transition-colors ' +
              (active?.id === t.id
                ? 'bg-amber-500/85 text-black border-amber-400 font-semibold'
                : 'bg-black/35 border-white/15 hover:bg-black/55')
            }
          >
            {t.title}
          </button>
        ))}
      </div>
      {active && (
        <>
          <div className="h-1 rounded-full bg-white/10 overflow-hidden">
            <div
              className="h-full bg-amber-400"
              style={{ width: `${formatProgress(elapsedSec, active.duration).pct}%` }}
            />
          </div>
          <div className="text-[0.7rem] opacity-65 font-mono">
            {formatProgress(elapsedSec, active.duration).label}
          </div>
          {narration && (
            <div className="text-[0.78rem] leading-snug bg-black/35 rounded p-2 border border-white/8">
              {narration}
            </div>
          )}
        </>
      )}
    </div>
  );
}
