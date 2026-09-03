interface TimeScrubberProps {
  /** Current sim time, hours. */
  timeHours: number;
  /** Maximum sim time (e.g. 30 d = 720 h). */
  maxHours: number;
  /** Whether playback is currently advancing. */
  playing: boolean;
  /** Playback rate multiplier. */
  speed: number;
  /** Logarithmic vs linear time axis. */
  logScale: boolean;
  onSeek: (hours: number) => void;
  onTogglePlay: () => void;
  onSpeedChange: (s: number) => void;
  onToggleScale: () => void;
  onReset: () => void;
}

function formatTime(hours: number) {
  if (hours < 1) return `${(hours * 60).toFixed(1)} min`;
  if (hours < 24) return `${hours.toFixed(1)} h`;
  const days = hours / 24;
  return `${days.toFixed(2)} d`;
}

export function TimeScrubber({
  timeHours,
  maxHours,
  playing,
  speed,
  logScale,
  onSeek,
  onTogglePlay,
  onSpeedChange,
  onToggleScale,
  onReset,
}: TimeScrubberProps) {
  const minLog = Math.log10(0.1);
  const maxLog = Math.log10(maxHours);

  const sliderToHours = (s: number) => {
    if (!logScale) return s;
    return Math.pow(10, minLog + (s / 100) * (maxLog - minLog));
  };
  const hoursToSlider = (h: number) => {
    if (!logScale) return h;
    const safe = Math.max(0.1, h);
    return ((Math.log10(safe) - minLog) / (maxLog - minLog)) * 100;
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-wide opacity-70">
          Time since implant
        </span>
        <span className="font-mono text-sm">{formatTime(timeHours)}</span>
      </div>
      <input
        type="range"
        min={logScale ? 0 : 0}
        max={logScale ? 100 : maxHours}
        step={logScale ? 0.1 : 0.5}
        value={hoursToSlider(timeHours)}
        onChange={(e) => onSeek(sliderToHours(Number(e.target.value)))}
        className="w-full accent-amber-400"
      />
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <button
          type="button"
          onClick={onTogglePlay}
          className="px-2.5 py-1 rounded bg-amber-500/85 text-black font-semibold hover:bg-amber-500"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          type="button"
          onClick={onReset}
          className="px-2.5 py-1 rounded bg-white/10 hover:bg-white/15"
        >
          Restart
        </button>
        <label className="flex items-center gap-1 ml-2 opacity-80">
          <span>Speed</span>
          <input
            type="range"
            min={0.25}
            max={4}
            step={0.25}
            value={speed}
            onChange={(e) => onSpeedChange(Number(e.target.value))}
            className="w-20 accent-amber-300"
          />
          <span className="font-mono w-9 text-right">{speed.toFixed(2)}×</span>
        </label>
        <label className="flex items-center gap-1 ml-2 opacity-80">
          <input type="checkbox" checked={logScale} onChange={onToggleScale} />
          <span>log scale</span>
        </label>
      </div>
    </div>
  );
}
