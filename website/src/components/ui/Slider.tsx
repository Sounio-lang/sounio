export function Slider({
  value,
  onChange,
  min,
  max,
  step,
  className,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  className?: string;
}) {
  return (
    <input
      type="range"
      className={`w-full ${className ?? ''}`}
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}
