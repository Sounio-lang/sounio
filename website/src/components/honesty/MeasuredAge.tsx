import { useEffect, useState } from 'react';
import { ageBand, ageMs, formatAge } from '../../lib/dissertationHonestyNow';

type Props = {
  measuredAt: string;
};

/**
 * Live age of a dated measurement. The ISO timestamp lives in the
 * parent so a reader without JS still sees when the number was taken.
 * This island only adds "11 h" / "62 d". A June 6/6 would read as
 * days, in the refused token — that is the point of the control.
 */
export function MeasuredAge({ measuredAt }: Props) {
  const [nowMs, setNowMs] = useState<number | null>(null);

  useEffect(() => {
    setNowMs(Date.now());
  }, []);

  if (nowMs === null) {
    return <span className="dnow-age" data-age="pending" hidden />;
  }

  const ms = ageMs(measuredAt, nowMs);
  const formatted = formatAge(ms);
  const band = ageBand(ms);

  if (!formatted || band === 'invalid') {
    return (
      <span className="dnow-age" data-age="invalid">
        {measuredAt}
      </span>
    );
  }

  return (
    <span className="dnow-age" data-age={band}>
      {formatted.value}&nbsp;{formatted.unit}
    </span>
  );
}
