import { useState } from 'react';
import { Interval } from '../components/Interval';
import { Claim } from '../components/Claim';
import { CorpusField, CORPUS } from '../components/Corpus';
import { EVIDENCE_STATS, territories, measuredOn } from '../evidence/claim';
import type { Artifact, Kind } from '../evidence/types';
import './proof.css';

const VERDICT = { verified: 'passes', unbounded: 'no measured bound', refused: 'refused' } as const;
const MARK = { verified: '◈', unbounded: '△', refused: '⊘' } as const;
const PREVIEW = 8;
const P = EVIDENCE_STATS.proportions;

function Row({ a }: { a: Artifact }) {
  const date = measuredOn(a);
  const n = Object.keys(a.metrics).length;

  return (
    <li className="row" data-kind={a.kind} data-level={a.level ?? undefined}>
      <span className="row-mark">{a.level ? MARK[a.level] : '·'}</span>
      <span className="row-id">{a.id}</span>
      <span className="row-verdict">
        {a.level ? VERDICT[a.level] : 'measurement · no verdict'}
      </span>
      <span className="row-meta">
        {n > 0 && <span>{n}m</span>}
        {date && <span>{date}</span>}
      </span>
      {a.reason && <span className="row-reason">{a.reason}</span>}
      {a.proportion && <span className="row-iv"><Interval a={a} showId={false} /></span>}
    </li>
  );
}

function Territory({ name, artifacts, gates, records, refused, filter }: {
  name: string; artifacts: Artifact[]; gates: number; records: number;
  refused: number; filter: Kind | 'all';
}) {
  const [open, setOpen] = useState(false);
  const shown = filter === 'all' ? artifacts : artifacts.filter(a => a.kind === filter);
  if (!shown.length) return null;

  return (
    <section className="territory">
      <header className="territory-head">
        <h3>{name}</h3>
        <span className="territory-counts">
          {gates > 0 && <span>{gates} gates</span>}
          {records > 0 && <span>{records} records</span>}
          {refused > 0 && <span className="is-refused">{refused} refused</span>}
        </span>
      </header>
      <ul className="rows">
        {(open ? shown : shown.slice(0, PREVIEW)).map(a => <Row key={a.id} a={a} />)}
      </ul>
      {shown.length > PREVIEW && (
        <button className="more" type="button" onClick={() => setOpen(o => !o)}>
          {open ? '— fewer' : `+ ${shown.length - PREVIEW} more`}
        </button>
      )}
    </section>
  );
}

export function Proof() {
  const [filter, setFilter] = useState<Kind | 'all'>('all');

  return (
    <div className="wrap">
      <section className="band" style={{ borderTop: 0 }}>
        <p className="kicker">the corpus · {EVIDENCE_STATS.artifacts} artifacts · unedited</p>
        <h1>What can be verified,<br /><em>and what merely happened.</em></h1>
        <p className="lede">
          Every claim on this site resolves to a file in <b>artifacts/</b>. The corpus holds
          two different things, and conflating them would be the first lie.
        </p>
        <p className="lede">
          They were produced by a body of code this size — <b><Claim of={CORPUS} metric="written.lines" /> written
          lines</b> across <Claim of={CORPUS} metric="written.files" /> files, of which{' '}
          <Claim of={CORPUS} metric="compiler.lines" /> are the compiler itself. That
          measurement is in this corpus too, as a record:
        </p>

        <CorpusField />

        <p className="lede">
          <b>Gates</b> declare a verdict — it passes, it fails, or its bound was never
          measured. <b>Records</b> carry measurement, research result and provenance. They
          have no verdict because they are not gates: a seed-refresh receipt has nothing to
          pass or fail.
        </p>

        <div className="readout">
          <div><b>{EVIDENCE_STATS.byKind.gate}</b><span>gates</span></div>
          <div><b>{EVIDENCE_STATS.byKind.record}</b><span>records</span></div>
          <div><b>{EVIDENCE_STATS.byLevel.refused}</b><span>refusals</span></div>
          <div><b>{P.count}</b><span>with an interval</span></div>
          <div><b>{P.widest.toFixed(3)}</b><span>widest interval</span></div>
          <div><b>{P.narrowest.toFixed(3)}</b><span>narrowest</span></div>
        </div>

        <div className="controls">
          <div className="filters" role="group" aria-label="Filter by species">
            {(['all', 'gate', 'record'] as const).map(k => (
              <button key={k} type="button" className="filter"
                      aria-pressed={filter === k} onClick={() => setFilter(k)}>
                {k === 'all' ? 'everything' : k === 'gate' ? 'gates' : 'records'}
              </button>
            ))}
          </div>
          <p className="note" style={{ margin: 0 }}>
            Grouped by territory. Within each, refusals first. Where a gate reports
            passed/total, its Wilson 95% interval is drawn.
          </p>
        </div>
      </section>

      <div className="territories">
        {territories().map(t => <Territory key={t.name} {...t} filter={filter} />)}
      </div>
    </div>
  );
}
