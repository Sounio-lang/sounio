import { claim, format } from '../evidence/claim';
import type { Level } from '../evidence/types';

const GLYPH: Record<Level, string> = {
  verified: '◈', unbounded: '△', refused: '⊘',
};
// Um número vindo de registro não recebe glifo de veredito — ele não tem um.
const RECORD_GLYPH = '·';

/**
 * Um número com a sua procedência presa a ele.
 *
 * Não existe prop `value`. O valor vem do artefato; se o artefato ou a
 * métrica não existirem, `claim()` lança e o build para. É esta ausência de
 * prop que faz o site não conseguir inventar um número.
 */
export function Claim({ of, metric }: { of: string; metric: string }) {
  const c = claim(of, metric);
  return (
    <span className="claim" data-level={c.level ?? 'record'}
          title={`${c.artifact} · ${c.metric}${c.reason ? ` · ${c.reason}` : ''}`}>
      <span className="claim-glyph" aria-hidden="true">
        {c.level ? GLYPH[c.level] : RECORD_GLYPH}
      </span>
      <span className="claim-value">{format(c.value)}</span>
    </span>
  );
}
