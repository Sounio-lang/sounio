import { EVIDENCE, EVIDENCE_STATS } from './index.generated';
import { MissingEvidence, type Artifact, type Claim, type Kind, type Level } from './types';

export { EVIDENCE, EVIDENCE_STATS };
export type { Artifact, Claim, Kind, Level };

/**
 * A única porta de entrada para um número no site.
 *
 * Lança se o artefato não existe ou não expõe a métrica. Isso é deliberado:
 * o build quebra em vez de o site publicar um número sem procedência.
 */
export function claim(artifactId: string, metric: string): Claim {
  const a = EVIDENCE[artifactId];
  if (!a) throw new MissingEvidence(artifactId);
  const value = a.metrics[metric];
  if (typeof value !== 'number') throw new MissingEvidence(artifactId, metric);
  // único ponto do código onde um Claim é construído
  return {
    value, level: a.level, artifact: a.id, metric,
    reason: a.reason, generatedAt: a.generatedAt,
  } as Claim;
}

/** Existe evidência para isto? Use antes de renderizar um caminho opcional. */
export function hasClaim(artifactId: string, metric: string): boolean {
  const a = EVIDENCE[artifactId];
  return !!a && typeof a.metrics[metric] === 'number';
}

/** Todos os artefatos de um veredito. Só portões têm veredito. */
export function byLevel(level: Level): Artifact[] {
  return Object.values(EVIDENCE).filter(a => a.level === level);
}

/** A recusa é um resultado de primeira classe, não um erro. */
export function refusals(): Artifact[] {
  return byLevel('refused');
}

export function byKind(kind: Kind): Artifact[] {
  return Object.values(EVIDENCE).filter(a => a.kind === kind);
}

/** Os territórios, ordenados por quantidade, com a contagem de cada espécie. */
export interface Territory {
  name: string;
  artifacts: Artifact[];
  gates: number;
  records: number;
  refused: number;
}

export function territories(): Territory[] {
  const map = new Map<string, Artifact[]>();
  for (const a of Object.values(EVIDENCE)) {
    const list = map.get(a.territory) ?? [];
    list.push(a);
    map.set(a.territory, list);
  }
  return [...map.entries()]
    .map(([name, artifacts]) => ({
      name,
      // recusas primeiro dentro do território: é o que mais importa ler
      artifacts: artifacts.sort((x, y) => {
        const rank = (a: Artifact) =>
          a.level === 'refused' ? 0 : a.kind === 'gate' ? 1 : 2;
        return rank(x) - rank(y) || x.id.localeCompare(y.id);
      }),
      gates: artifacts.filter(a => a.kind === 'gate').length,
      records: artifacts.filter(a => a.kind === 'record').length,
      refused: artifacts.filter(a => a.level === 'refused').length,
    }))
    .sort((a, b) => b.artifacts.length - a.artifacts.length);
}

/** Data legível a partir do carimbo do artefato, ou null. */
export function measuredOn(a: Artifact): string | null {
  if (!a.generatedAt) return null;
  const d = new Date(a.generatedAt);
  return Number.isNaN(d.getTime())
    ? null
    : d.toLocaleDateString('en-GB', { year: 'numeric', month: 'short', day: 'numeric' });
}

/**
 * Formata sem PERDER precisão.
 *
 * `toLocaleString` sem opções arredonda a três casas decimais em silêncio, e
 * este site vinha publicando `1.99994` como `2` e `0.999999` como `1` — o que
 * apagava exactamente a diferença que a medição existe para mostrar. Num site
 * cuja tese é fidelidade numérica, arredondar a afirmação é o pior defeito
 * possível: o portão garante a PROCEDÊNCIA do número e nada garantia o número.
 *
 * A separação de milhares fica; o que sai é o corte de casas.
 */
export function format(v: number): string {
  return v.toLocaleString('en-GB', { maximumFractionDigits: 20 });
}

