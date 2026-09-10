/** O vocabulário epistêmico do site, em tipos. */

/** Veredito de um portão. Registros não têm nível — não são portões. */
export type Level = 'verified' | 'unbounded' | 'refused';

/** Portão declara veredito; registro traz medição e procedência. */
export type Kind = 'gate' | 'record';

/** O que uma amostra finita sustenta, a 95%. */
export interface Proportion {
  passed: number;
  total: number;
  point: number;
  lo: number;
  hi: number;
  width: number;
}

export interface Artifact {
  id: string;
  kind: Kind;
  /** Presente só quando o artefato reporta passed e total. */
  proportion: Proportion | null;
  territory: string;
  schema: string | null;
  status: string | null;
  level: Level | null;
  reason: string | null;
  generatedAt: string | null;
  metrics: Record<string, number>;
  claims: number;
}

/**
 * Um número que o site tem permissão para exibir.
 *
 * Não há construtor público: a única forma de obter um Claim é `claim()`,
 * que exige um artefato existente e uma métrica que exista nele. É por isso
 * que o site não consegue inventar um número — não há como digitar um.
 */
export interface Claim {
  readonly value: number;
  /** Null quando a origem é um registro: medição sem veredito. */
  readonly level: Level | null;
  readonly artifact: string;
  readonly metric: string;
  readonly reason: string | null;
  readonly generatedAt: string | null;
  readonly [EVIDENCE_BRAND]: true;
}

/** Marca privada: impede construir um Claim fora deste módulo. */
declare const EVIDENCE_BRAND: unique symbol;

export class MissingEvidence extends Error {
  constructor(artifact: string, metric?: string) {
    super(
      metric
        ? `Sem evidência: o artefato "${artifact}" não expõe a métrica "${metric}".`
        : `Sem evidência: não existe artefato "${artifact}".`
    );
    this.name = 'MissingEvidence';
  }
}
