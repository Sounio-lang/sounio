/**
 * Consulta das leituras GUM calculadas pelo produtor C++23.
 * React e a ilha estática consomem o mesmo artefato source-linked.
 * Esta camada recusa chaves ausentes; não recalcula propagação científica.
 */
export interface Scene {
  readings: Record<string, Reading>;
  weight: number; perKg: number; perKgVar: number;
  windowLo: number; windowHi: number; z: number;
  axisLo: number; axisHi: number; tickStep: number;
}

export interface Reading {
  mean: number; sd: number; lo: number; hi: number;
  variance: number;
  /** O intervalo saiu da janela? Então não há dose segura a afirmar. */
  outside: boolean;
}

/**
 * Método delta: Var(XY) = Y²Var(X) + X²Var(Y).
 * É a mesma fórmula que o exemplo escreve por extenso.
 */
export function propagate(s: Scene, weightSd: number): Reading {
  const key = weightSd.toFixed(1);
  if (Number(key) !== weightSd) throw new RangeError("Slider value is not on the recorded one-decimal grid");
  const r = s.readings[key];
  if (!r) throw new Error("No C++23 source-linked reading for slider key " + key);
  return r;
}

/** Como a balança seria descrita numa enfermaria, dado o σ. */
export function scaleKind(_sd: number): string { return "illustrative measurement uncertainty"; }

/** Posição de uma dose no eixo desenhado, em porcento. Layout, não afirmação. */
export function pos(s: Scene, mg: number): number {
  return ((mg - s.axisLo) / (s.axisHi - s.axisLo)) * 100;
}

/** As marcas do eixo, geradas a partir da extensão — nenhuma escolhida à mão. */
export function ticks(s: Scene): number[] {
  const out: number[] = [];
  for (let v = s.axisLo; v <= s.axisHi + 1e-9; v += s.tickStep) out.push(Math.round(v));
  return out;
}

export const f1 = (v: number) => v.toFixed(1);
