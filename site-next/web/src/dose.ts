/**
 * A propagação GUM da cena da dose — uma única cópia.
 *
 * O componente React e a ilha estática desenham coisas diferentes, mas não
 * podem calcular coisas diferentes. Aqui mora só a FÓRMULA; todos os
 * parâmetros (peso, diretriz, janela, z) chegam de fora, lidos do artefato
 * `site/dose_scene.v1`. Nenhum número desta cena é escolhido aqui.
 */
export interface Scene {
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
  const variance = s.perKg * s.perKg * weightSd * weightSd
                 + s.weight * s.weight * s.perKgVar;
  const sd = Math.sqrt(variance);
  const mean = s.weight * s.perKg;
  const lo = mean - s.z * sd;
  const hi = mean + s.z * sd;
  return { mean, sd, lo, hi, variance, outside: lo < s.windowLo || hi > s.windowHi };
}

/** Como a balança seria descrita numa enfermaria, dado o σ. */
export function scaleKind(sd: number): string {
  if (sd <= 0.8) return 'calibrated clinical scale';
  if (sd <= 2.5) return 'ward scale';
  if (sd <= 4.5) return 'old scale, patient in clothes';
  return 'eyeballed from the chart';
}

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
