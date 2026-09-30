/**
 * A cena da combustão de hidrogénio — a derivação, em cópia única.
 *
 * As medições dão RAZÕES entre bandas (banda em dt=4e-9 dividida pela banda
 * em dt menor). O desenho precisa da largura relativa: quanto a banda vale em
 * cada passo, tomando o próprio valor em dt = 4e-9 como 1.
 *
 * Não há âncora absoluta porque o documento-fonte não publica uma para o lado
 * da quadratura. Normalizar cada implementação ao seu próprio ponto de partida
 * mantém os dois lados como medições.
 */
export interface Ratios { half1: number; half2: number; quarter: number }

/** Os três passos medidos. Não há continuum: a interação para onde a medição parou. */
export const STEPS = [
  { key: 'dt4', label: '4e-9 s' },
  { key: 'dt2', label: '2e-9 s' },
  { key: 'dt1', label: '1e-9 s' },
] as const;

export type StepKey = (typeof STEPS)[number]['key'];

/** Largura relativa da banda em cada passo, com dt = 4e-9 valendo 1. */
export function widths(r: Ratios): Record<StepKey, number> {
  return { dt4: 1, dt2: 1 / r.half1, dt1: 1 / r.quarter };
}

export const f3 = (v: number) => v.toFixed(3);
