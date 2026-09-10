#!/usr/bin/env node
/**
 * measure-dose-scene.mjs — lê o exemplo de dose e emite a cena de abertura.
 *
 * A abertura do site é uma cena clínica: um paciente pesado, uma diretriz, uma
 * janela terapêutica. Nada disso é inventado — vem de
 * `examples/real_world/01_dose_uncertainty.sio`. Este script EXTRAI os
 * parâmetros do arquivo e falha alto se não os encontrar, para a cena não
 * poder divergir em silêncio do exemplo que ela diz ilustrar.
 *
 * Registra também a discrepância que a extração revelou: o cabeçalho do
 * arquivo documenta uma incerteza menor do que a aritmética do próprio
 * arquivo produz. Qual dos dois está errado está sob investigação; o que está
 * estabelecido é que discordam, e é isso que o artefato afirma.
 */
import { execFileSync } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '../..');
// HEAD, não origin/main: o site descreve a árvore de que foi construído, e
// qualquer pessoa pode fazer checkout desse commit e repetir a medição. Medir
// um ramo que se move faria os números descreverem uma árvore que não é a
// publicada, e reprovaria a guarda de deriva a cada commit alheio.
const REF = process.env.CORPUS_REF ?? 'HEAD';
const SRC = 'examples/real_world/01_dose_uncertainty.sio';
const OUT = join(REPO, 'artifacts/site/dose_scene.v1.json');

const git = (...a) => execFileSync('git', ['-C', REPO, ...a],
  { encoding: 'utf-8', maxBuffer: 1 << 30 });

const src = git('show', `${REF}:${SRC}`);

/** Extrai ou morre. Um exemplo que mudou de forma precisa parar o build. */
function grab(re, what) {
  const m = re.exec(src);
  if (!m) throw new Error(`não encontrei ${what} em ${SRC} — a cena não pode ser montada às cegas`);
  return m.slice(1).map(Number);
}

const [weight, weightVar]  = grab(/let weight\s*=\s*epistemic_new\(([\d.]+),\s*([\d.]+)\)/, 'o peso');
const [perKg, perKgVar]    = grab(/let dose_per_kg\s*=\s*epistemic_new\(([\d.]+),\s*([\d.]+)\)/, 'a dose por kg');
const [windowLo]           = grab(/let therapeutic_min\s*=\s*([\d.]+)/, 'o limite inferior da janela');
const [windowHi]           = grab(/let therapeutic_max\s*=\s*([\d.]+)/, 'o limite superior da janela');
const [docSd]              = grab(/Total Dose:[^\n]*?σ=([\d.]+)/, "o desvio padrão documentado para a dose");
const [docLo, docHi]       = grab(/95% CI:\s*\[([\d.]+),\s*([\d.]+)\]/, 'o intervalo documentado no cabeçalho');

const Z = 1.96;
// GUM, método delta — a mesma fórmula que o arquivo escreve por extenso
const variance = perKg * perKg * weightVar + weight * weight * perKgVar;
const sd = Math.sqrt(variance);
const mean = weight * perKg;
const lo = mean - Z * sd, hi = mean + Z * sd;

/** σ acima do qual o intervalo deixa a janela: onde a banda cruza a linha. */
const halfWidth = Math.min(mean - windowLo, windowHi - mean) / Z;
const sigmaCross = Math.sqrt((halfWidth * halfWidth - weight * weight * perKgVar) / (perKg * perKg));

const round = (v, n = 4) => Number(v.toFixed(n));

// O alcance que o leitor pode explorar. O máximo é DERIVADO do ponto de
// cruzamento: se o slider não passasse dele, a demonstração não poderia ser
// feita — o leitor nunca veria a banda deixar a janela. O mínimo é uma balança
// clínica calibrada. Viajam no artefato para o React e a ilha não escolherem
// alcances diferentes.
const sigmaMax = Math.ceil(sigmaCross * 2 * 2) / 2;
const sigmaMin = 0.3;
const sigmaStep = 0.1;

// A extensão do eixo é DERIVADA: tem de caber a banda mais larga que o leitor
// consegue produzir, senão a demonstração sai do desenho no fim do curso. Os
// ticks saem de um passo redondo que cubra essa extensão.
const widestSd = Math.sqrt(perKg * perKg * sigmaMax * sigmaMax + weight * weight * perKgVar);
const tickStep = 100;
const axisLo = Math.floor((mean - Z * widestSd - tickStep / 2) / tickStep) * tickStep;
const axisHi = Math.ceil((mean + Z * widestSd + tickStep / 2) / tickStep) * tickStep;

const artifact = {
  schema: 'sounio.site.dose_scene.v1',
  generated_at: new Date().toISOString(),
  // registro: mede um exemplo, não declara veredito sobre ele
  reason:
    `Parâmetros extraídos de ${SRC} em ${REF}, não digitados. A propagação ` +
    `é a GUM pelo método delta que o próprio arquivo escreve. O cabeçalho do ` +
    `arquivo documenta um desvio padrão que a sua aritmética não produz; qual ` +
    `dos dois está errado está sob investigação.`,
  ref: REF,
  source: SRC,
  commit: git('rev-parse', REF).trim(),
  metrics: {
    weight, weight_sd: round(Math.sqrt(weightVar)), weight_var: weightVar,
    per_kg: perKg, per_kg_sd: round(Math.sqrt(perKgVar)), per_kg_var: perKgVar,
    window_lo: windowLo, window_hi: windowHi,
    dose: mean,
    variance: round(variance),
    sd: round(sd),
    ci_lo: round(lo), ci_hi: round(hi),
    sigma_crossing: round(sigmaCross),
    sigma_min: sigmaMin, sigma_max: sigmaMax, sigma_step: sigmaStep,
    axis_lo: axisLo, axis_hi: axisHi, tick_step: tickStep,
    z: Z,
    // o que o cabeçalho promete, para a página poder mostrar a discordância
    documented_sd: docSd,
    documented_ci_lo: docLo, documented_ci_hi: docHi,
    documented_width: round(docHi - docLo),
    computed_width: round(hi - lo),
  },
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(artifact, null, 2) + '\n');
console.log(`escrito ${OUT}`);
console.log(`  cena: ${weight} kg (σ ${Math.sqrt(weightVar)}) × ${perKg} mg/kg (σ ${Math.sqrt(perKgVar)})`);
console.log(`  dose: ${mean} ± ${sd.toFixed(1)} mg   IC [${lo.toFixed(1)}, ${hi.toFixed(1)}]`);
console.log(`  janela: ${windowLo}–${windowHi} mg · a banda cruza em σ = ${sigmaCross.toFixed(2)} kg`);
console.log(`  cabeçalho promete σ ${docSd}, IC [${docLo}, ${docHi}] — largura ${(docHi-docLo).toFixed(1)} contra ${(hi-lo).toFixed(1)}`);
