/**
 * Ilhas — o mínimo de JavaScript sobre o HTML que o Swift gerou.
 *
 * Sem React aqui, e é deliberado: a primeira versão usava React para desenhar
 * três botões que só marcam um atributo, e custava 73 KB comprimidos. Estas
 * ilhas manipulam atributos; o CSS faz o trabalho visual. React continua
 * disponível para as ilhas que realmente precisam dele — simuladores three.js,
 * o seletor de idioma por página — e será carregado só nas páginas que os têm.
 */

// ---------------------------------------------------------------- tema
const KEY = 'sounio-theme';
try {
  const saved = localStorage.getItem(KEY);
  if (saved === 'dark' || saved === 'light') {
    document.documentElement.setAttribute('data-theme', saved);
  }
} catch { /* navegador sem storage: o sistema decide */ }

document.addEventListener('click', e => {
  const t = (e.target as HTMLElement | null)?.closest('[data-theme-toggle]');
  if (!t) return;
  const root = document.documentElement;
  const current = root.getAttribute('data-theme')
    ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const next = current === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  try { localStorage.setItem(KEY, next); } catch { /* idem */ }
});

// -------------------------------------------------- filtro de espécie
//
// A ilha não conhece os artefatos: o Swift já escreveu as 523 linhas com
// data-kind. Aqui só se marca data-filter no contêiner; o CSS esconde.
// Sem JavaScript a página fica completa — sem filtro, mas inteira.

const SPECIES: [string, string][] = [
  ['all', 'Everything'], ['gate', 'Gates'], ['record', 'Records'],
];

function mountFilter(host: HTMLElement) {
  const target = document.querySelector<HTMLElement>(host.dataset.target ?? '.territories');
  if (!target) return;

  const group = document.createElement('div');
  group.className = 'filters';
  group.setAttribute('role', 'group');
  group.setAttribute('aria-label', 'Filter by species');

  const buttons = SPECIES.map(([value, label]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'filter';
    b.textContent = label;
    b.setAttribute('aria-pressed', String(value === 'all'));
    b.addEventListener('click', () => {
      target.dataset.filter = value;
      for (const other of buttons) {
        other.setAttribute('aria-pressed', String(other === b));
      }
    });
    group.append(b);
    return b;
  });

  host.replaceChildren(group);
}

for (const el of document.querySelectorAll<HTMLElement>('[data-island="proof-filter"]')) {
  mountFilter(el);
}

// ----------------------------------------------------- a cena da dose
//
// A ilha NÃO conhece a cena: o Swift já escreveu os parâmetros em data-*
// no contêiner, lidos do mesmo artefato que o React lê. E não reimplementa
// a propagação — importa `propagate` de ../dose, a mesma função que o
// componente React usa. Duas renderizações, um cálculo.
//
// Sem JavaScript o rig fica parado no σ que o gerador escreveu: um valor
// legítimo da cena, não um erro.

import { propagate, scaleKind, pos, f1, type Scene } from '../dose';

function num(el: HTMLElement, k: string): number {
  const v = Number(el.dataset[k]);
  if (!Number.isFinite(v)) throw new Error(`rig da dose sem data-${k}`);
  return v;
}

function mountRig(host: HTMLElement) {
  const scene: Scene = {
    weight: num(host, 'weight'), perKg: num(host, 'perKg'),
    perKgVar: num(host, 'perKgVar'),
    windowLo: num(host, 'windowLo'), windowHi: num(host, 'windowHi'),
    z: num(host, 'z'),
    axisLo: num(host, 'axisLo'), axisHi: num(host, 'axisHi'),
    tickStep: num(host, 'tickStep'),
  };

  const q = <T extends HTMLElement>(sel: string) => host.querySelector<T>(sel);
  const input = q<HTMLInputElement>('input[type=range]');
  const axis = q('.axis'), band = q('.bandbar'), point = q('.pointmk');
  const plabel = q('.pointlab'), verdict = q('.verdict');
  const vv = q('.verdict-v'), vw = q('.verdict-w');
  const sv = q('.ctl-val b'), kind = q('.ctl-val span:last-child');
  const variance = q('.rig-var');
  if (!input || !axis || !band || !point || !verdict || !vv || !vw) return;

  const at = (mg: number) => `${pos(scene, mg)}%`;

  function draw(sd: number) {
    const r = propagate(scene, sd);
    band!.style.left = at(r.lo);
    band!.style.width = `${pos(scene, r.hi) - pos(scene, r.lo)}%`;
    point!.style.left = at(r.mean);
    if (plabel) { plabel.style.left = at(r.mean); plabel.textContent = f1(r.mean); }
    axis!.toggleAttribute('data-refused', r.outside);
    verdict!.toggleAttribute('data-refused', r.outside);
    if (sv) sv.textContent = f1(sd);
    if (kind) kind.textContent = scaleKind(sd);
    if (variance) variance.textContent = `Var = ${Math.round(r.variance)}`;
    vv!.textContent = r.outside ? 'REFUSED' : `${f1(r.mean)} ± ${f1(r.sd)} mg`;
    vw!.textContent = r.outside
      ? `The interval [${f1(r.lo)}, ${f1(r.hi)}] leaves the window. The bound is `
        + `checked against the interval, so there is no longer a safe dose to assert.`
      : `Interval [${f1(r.lo)}, ${f1(r.hi)}] — inside the window`;
  }

  input.addEventListener('input', () => draw(+input.value));
  draw(+input.value);
}

for (const el of document.querySelectorAll<HTMLElement>('[data-island="dose-rig"]')) {
  mountRig(el);
}

// ------------------------------------------------ a varredura do hidrogénio
//
// A ilha NÃO calcula: o gerador já escreveu as três larguras medidas em
// data-dt4/dt2/dt1 em cada barra. Aqui só se troca qual delas está aplicada.
// É o mesmo princípio do filtro do /proof — o cálculo fica onde há evidência,
// e o JavaScript move atributos.

function mountSweep(host: HTMLElement) {
  const buttons = [...host.querySelectorAll<HTMLButtonElement>('.step')];
  const bars = [...host.querySelectorAll<HTMLElement>('.sp-bar')];
  const nums = [...host.querySelectorAll<HTMLElement>('.sp-n')];
  if (!buttons.length) return;

  function apply(key: string) {
    for (const el of bars) {
      const w = el.dataset[key];
      if (w) el.style.width = `${Number(w) * 100}%`;
    }
    for (const el of nums) {
      const w = el.dataset[key];
      if (w) el.textContent = w;
    }
    host.dataset.step = key;
    for (const b of buttons) {
      b.setAttribute('aria-pressed', String(b.dataset.step === key));
    }
  }

  for (const b of buttons) {
    b.addEventListener('click', () => { if (b.dataset.step) apply(b.dataset.step); });
  }
}

for (const el of document.querySelectorAll<HTMLElement>('[data-island="h2-sweep"]')) {
  mountSweep(el);
}
