/**
 * One claim type for every numeral the site is allowed to print.
 *
 * suiteFaceParts is the pbpk_suite instance. The GPU 13/13 is another.
 * A third one-off kernel would be the same class as an undated 6/6:
 * a new place that looks protected and is not.
 *
 * A claim may not print unless three things close: the parts sum,
 * the engine is named (Madaros or lean_single), and reachability is
 * complete. Reachability that can only say "no" is a constant, not
 * a label. WORKFLOW-REACHABLE therefore requires the workflow file
 * and the line that invokes the gate. Without that pair, a 13/13
 * that enters CI tomorrow would keep saying "not in CI".
 *
 * `bin/souc` and `souc-linux-x86_64-gpu` are launchers, not engines.
 * Do not invent Madaros or lean_single from a path.
 *
 * The site does not classify digits. It classifies who has license
 * to assert. Three surfaces:
 *   1. claimFace / claimRefusal — a MeasurementClaim that may print.
 *   2. Unlicensed wells — a live page reading metrics.* / proof.*
 *      without (1) or (3) fails the check.
 *   3. claimEscape — a numeral that is not a scientific assertion.
 *      Annotated, counted, printed as a table on every check run.
 * Without (3), (1) and (2) push people to turn the script off.
 * A silent skip is a door. A counted escape is a measurement.
 *
 * The CLOSED_PATTERNS denylist is debt. Do not grow it. Retire it
 * after the license rule is the sole green on one Website CI run
 * on main and the positive control still fires. Two mechanisms
 * for the same thing is how neither is owned.
 */

export const COMPILER_ENGINES = ['Madaros', 'lean_single'] as const;
export type CompilerEngine = (typeof COMPILER_ENGINES)[number];

export const REACHABILITIES = ['WORKFLOW-UNREACHABLE', 'WORKFLOW-REACHABLE'] as const;
export type Reachability = (typeof REACHABILITIES)[number];

export type ReachabilityBinding =
  | { reachability: 'WORKFLOW-UNREACHABLE' }
  | { reachability: 'WORKFLOW-REACHABLE'; workflow: string; line: number };

export type ClaimKind = 'outcome' | 'variance' | 'green-count' | 'inventory';

export type MeasurementClaim = ReachabilityBinding & {
  id: string;
  kind: ClaimKind;
  gate: string | null;
  engine: CompilerEngine | null;
  measuredAt: string;
  artifact: string;
  parts: Record<string, number>;
};

export function isCompilerEngine(value: string): value is CompilerEngine {
  return (COMPILER_ENGINES as readonly string[]).includes(value);
}

export function isReachability(value: string): value is Reachability {
  return (REACHABILITIES as readonly string[]).includes(value);
}

export function reachabilityComplete(b: ReachabilityBinding): boolean {
  if (b.reachability === 'WORKFLOW-UNREACHABLE') return true;
  return (
    b.reachability === 'WORKFLOW-REACHABLE' &&
    typeof b.workflow === 'string' &&
    b.workflow.endsWith('.yml') &&
    Number.isInteger(b.line) &&
    b.line > 0
  );
}

/**
 * Construction refusal. A REACHABLE claim missing the workflow line
 * must not load — that is how a label that can only say "no" comes back.
 */
export function assertReachabilityComplete(b: ReachabilityBinding, id: string): void {
  if (!reachabilityComplete(b)) {
    throw new Error(
      `measurementClaim: ${id} reachability incomplete — WORKFLOW-REACHABLE requires a .yml path and line > 0`,
    );
  }
}

export function reachabilityFace(b: ReachabilityBinding): string {
  if (b.reachability === 'WORKFLOW-UNREACHABLE') return 'not in CI';
  return `in CI · ${b.workflow}:${b.line}`;
}

export function claimEngineOk(c: MeasurementClaim): boolean {
  return c.engine !== null && isCompilerEngine(c.engine);
}

export function claimMayPrint(
  c: MeasurementClaim,
  partsClose: (claim: MeasurementClaim) => boolean,
): boolean {
  return partsClose(c) && claimEngineOk(c) && reachabilityComplete(c);
}

export function claimFace(
  c: MeasurementClaim,
  partsClose: (claim: MeasurementClaim) => boolean,
  numeral: string,
): string {
  if (!claimMayPrint(c, partsClose) || c.engine === null) {
    throw new Error(
      `measurementClaim: refuse to print ${c.id} without reachability and engine`,
    );
  }
  return `${numeral} · ${c.engine} · ${reachabilityFace(c)} · artifact ${c.measuredAt} · ${c.artifact}`;
}

export function claimRefusal(c: MeasurementClaim, noun: string): string {
  return `${noun} · ${reachabilityFace(c)} · artifact ${c.measuredAt.slice(0, 10)} · engine unnamed`;
}

export const ESCAPE_CLASSES = ['inventory', 'literature', 'identity', 'threshold', 'version'] as const;
export type EscapeClass = (typeof ESCAPE_CLASSES)[number];

export type ClaimEscape = {
  id: string;
  class: EscapeClass;
  reason: string;
  text: string;
};

const escapeRegistry: ClaimEscape[] = [];

function isEscapeClass(value: string): value is EscapeClass {
  return (ESCAPE_CLASSES as readonly string[]).includes(value);
}

/**
 * Mark a numeral that is not a MeasurementClaim. Returns the text
 * without going through claimFace — this is not a green.
 *
 * id, class, and reason are required. The check parses every call
 * and prints the table on every run. Growing the table is a review
 * signal. An unannotated skip is a refuse.
 */
export function claimEscape(spec: ClaimEscape): string {
  if (typeof spec.id !== 'string' || spec.id.trim().length === 0) {
    throw new Error('claimEscape: id is required');
  }
  if (!isEscapeClass(spec.class)) {
    throw new Error(
      `claimEscape: class must be one of ${ESCAPE_CLASSES.join(', ')} (got ${String(spec.class)})`,
    );
  }
  if (typeof spec.reason !== 'string' || spec.reason.trim().length < 20) {
    throw new Error(
      'claimEscape: reason must say why this numeral is not a MeasurementClaim (≥ 20 characters)',
    );
  }
  escapeRegistry.push({
    id: spec.id.trim(),
    class: spec.class,
    reason: spec.reason.trim(),
    text: spec.text,
  });
  return spec.text;
}

export function claimEscapeTable(): readonly ClaimEscape[] {
  return escapeRegistry.slice();
}
