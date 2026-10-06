/**
 * Honesty kernel for Knowledge<T> display.
 *
 * A well-typed program produces a value or refuses. It does not invent a
 * zero. That is the Lean thesis of SounioRefusalHonesty
 * (`refusal_is_not_zero`, `well_typed_value_or_refuse`) — a model of the
 * E219 fragment, not a proof that today's backend implements it.
 *
 * This module is the visual counterpart: it decides what a control may
 * print. It will not render an uncalibrated variance as `var=0.000`, and
 * it will not render a non-unit-interval bit pattern as ε.
 */

export type EpistemicState = 'verified' | 'uncertain' | 'refused';

export type VarianceStatus = 'calibrated' | 'uncalibrated' | 'missing' | 'invalid';

export type EpsilonStatus = 'readable' | 'missing' | 'unreadable';

export type KnowledgeInput = {
  value?: number | null;
  variance?: number | null;
  /** When false, a stored 0 is "uncalibrated", not "exact". Default: treat 0 as uncalibrated. */
  varianceCalibrated?: boolean;
  epsilon?: number | null;
  /** Guard of the form `where x.ε >= threshold`. Missing threshold does not refuse. */
  threshold?: number | null;
  /** Caller-claimed triad state. Honesty failures override a verified/uncertain claim. */
  state?: EpistemicState;
  boundLow?: number | null;
  boundHigh?: number | null;
};

export type KnowledgeReading = {
  state: EpistemicState;
  /** Null means: print no numeral. A refused fabrication has no value slot. */
  value: number | null;
  /** Null means: print no `var=…`. Uncalibrated zero is not a variance. */
  variance: number | null;
  varianceStatus: VarianceStatus;
  epsilon: number | null;
  epsilonStatus: EpsilonStatus;
  expandedU: number | null;
  boundLow: number | null;
  boundHigh: number | null;
  reason:
    | 'caller-state'
    | 'guard-failed'
    | 'unreadable-epsilon'
    | 'fabrication-shape'
    | 'invalid-bounds';
};

export function isUnitInterval(value: number): boolean {
  return Number.isFinite(value) && value >= 0 && value <= 1;
}

export function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

export function isNonNegativeFinite(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

function asOptionalNumber(value: number | null | undefined): number | null {
  if (value === null || value === undefined) return null;
  return Number.isFinite(value) ? value : null;
}

export function classifyVariance(
  variance: number | null | undefined,
  calibrated?: boolean,
): { status: VarianceStatus; variance: number | null } {
  if (variance === null || variance === undefined) {
    return { status: 'missing', variance: null };
  }
  if (!isNonNegativeFinite(variance)) {
    return { status: 'invalid', variance: null };
  }
  if (variance === 0 && calibrated !== true) {
    // stdlib/clinical/vancomycin_pbpk.sio leaves v = 0.0 when the joint
    // is unknown: "Leave variance uncalibrated instead of inventing
    // pseudo-statistics". Printing that as var=0.000 would be a lie.
    return { status: 'uncalibrated', variance: null };
  }
  return { status: 'calibrated', variance };
}

export function classifyEpsilon(
  epsilon: number | null | undefined,
): { status: EpsilonStatus; epsilon: number | null } {
  if (epsilon === null || epsilon === undefined) {
    return { status: 'missing', epsilon: null };
  }
  if (!isUnitInterval(epsilon)) {
    return { status: 'unreadable', epsilon: null };
  }
  return { status: 'readable', epsilon };
}

/**
 * Detect the dissertation-suite fabrication shape: a stub zero variance
 * paired with a confidence that is not a unit interval (the observed
 * raw integer 4604219396932172800 is the witness, not a display value).
 */
export function isFabricationShape(
  variance: number | null | undefined,
  epsilon: number | null | undefined,
): boolean {
  const epsilonGiven = epsilon !== null && epsilon !== undefined;
  if (!epsilonGiven || isUnitInterval(epsilon)) return false;
  if (variance === null || variance === undefined) return true;
  return variance === 0 || !isNonNegativeFinite(variance);
}

export function readKnowledge(
  input: KnowledgeInput,
  coverageK = 2,
): KnowledgeReading {
  const fabrication = isFabricationShape(input.variance, input.epsilon);
  const { status: varianceStatus, variance } = classifyVariance(
    input.variance,
    input.varianceCalibrated,
  );
  const { status: epsilonStatus, epsilon } = classifyEpsilon(input.epsilon);

  const rawValue = asOptionalNumber(input.value);
  const boundLow = asOptionalNumber(input.boundLow);
  const boundHigh = asOptionalNumber(input.boundHigh);
  const boundsValid =
    boundLow !== null && boundHigh !== null && boundLow <= boundHigh;

  let state: EpistemicState = input.state ?? 'uncertain';
  let reason: KnowledgeReading['reason'] = 'caller-state';

  if (fabrication) {
    state = 'refused';
    reason = 'fabrication-shape';
  } else if (epsilonStatus === 'unreadable') {
    state = 'refused';
    reason = 'unreadable-epsilon';
  } else if (
    epsilonStatus === 'readable' &&
    epsilon !== null &&
    input.threshold !== null &&
    input.threshold !== undefined &&
    isUnitInterval(input.threshold) &&
    epsilon < input.threshold
  ) {
    state = 'refused';
    reason = 'guard-failed';
  } else if (input.boundLow != null && input.boundHigh != null && !boundsValid) {
    state = 'refused';
    reason = 'invalid-bounds';
  } else if (input.state) {
    state = input.state;
    reason = 'caller-state';
  }

  const hideNumeral = reason === 'fabrication-shape';

  const expandedU =
    varianceStatus === 'calibrated' &&
    variance !== null &&
    Number.isFinite(coverageK) &&
    coverageK > 0
      ? coverageK * Math.sqrt(variance)
      : null;

  return {
    state,
    value: hideNumeral ? null : rawValue,
    variance: hideNumeral ? null : variance,
    varianceStatus: hideNumeral ? 'invalid' : varianceStatus,
    epsilon: hideNumeral ? null : epsilon,
    epsilonStatus: hideNumeral ? 'unreadable' : epsilonStatus,
    expandedU: hideNumeral ? null : expandedU,
    boundLow: hideNumeral ? null : boundsValid ? boundLow : null,
    boundHigh: hideNumeral ? null : boundsValid ? boundHigh : null,
    reason,
  };
}

export function formatFixed(value: number, digits: number): string {
  if (!Number.isFinite(value)) return '—';
  return value.toLocaleString('en-GB', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
    useGrouping: false,
  });
}
