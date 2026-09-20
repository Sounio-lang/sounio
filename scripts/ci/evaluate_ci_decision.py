#!/usr/bin/env python3
"""Fail closed on missing evidence in Fast, Deep, or complete CI qualification."""
from __future__ import annotations
import json
import os
import sys

IMPACT_KEYS = ('docs', 'website', 'compiler', 'runtime', 'stdlib', 'tests',
               'lean', 'math', 'ontology', 'clinical', 'sio', 'full')
FAST = {'contracts', 'canonical-madaros', 'gate-wave-0', 'sounio-lint', 'website'}
DEEP = {'contracts-ontology', 'correlated-effect', 'canonical-madaros',
        'madaros-current-source-deref-f64', 'madaros-witness-gate', 'lean-proofs'}


def selected_jobs(impact, event, nightly=False):
    if event not in {'pull_request', 'push', 'merge_group', 'schedule', 'workflow_dispatch'}:
        raise ValueError(f'unsupported event: {event}')
    if any(impact.get(k) not in {'true', 'false'} for k in IMPACT_KEYS):
        raise ValueError('missing or malformed impact classification')
    has = lambda *keys: any(impact[k] == 'true' for k in keys)
    exhaustive = event != 'pull_request' or has('full')
    if impact.get('exhaustive') != str(exhaustive).lower():
        raise ValueError('exhaustive selection disagrees with event/impact')
    compiler = has('compiler', 'runtime', 'stdlib', 'tests', 'full')
    return {
        'contracts': True,
        'contracts-ontology': True,
        # Correlated-effect behavioural controls historically run on EVERY PR.
        'canonical-madaros': True,
        'correlated-effect': True,
        'native-selfhost-linux-x86_64': exhaustive,
        'source-bootstrap-selfhost-linux-x86_64': exhaustive,
        'madaros-current-source-deref-f64': has('compiler', 'stdlib', 'tests', 'full'),
        'madaros-fixed-point': exhaustive,
        'native-selfhost-macos-arm64': exhaustive,
        'full-test-suite': exhaustive,
        'madaros-witness-gate': compiler,
        'gate-wave-0': compiler,
        'sounio-lint': has('compiler', 'stdlib', 'tests', 'sio', 'full'),
        'lean-proofs': has('lean', 'full'),
        'website': has('website', 'full'),
        'chemistry': exhaustive,
        'r6-corpus-sweep': event == 'schedule' or (event == 'workflow_dispatch' and nightly),
        'fast-pr-gate': True,
        'deep-compiler-gate': True,
    }


def evaluate(needs, event, nightly=False, layer='all'):
    if needs.get('impact', {}).get('result') != 'success':
        return ['impact did not succeed']
    try:
        required = selected_jobs(needs['impact'].get('outputs', {}), event, nightly)
    except ValueError as exc:
        return [str(exc)]
    if layer == 'fast':
        required = {key: required[key] for key in FAST}
    elif layer == 'deep':
        required = {key: required[key] for key in DEEP}
    elif layer != 'all':
        return [f'unknown CI layer: {layer}']
    expected = set(required) | {'impact'}
    failures = []
    if set(needs) != expected:
        failures.append(f'needs mismatch: missing={sorted(expected-set(needs))} extra={sorted(set(needs)-expected)}')
    for job, selected in required.items():
        result = needs.get(job, {}).get('result', 'missing')
        if selected and result != 'success':
            failures.append(f'selected job {job} ended as {result}')
        elif not selected and result not in {'success', 'skipped'}:
            failures.append(f'unselected job {job} ended as {result}')
    return failures


def main():
    try:
        failures = evaluate(json.loads(os.environ.get('NEEDS_JSON', '{}')),
                            os.environ.get('GITHUB_EVENT_NAME', ''),
                            os.environ.get('NIGHTLY_INPUT', '').lower() == 'true',
                            os.environ.get('CI_LAYER', 'all'))
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        failures = [f'invalid evidence: {exc}']
    for failure in failures:
        print(f'CI_DECISION_FAIL: {failure}', file=sys.stderr)
    if not failures:
        print('CI_DECISION_PASS')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
