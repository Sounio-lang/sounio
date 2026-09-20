#!/usr/bin/env python3
"""Selection, negative controls, and workflow/evaluator consistency (stdlib only)."""
import copy
from pathlib import Path
import re
import unittest
from evaluate_ci_decision import IMPACT_KEYS, FAST, DEEP, selected_jobs, evaluate

ROOT = Path(__file__).resolve().parents[2]


def fixture(keys=(), event='pull_request', nightly=False, layer='all'):
    impact = {k: str(k in keys or event != 'pull_request' or 'full' in keys).lower()
              for k in IMPACT_KEYS}
    impact['exhaustive'] = str(event != 'pull_request' or 'full' in keys).lower()
    required = selected_jobs(impact, event, nightly)
    scope = FAST if layer == 'fast' else DEEP if layer == 'deep' else required
    needs = {k: {'result': 'success' if required[k] else 'skipped'} for k in scope}
    needs['impact'] = {'result': 'success', 'outputs': impact}
    return needs


class Layers(unittest.TestCase):
    def test_every_selected_job_must_answer(self):
        for event in ('pull_request', 'push', 'merge_group', 'schedule', 'workflow_dispatch'):
            for keys in ((), ('compiler',), ('runtime',), ('stdlib',), ('tests',), ('lean',), ('full',)):
                for nightly in (False, True):
                    for layer in ('fast', 'deep', 'all'):
                        n = fixture(keys, event, nightly, layer)
                        self.assertEqual(evaluate(n, event, nightly, layer), [])
                        for job in n:
                            for result in ('failure', 'cancelled', 'missing', 'skipped'):
                                if result == 'skipped' and n[job]['result'] == 'skipped':
                                    continue
                                bad = copy.deepcopy(n)
                                if result == 'missing': del bad[job]
                                else: bad[job]['result'] = result
                                with self.subTest(event=event, keys=keys, layer=layer, job=job, result=result):
                                    self.assertTrue(evaluate(bad, event, nightly, layer))

    def test_classification_is_evidence(self):
        n = fixture(('compiler',))
        for key in (*IMPACT_KEYS, 'exhaustive'):
            bad = copy.deepcopy(n)
            del bad['impact']['outputs'][key]
            self.assertTrue(evaluate(bad, 'pull_request'))
        n['impact']['outputs']['exhaustive'] = 'true'
        self.assertTrue(evaluate(n, 'pull_request'))

    def test_intended_policy(self):
        for key in ('compiler', 'stdlib', 'tests'):
            n = fixture((key,))
            self.assertEqual(n['madaros-current-source-deref-f64']['result'], 'success')
            self.assertEqual(n['madaros-witness-gate']['result'], 'success')
            self.assertEqual(n['madaros-fixed-point']['result'], 'skipped')
            self.assertEqual(n['full-test-suite']['result'], 'skipped')
        for event in ('merge_group', 'push', 'schedule', 'workflow_dispatch'):
            n = fixture(event=event)
            for job in ('chemistry', 'lean-proofs', 'full-test-suite', 'madaros-fixed-point', 'native-selfhost-macos-arm64'):
                self.assertEqual(n[job]['result'], 'success')
        self.assertEqual(fixture(event='schedule')['r6-corpus-sweep']['result'], 'success')
        self.assertEqual(fixture(event='workflow_dispatch', nightly=True)['r6-corpus-sweep']['result'], 'success')

    def test_workflow_needs_and_conditions(self):
        # Parse only the deliberately simple job-level fields; actionlint handles YAML.
        text = (ROOT / '.github/workflows/ci.yml').read_text().split('\njobs:\n', 1)[1]
        blocks = dict(re.findall(r'^  ([a-z][\w-]*):\n(.*?)(?=^  [a-z][\w-]*:|\Z)', text, re.M | re.S))
        all_jobs = set(fixture()) - {'impact'}
        self.assertEqual(set(blocks), all_jobs | {'impact', 'ci-decision'})
        for name, scope in (('ci-decision', all_jobs | {'impact'}), ('fast-pr-gate', FAST | {'impact'}), ('deep-compiler-gate', DEEP | {'impact'})):
            b = blocks[name]
            match = re.search(r'^    needs: \[(.*)\]$', b, re.M)
            actual = set(x.strip() for x in match[1].split(',')) if match else set(re.findall(r'^      - ([\w-]+)$', b, re.M))
            self.assertEqual(actual, scope)
        for event in ('pull_request', 'push', 'merge_group', 'schedule', 'workflow_dispatch'):
            for keys in ((), ('compiler',), ('stdlib',), ('runtime',), ('tests',), ('lean',), ('full',)):
                for nightly in (False, True):
                    n = fixture(keys, event, nightly)
                    impact = n['impact']['outputs']
                    for job in all_jobs:
                        cond = re.search(r'^    if: (.*)$', blocks[job], re.M)
                        expression = cond[1] if cond else 'True'
                        expression = re.sub(r'needs\.impact\.outputs\.(\w+)', lambda m: repr(impact[m[1]]), expression)
                        expression = re.sub(r'needs\.[\w-]+\.result', "'success'", expression)
                        expression = expression.replace('github.event_name', repr(event)).replace('inputs.nightly', repr(nightly))
                        expression = expression.replace('always()', 'True').replace('&&', ' and ').replace('||', ' or ')
                        expression = re.sub(r'\btrue\b(?!\')', 'True', expression)
                        with self.subTest(job=job, event=event, keys=keys, nightly=nightly):
                            self.assertEqual(bool(eval(expression, {'__builtins__': {}}, {})), n[job]['result'] == 'success')


if __name__ == '__main__':
    unittest.main()
