A pull request whose earlier head already ran, and passed, every lane its current head needs no longer reruns the heavy jobs after merging main, provided the merge brought in only website files, docs or new changelog fragments. `scripts/ci/ci_reuse_green_ancestor.sh` looks for that head and accepts it only under these conditions:
- the earlier head has a successful `pull_request` run of `ci.yml` for this same PR;
- that run's own Impact job selected every lane that the full BASE..HEAD classification selects;
- that run did not itself reuse a verdict.

The impact classifier then schedules jobs from the delta since that head. Policy gates read the full classification instead, through the new `policy_*` outputs, and the changelog fragment gate is one of them.

The earlier head is disqualified if the delta deletes a fragment, edits `CHANGELOG.md` by hand, or touches a workflow, a script, stdlib, tests or the compiler. If the API cannot answer, the full classification runs. `impact_ci_selftest.sh` has ten new controls.
