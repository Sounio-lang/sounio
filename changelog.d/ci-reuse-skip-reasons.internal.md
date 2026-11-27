When `scripts/ci/ci_reuse_green_ancestor.sh` turns down an earlier head of the pull request, it now says why on stderr, one `ci-reuse: skip <sha>: <reason>` line per candidate, and the Impact log keeps that text. The possible reasons are:
- the raw delta is not reusable;
- there is no successful `pull_request` run of `ci.yml` for the head;
- there are green runs, but none lists this PR (each run is listed with its PR numbers);
- an API call failed (the reason carries the error text);
- the run itself reused a verdict;
- the impact rows or the checkout line are missing;
- the lanes are not covered;
- the tested tree differs from the one tested now.

The first live miss was #3750 at `3aefd8f93a`: its only change since a green head was in `docs/`, yet it got `reused_from=` empty, and nothing in the log said which condition failed. In `impact_ci_selftest.sh`, control 13 checks that a refused candidate prints its reason. Controls 14 to 17 drive the API leg through a stub `gh`: a green run that does not list the PR, a failed runs call and no green run each give a distinct reason, and a green run that does list the PR is reused.
