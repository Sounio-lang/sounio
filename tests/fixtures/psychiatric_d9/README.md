<!-- docs:meta
topic_id: repo.tests.fixtures.psychiatric-d9-readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.tests.fixtures.psychiatric-d9-readme
-->

# Psychiatric D9 External Software-Integration Fixture

This directory contains a frozen copy of the UCI Machine Learning Repository
dataset **Drug Consumption (Quantified)** for a bounded external-evaluation
candidate in D9.

Citation:

> Fehrman, E., Egan, V., and Mirkes, E. (2015). Drug Consumption
> (Quantified) [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5TC7S

Source archive:

`https://archive.ics.uci.edu/static/public/373/drug+consumption+quantified.zip`

UCI publishes the dataset under the Creative Commons Attribution 4.0
International license. The license permits sharing and adaptation with
attribution: <https://creativecommons.org/licenses/by/4.0/>.

The frozen file has 1,885 rows and 32 comma-separated columns. The schema uses
numeric respondent IDs and quantified demographic/personality fields; no
direct names or contact fields occur in the supplied table. D9 has not audited
the original recruitment, consent, deidentification process, collection
dates, or chain of clinical custody. Public availability and a SHA-256 digest
do not establish any of those properties.

The prespecified evaluation target is only the self-reported recency category
for the dataset's `Benzos` column. It is not clonazepam response, medication
adherence, a diagnosis, a latent psychiatric state, or a treatment outcome.
The protocol deliberately ends in abstention from real empirical binding
because metrological calibration, collection-window verification, external
custody sealing, and sealed validation are unavailable.

The directory name `psychiatric_d9` and the `clinical_` prefix on some tests
name the semantic domain whose authority is being refused. They do not label
this fixture as a clinical study or an intended-use model. The reported counts
must not be cited as clinical or predictive performance.

The deterministic development partition is deliberately unused: the fixed
score is not fitted, and no coefficient or threshold is selected from those
rows. The `3/4` coverage threshold is an arbitrary software-fixture requirement,
not a statistically or clinically justified decision boundary.

The protocol file was created and hashed before the full-data calculation in
the D9 execution workflow, but that ordering has no independent timestamp,
public preregistration record, or pre-analysis Git commit. It is a local
workflow declaration, not a certified preregistration claim. Its SHA-256 gate
detects later mutation of the committed bytes, but cannot detect outcome
inspection or protocol selection before commit. The protocol bytes remain
frozen after analysis so this limitation can be audited rather than silently
rewritten.

Files:

- `uci_drug_consumption_373.data`: frozen upstream data bytes;
- `evaluation_protocol.v1.json`: locally predeclared protocol, frozen before
  full-data analysis but not independently timestamped;
- `dataset_manifest.v1.json`: source, license, shape, and SHA-256 identities.
