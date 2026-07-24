# TODO: missing artifact pieces (honest list)

The analysis scripts in `analysis/` cover P3 (cross-benchmark mechanism) and
the two probe harnesses (format-lever, WikiContradict). They do **not** yet
cover the core policy-ladder analyses that produce Tables 1 and 2 of the paper.

The following scripts, described in
`research/matrix-mining-reader-conditioning-2026-07.md`, were originally written
in a session scratchpad that no longer exists. They must be **re-implemented in
`analysis/` with tests, to repo standards, before submission**:

- **Policy ladder** (`oracle_gap.py` → e.g. `analysis/policy_ladder.py`):
  10-fold CV over queries; fixed / per-reader-best-fixed / per-question-type /
  per-(reader, question-type) policies; reader-blind and per-(row, reader)
  oracles. Produces Table 1 and the §4 headroom arithmetic (incl.
  flag-drop and label-flip-injection sensitivity).

- **Learnability / calibration sweep** (`learnability.py`): per-(reader,
  question-type) with K ∈ {25–200} labeled calibration rows/reader.

- **Leave-family-out transfer** (`loro_transfer.py` →
  e.g. `analysis/transfer.py`): 100-rep leave-reader-family-out; ensemble-bound
  and realizable regimes; donor majority vote, similarity-weighted vote, and
  single-most-similar-donor rows. Produces Table 2.

- **Oracle noise correction:** the judge-flag-drop and random-label-injection
  passes feeding the §4 numbers (may fold into the policy-ladder script).

These are **not** re-implemented in this session. Until they are, this artifact
regenerates the P3 mechanism results but not the policy-ladder tables.
