# Archived v1 artifacts (superseded)

These files are the **v1** results produced by experiment version `v1-substring-match`
(commit `9a64e1d`). They are preserved verbatim for traceability and must not be
mixed with the v2 results in the parent directory.

Why they were superseded:

- v1 matched keywords as **substrings**, so the positive keyword `safe` also fired
  on `unsafe` and `clear` also fired on `unclear`. The v2 runner matches whole
  word tokens.
- v1 decision label `rollback_regression` actually means "the candidate was
  rejected"; no already-applied policy state is ever reverted.
- v1 candidates could add one word to both polarities, producing a self-cancelling
  no-op candidate. v2 rejects that by construction and in `validate_proposal`.
- v1 artifacts recorded `accepted_count` as "accepted versions including the
  baseline", which the report presented as an accepted-candidate count.

The v2 artifacts live in `results/` and carry
`"experiment_version": "v2-token-match"`. Do not pool v1 and v2 numbers.
