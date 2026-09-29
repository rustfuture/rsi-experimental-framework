# What the 32-sentence pool can and cannot distinguish

This note explains why `ablation_no_selection` and `ablation_no_rollback` match
`baseline_full` in the committed ablation table (held-out 1.000, +0.625, 4 accepted
new changes), and what that does and does not say about those mechanisms. It is an
analysis of the existing deterministic harness (`HARNESS_BASELINE_NOT_LLM`); no
committed result file was changed and no model was involved. The numbers below come
from re-running `run_experiment` with `config/default.json` and can be reproduced
with the commands in [reference.md](reference.md).

## Why the three variants tie on seed 20260915

Structure of the task:

- The pool is 8 cue words (4 positive, 4 negative) times 4 sentence variants. The
  label is fully determined by the cue word; the filler phrase carries no signal.
- The initial policy (`good` / `bad`, bias 0) never fires on any sentence, so it
  predicts 0 everywhere. Every negative example is already correct.
- Consequently `add_negative:*` mutations change no prediction, and the only
  accuracy-improving moves are `add_positive:<true positive cue>`. Each such move
  can only turn a wrong prediction into a right one, so train and dev accuracy are
  both non-decreasing along any path made of these moves.

On the primary seed the greedy path is `useful, safe, clear, reliable` (baseline_full
and no_rollback) with train/dev correct counts 10/3 -> 12/4 -> 14/5 -> 15/7 -> 16/8.
Every accepted step raises train and does not lower dev, so:

- **Rejection never fires.** The baseline run records 0 `rejected_dev_regression`
  events, so disabling it (`no_rollback`) changes nothing. The two runs are
  identical step for step.
- **Dev-blind selection reaches the same place.** `no_selection` ranks by train
  accuracy only. It differs only in tie-breaking (it picks `reliable` before `clear`,
  dev 6 instead of 7 at step 3) and ends with the same four positive words, 16/8
  train/dev and 8/8 held-out.

So the tie is a property of this seed and this monotone task, not evidence that the
mechanisms are useless, and not evidence that they are equivalent in general.

## Where the mechanisms do differ (other seeds)

The ablation table uses one seed. Re-running the same code on the multi-seed seeds
(held-out correct out of 8, my own re-run for this note, not a committed artifact):

| Seed | baseline_full | no_selection | no_rollback | random_selection |
|---:|---:|---:|---:|---:|
| 20260915 | 8 | 8 | 8 | 3 |
| 42 | 8 | 7 | 8 | 4 |
| 1337 | 8 | 8 | 8 | 3 |
| 2026 | 5 | 6 | 8 | 5 |

- Seed 2026 is the informative case. The initial train/dev correct counts are 6/5,
  and the best train candidate is `bias:+1` (train 10, dev 3). Dev falls from 5 to 3,
  so the rejection rule blocks it, all 8 generations propose the same rejected move,
  and the run ends stuck at the baseline (held-out 5/8). Without rejection the run
  accepts the bias move and then reaches 8/8. Here the dev-regression rule made the
  result worse, because the greedy proposer only offers the best-by-train candidate
  each generation and never falls through to the next one.
- Seed 42 is the only place dev-aware ordering helped (8 vs 7 for `no_selection`), and
  seed 2026 goes the other way (5 vs 6). One example is 0.125.

## What the pool can distinguish

- Any real proposer versus none (`no_mutation` 3/8) and versus score-free choice
  (`random_selection` 3-5/8): the gap is large relative to one example.
- Whether a proposer discovers the four true positive cue words.

## What the pool cannot distinguish

- Whether dev-based rejection or dev-aware ranking helps. Dev and train are
  perfectly aligned on the winning path, so there is no overfitting for the dev split
  to catch; the only observed dev-regression event is the `bias:+1` case, and there
  rejection hurts.
- Anything about generalisation. Held-out and dev share the same 8 cue words and
  the same templates as train; held-out never contains an unseen cue. Rejection could
  only matter if train-good/dev-bad candidates existed, and the task has no
  spurious features.
- Statistical differences of any size: 8 held-out examples, one toy pool, seeds that
  reshuffle the same rows, a single deterministic proposer.

## Suggested follow-ups (not implemented)

- A pool with distractor tokens (words correlated with the label in train but not in
  dev/held-out) so that dev-aware selection has something to reject.
- A proposer that falls through to the next-ranked candidate after a rejection, so
  `bias:+1` does not stall a run.
- More held-out rows and more than one underlying pool before reading any ablation
  difference as anything other than noise.

## Status of a real-model run

A run of `python3 -m rsi_framework --provider llm --model Qwen/Qwen2.5-0.5B-Instruct
--device auto` was attempted on 2026-09-29 and could not be performed in that
environment. The egress policy returned HTTP 403 to CONNECT for
`download.pytorch.org` (CPU torch wheels) and `huggingface.co` /
`cdn-lfs.huggingface.co` (model weights), so no weights could be fetched and no
smaller open model was reachable either. No LLM result exists, nothing was
simulated, and the README status row `Real-model RSI evidence: Not yet recorded` is
unchanged. Separately, reading `LocalTransformersProvider` shows that its prompt
lists only the current keywords (no task data, allowed vocabulary, or chat
template) and that the CLI does not restrict the proposer's vocabulary unless
`provider_keyword_vocabulary` is set in the config; a first real run will likely
have most proposals rejected or uninformative. That is untested speculation until a
run is recorded.
