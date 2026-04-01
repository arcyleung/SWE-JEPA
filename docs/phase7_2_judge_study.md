# Experiment 7.2 — Judge Preference Study

## Motivation

The phase 7.1 scaffold judging exposed a consistent split in model preferences.
The two Qwen judges leaned against the JEPA-steered patch, while the frontier
judges more often preferred it. This is interesting beyond the immediate
FeatBench result because it suggests that "good patch" is not a judge-invariant
concept. Different judge families appear to encode different software review
priors.

This connects directly back to the framing in
`docs/swe-jepa-research-proposal.md` and to Kim et al., *An Empirical Study of
Refactoring Challenges and Benefits at Microsoft* (IEEE TSE, 2014). Kim et al.
argue that software quality is multi-dimensional and tied to dependencies,
defects, churn, test adequacy, and organizational ownership, not just local
correctness. If different judge families prefer different styles of patch, the
next question is not only "which judge is harsher?" but "which preference
better predicts lower long-run maintenance cost?"

## Core Observation

Current phase 7.1 scaffold judge snapshot:

| Judge | Valid | JEPA-steered wins | Win rate |
|---|---:|---:|---:|
| `qwen3.5-397b-a17b` | 121 | 60 | 49.59% |
| `qwen3-coder-480b-a35b-instruct` | 121 | 57 | 47.11% |
| `gpt-5.3-codex` | 121 | 66 | 54.55% |
| `gpt-5.4` | 121 | 64 | 52.89% |
| `claude-opus-4-6` | 101 | 53 | 52.48% |
| `MiniMax-M2.7` | 121 | 69 | 57.02% |
| `kimi-k2.5` | 121 | 67 | 55.37% |

The headline pattern is:

- Qwen judges are below parity on JEPA-steered patches.
- Codex / GPT / Claude / MiniMax / Kimi are above parity.
- This is not a small-noise effect. The split is systematic in the
  justifications and in the dimension scores.

## Direct Disagreement Evidence

On the `121` instances jointly judged by:

- `qwen3.5-397b-a17b`
- `qwen3-coder-480b-a35b-instruct`
- `gpt-5.3-codex`
- `gpt-5.4`

there are:

- `15` cases where both GPT judges chose `steered` and both Qwen judges chose
  `baseline`
- `9` cases where both GPT judges chose `baseline` and both Qwen judges chose
  `steered`

So the disagreement is asymmetric in the direction that hurts the JEPA-steered
patch.

Examples where both Qwen judges preferred baseline but both GPT judges preferred
JEPA-steered:

- `openai__openai-agents-python-1124`
- `pydata__xarray-10161`
- `modelcontextprotocol__python-sdk-722`
- `slackapi__bolt-python-1173`
- `koxudaxi__datamodel-code-generator-2420`
- `huggingface__smolagents-783`

On the `101` instances jointly judged by:

- the two Qwen judges
- `gpt-5.3-codex`
- `gpt-5.4`
- `claude-opus-4-6`

the frontier side is more pro-steered than the Qwen side on `32` instances,
while the Qwen side is more pro-steered on `24`, with `45` ties.

## What The Qwen Judges Seem To Reward

In the disagreement set where both GPT judges picked JEPA-steered and both Qwen
judges picked baseline, the Qwen justifications repeatedly praise:

- more comprehensive test coverage
- broader handling of edge cases
- more complete implementation scope
- stronger documentation
- convention-following naming / fixture layout
- "production-ready" packaging

Keyword counts across the disagreement justifications support this:

- `comprehensive`: `21`
- `test`: `20`
- `coverage`: `14`
- `complete`: `10`
- `documentation`: `7`

Representative baseline-favoring Qwen judgments:

- `modelcontextprotocol__python-sdk-722`
  - baseline preferred for adding more direct server-side tests
- `openai__openai-agents-python-1124`
  - baseline preferred for broader support and extensive tests
- `slackapi__bolt-python-1173`
  - baseline preferred because it includes tests while the steered patch does
    not
- `pydata__xarray-10161`
  - baseline preferred for documentation and broader test coverage

This suggests that the Qwen judges are strongly completeness-biased. They often
reward the patch that looks more like a full PR package:

- implementation
- tests
- documentation
- broader case coverage

even when that patch is larger or riskier.

## What The Frontier Judges Seem To Reward

The frontier judges, especially Codex and GPT-5.4, more often favor:

- lower-risk changes
- tighter scope discipline
- better review readiness
- smaller architectural blast radius
- cleaner fit to existing repo structure

On the `15` strong disagreement cases where GPT picked steered and Qwen picked
baseline, the average score deltas show the split clearly.

Qwen penalties to steered:

- `test_coverage`: about `-1.5`
- `review_readiness`: about `-1.4`
- `correctness`: about `-0.8` to `-1.2`
- `documentation`: about `-0.6` to `-1.0`

GPT rewards to steered on the same rows:

- `scope_discipline`: about `+1.5` to `+1.7`
- `correctness`: about `+1.1` to `+1.3`
- `interface_design`: about `+1.1`
- `review_readiness`: about `+1.3`

Qualitatively, Codex / GPT / Claude are much more willing to say:

- the steered patch is narrower
- the steered patch is safer
- the baseline patch is broader than necessary
- extra tests or broader coverage do not compensate for increased risk

So the frontier judges appear to reward "surgical edit quality" more than
"complete PR packaging."

## Interpretation

The most interesting hypothesis is that these are not random judge quirks. They
may reflect different training-distribution priors about what good engineering
looks like.

Possible interpretation:

- Qwen-family judges may be more influenced by open-source PR style where:
  - explicit tests are heavily rewarded
  - broader coverage is interpreted as diligence
  - documentation and completeness are strong merge signals
- Codex / GPT / Claude may be more influenced by review settings where:
  - minimizing blast radius matters more
  - smaller diffs are easier to merge safely
  - overreaching scope is penalized even if accompanied by extra tests

This is not yet a claim about proprietary versus open data. It is a hypothesis
about learned review priors. The important point is that a judge panel is not
measuring one scalar concept of code quality. It is aggregating multiple
organizationally-loaded coding norms.

That is precisely why this matters for SWE-JEPA. If the target is long-run
engineering quality, then we should not let one judge family's implicit review
culture define the whole metric.

## Relation To Kim et al. and Conway

Kim et al. argue that useful refactoring work is not just atomic code cleanup;
it is bound up with interfaces, contracts, test adequacy, ownership, and
dependency structure. Conway's Law adds that these qualities are shaped by the
communication and review structure of the organization producing the code.

The judge split observed here is a concrete evaluation-time reflection of that
idea:

- one judge family appears to value broad, heavily-tested, explanatory patches
- another appears to value concentrated, low-risk, architecture-fitting patches

Those are not merely stylistic preferences. They correspond to different views
of how engineering organizations manage risk:

- prevent regressions by expanding tests and documenting everything
- or prevent regressions by keeping edits surgical and preserving existing
  boundaries

The question for phase 7.2 is which preference better correlates with real
outcomes.

## Proposed Phase 7.2 Study

### Question

Which judge family's preferences align better with the real-world longitudinal
quality signal already available in our historical PR / follow-up data?

### Study design

For each judged patch pair:

1. Partition instances into:
   - Qwen-prefers-baseline / frontier-prefers-steered
   - Qwen-prefers-steered / frontier-prefers-baseline
   - agreement groups
2. For each side, map the corresponding patch state to historical analogues in:
   - acceptance / non-acceptance rates
   - follow-up bugfix rates
   - refactor churn
   - files-per-change / blast-radius proxies
   - ownership / co-change / Conway metrics from phase 4.x / 6.x
3. Measure whether the style preferred by Qwen or by frontier judges better
   predicts:
   - lower follow-up bugfix probability
   - lower future churn
   - better acceptance-like structural signals

### Minimal quantitative tests

- Compare average historical `bugfix_rate` of the HDBSCAN cluster chosen by the
  preferred patch.
- Compare average acceptance-head score of the preferred patch.
- Compare scope / churn / dependency proxies between Qwen-favored and
  frontier-favored winners.
- Fit a simple regression:
  - dependent variable: follow-up bugfix / acceptance outcome
  - predictors: judge-family preference, patch size, test additions, doc
    additions, cluster risk tier

### Key hypotheses

- `H1`: Qwen-preferred winners will more often be larger patches with more test
  and documentation additions.
- `H2`: Frontier-preferred winners will more often have lower cluster risk and
  narrower blast radius.
- `H3`: If Kim/Conway-style longitudinal quality is the right target, then
  frontier-preferred winners may correlate better with lower follow-up debt even
  when Qwen judges score them lower on completeness.
- `H4`: Alternatively, Qwen-preferred winners may prove better in repos where
  explicit tests and broader edge-case coverage dominate review culture.

## Why This Matters

This is not just about fixing a low Qwen win rate. It is a deeper evaluation
question:

- are we trying to optimize for "looks like a complete PR"
- or "minimizes long-run integration risk"

The answer may vary by repository and organization. That makes judge diversity a
feature, not just noise. Phase 7.2 should turn that diversity into a measurable
signal about engineering style and long-run codebase outcomes.

## Immediate Next Steps

1. Freeze the current disagreement set between Qwen and frontier judges.
2. Materialize a per-instance comparison table with:
   - judge votes
   - dimension scores
   - patch size / tests / docs deltas
   - cluster risk / acceptance predictions
3. Correlate those disagreements with historical follow-up signals from the
   phase 6 teacher data.
4. Decide whether future panel aggregation should:
   - keep all judges equal
   - weight judges by correlation with longitudinal outcomes
   - or report separate "completeness-preferring" vs "surgical-edit-preferring"
     subpanels.
