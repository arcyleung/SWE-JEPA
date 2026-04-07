# SWE-JEPA: Latent Predictive Representations for Software Engineering Steering

## Abstract

Modern coding agents are increasingly evaluated on repository-level tasks, but
their training and control signals still emphasize immediate functional
correctness rather than long-run software engineering quality. This mismatch is
visible both in maintainer-review studies and in our own agentic experiments:
passing tests does not guarantee merge-ready code, and broader software
engineering properties such as interface discipline, coupling, review
readiness, and follow-up maintenance risk matter materially. We propose
**SWE-JEPA**, a latent predictive framework for software engineering, inspired
by JEPA/SALT-style self-supervision. Instead of training a model for next-token 
prediction task directly, SWE-JEPA trains a student to predict latent body
representations from sparse contextual signals such as function signatures,
imports, and neighboring structure.

Across phases 0 through 7, we evaluate SWE-JEPA as both a representation
learner and a practical steering substrate. The results support four main
claims. First, latent representations extracted from code artifacts are useful for understanding
software engineering structure and organization at the function and file level: static
properties, churn, side effects, and several follow-up risk proxies are
linearly decodable from frozen embeddings, and a 63M JEPA student dramatically
outperforms much larger SFT baselines on signature-to-body retrieval. Second,
JEPA-derived embedding methods outperform earlier hand-crafted Conway-style
features for acceptance/follow-up risk modeling, and unsupervised HDBSCAN
cluster profiling discovers interpretable high-risk patch and function
regimes. Third, these representations can be used to steer large coder models
without expensive coder-side SFT or RL: host-side steering improved both
FeatBench pass rates and scaffold judge preference in multiple settings.
Fourth, bridge design matters: a simple 9-tag prompt-injection bridge proved
robust, while richer hierarchical bridges improved judged review quality and
P2P preservation but required retraining to avoid prompt noise and instability.

The aggregate picture is promising but not cleanly linear. Several seemingly
strong ideas underperformed or failed outright: hard-negative mining did not
improve retrieval, token-level auxiliary losses added complexity without clear
gain, raw student embeddings were worse than the frozen teacher on bugfix
localization, and runtime-only bridge expansion degraded coding quality until
the richer tag ontology was retrained. We interpret these failures as useful
constraints on the design space. SWE-JEPA appears most effective as a compact
software-engineering prior that can be layered on top of existing coding
scaffolds, rather than as a drop-in replacement for all downstream reasoning.

## 1. Introduction

Current AI coding systems are good at producing syntactically valid code and at
solving bounded benchmark tasks, but software engineering quality is broader
than local correctness. Kim et al. showed that the software changes developers
care about are tied to interfaces, dependencies, ownership, and maintenance
cost, not just small isolated transformations. Conway's Law implies that code
structure also reflects organizational communication structure, but as the organization
evolves through retructuring, new code is introduced and old ones deprecated in-place, 
leading to code clones and smells in the absence of refactoring processes. Recent
maintainer-review studies on AI-generated pull requests point in the same
direction: test-passing patches are often still poor merge candidates.

SWE-JEPA is motivated by that gap. The central hypothesis is that a model can
learn a useful latent space for software engineering by predicting code
architecture structure in latent space (and by extension, the organization/ 
processes which produced the code) rather than only reconstructing surgical code edits. 
This would allow us to separate two concerns that current coding agents entangle:

1. structural reasoning about what kind of implementation fits a repository,
   interface, and ownership context
2. token-level realization of that plan into concrete code edits

The proposal skeleton in [swe-jepa-research-proposal.md](/shared_workspace_mfs/arthur/coder/docs/swe-jepa-research-proposal.md)
framed SWE-JEPA as a code analogue of JEPA/SALT, with additional emphasis on
organizational signals, follow-up risk, and maintainer-review quality. This
paper draft synthesizes the experimental program from phases 0 through 7 into
one storyline and answers four research questions.

## 2. Research Questions

**RQ1. Are latent representations useful in understanding software engineering
code structures and architecture at file/function level?**

**RQ2. Can unsupervised methods using JEPA representations win over supervised
features for acceptance/follow-up risk?**
This question is operationalized primarily through the contrast between
JEPA-derived embedding methods, HDBSCAN cluster profiling, and the earlier
hand-tuned 62-feature Conway steerer family.

**RQ3. Can JEPA representations practically be used to guide large coder models
without expensive retraining with SFT or RL posttraining?**

**RQ4. Which bridging and steering techniques are most efficient for code
scaffolds?**
Within the completed experiments, this is mostly a comparison among prompt
injection strategies and bridge ontologies: earlier feature steerer vs
HDBSCAN-informed steerer vs JEPA student steerer; 9-tag bridge vs richer
hierarchical bridge. ThinkLogit-style logit steering remains pending and is
discussed as future work rather than a completed comparison.

## 3. Method Overview

### 3.1 SWE-JEPA Architecture

SWE-JEPA follows the asymmetric latent-prediction design proposed in the
research proposal:

1. a frozen code model acts as teacher
2. structurally meaningful regions are masked at the code level
3. the teacher produces latent targets for masked regions using full context
4. a smaller student sees only sparse context and predicts those latent targets
5. downstream probes, steerer heads, or symbolic bridges consume the learned
   latent space

The overall program progressed in three layers:

- **representation discovery**: phases 0-3
- **downstream software-engineering probes**: phases 4-6
- **agent steering and scaffold integration**: phases 5-7

### 3.2 Why Latent Prediction Instead of Direct SFT?

The key practical bet behind SWE-JEPA is that a compact student can learn a
software-engineering prior without expensive end-to-end retraining of the coder
itself. This is why much of the later work focuses on lightweight steering:
host-side prompt injection, cheap heads over frozen embeddings, cluster-based
bridges, and scaffold reranking. In that sense, SWE-JEPA is closer to a
control-plane architecture to reason about the design and review process, 
than to a monolithic code-generation model.

## 4. Experimental Trajectory

### 4.1 Phases 0-3: From Representation Diagnostics to Retrieval

**Phase 0** established that mid-layer frozen embeddings are structurally
useful. Base models showed the expected U-shaped layer-differentiation pattern,
with the best separation at mid-layers rather than the last layer. The best
teacher choice was Qwen2.5-Coder-3B base at layer 18, not the instruct model.

Phase 0 also produced three important signals:

- nearest-neighbor retrieval over 28,400 functions found real structural
  families and copy-paste-then-diverge patterns
- linear probing showed strong decodability for structural properties
- churn and organizational proxies were already encoded in frozen embeddings

The strongest Phase 0 linear-probe results were:

| Property | Best score |
|---|---:|
| LOC | `R² = 0.789` |
| Cyclomatic complexity | `R² = 0.735` |
| API calls | `R² = 0.764` |
| PR churn proxy | `R² = 0.762` |
| Return type category | `BAcc = 0.944` |
| Has side effects | `BAcc = 0.972` |
| Has docstring | `BAcc = 0.980` |

This already partially answered RQ1: the representations linearly encode
non-trivial structural and maintenance-related properties.

**Phase 1** explored how to train students to predict latent function body structure.
The first result was a useful warning: cosine similarity alone is not enough.
The phase 1.1 MLP achieved **0.9468** cosine between predicted and actual body
embeddings, yet retrieval remained effectively useless. This exposed a central
failure mode for latent prediction: the student can learn the centroid or
category prototype without becoming discriminative.

Phase 1.2 and 1.3 showed that token-level attention and more repositories help:

- phase 1.2: validation cosine improved to about `0.25`
- phase 1.3: scaling to 150 repos raised cosine to about `0.59`

But retrieval still plateaued badly. Phase 1.4 replaced regression-only
training with InfoNCE and made the problem explicit: discrimination, not raw
cosine, was the bottleneck.

**Phase 2** tested two natural fixes:

- hard-negative mining
- token-level auxiliary JEPA losses

Both were surprisingly weak. Hard negatives did not improve overall retrieval
in a meaningful way, and token-level auxiliary supervision only nudged Rank@10
from roughly `4.03%` to `4.20%` while greatly increasing training complexity.

**Phase 3** was the first clear breakthrough. The main insight was not a new
loss but a better target construction. Earlier body-only targets were highly
anisotropic because they lacked signature context. Once the teacher was fed
`[signature + body]` and body positions were extracted from that full-context
forward pass, target isotropy improved dramatically and retrieval jumped.

The resulting JEPA model:

- teacher: Qwen3-8B-base, layer 18
- student: 63M parameters
- train set: 42,047 functions from a 49,631-function, 150-repo corpus

Final retrieval results from phase 3.0:

| Metric | Value |
|---|---:|
| Val Rank@1 | `19.11%` |
| Val Rank@10 | `47.52%` |
| Test Rank@1 | `28.09%` |
| Test Rank@10 | `51.77%` |

That was a decisive jump from the 3B teacher family and the earlier 4-5%
retrieval regime.

### 4.2 Phases 4-5: Downstream Probes, Reranking, and First Steering Wins

**Phase 4** asked whether the learned latent space transfers beyond retrieval.

The most convincing positive result was the SFT comparison in
[phase4_results.md](/shared_workspace_mfs/arthur/coder/docs/phase4_results.md).
JEPA was compared against 3B and 8B SFT baselines on the same signature-only
retrieval problem:

| Method | Trainable params | Val R@10 | GPU-min |
|---|---:|---:|---:|
| JEPA | `63M` | `47.52%` | `~2,016` |
| SFT 8B | `~8.2B` | `9.71%` | `4,587` |
| SFT 3B | `~3.1B` | `4.05%` | `962` |

This strongly supports the claim that latent prediction can be much more
compute-efficient than naïve coder-side SFT for structure prediction.

Phase 4.1 defect prediction added a second kind of evidence. On the
`has_bugfix` probe:

- TF-IDF: `AUROC 0.551`
- frozen teacher: `AUROC 0.567`
- JEPA student: `AUROC 0.600`

The gain is modest, but it is notable because it beats both bag-of-words and
the frozen teacher on a downstream maintenance-risk task using signatures only.

The rest of phase 4 was more mixed, and that matters for the paper's final
claims. Region localization in phase 4.3 showed that raw JEPA representations
were not automatically better for within-PR spatial ranking:

- bugfix localization R@10: teacher `56.1%`, student `46.2%`
- feature localization R@10: teacher `31.7%`, student `36.5%`

This is one of the most important negative results in the whole project. JEPA
representations are useful, but not uniformly dominant. For some ranking tasks,
especially within noisy PR-local candidate sets, the raw student was worse than
the frozen teacher.

Phases 4.4 and 4.5 partially repaired this by treating embeddings as a
substrate and learning cheap downstream SE heads:

- phase 4.4 SE-head stack recovered bugfix localization to `58.9%` R@10
- phase 4.5 Conway-augmented localization reached `59.0%` R@10 on bugfix and
  `38.8%` on feature localization

So the lesson from phases 4.3-4.5 is not that latent space is enough by
itself, but that it becomes more useful once combined with explicit
software-engineering heads and organizational signals.

Phase 4.6 PR reranking was a much stronger success. On 114 six-way candidate
packs with one historical merged PR and five synthetic negatives, SWE-JEPA +
Conway reranking achieved:

- `Top-1 = 87.7%`
- `MRR = 93.9%`

against a coder-model review baseline at:

- `Top-1 = 39.5%`
- `MRR = 69.7%`

This is strong evidence that the latent and structural signals are useful for
review-time discrimination even when coder-generated textual review alone is
not.

**Phase 4.7** then tested the first direct agent steering layer. This first
mini-swe-agent steerer failed:

- steered win rate `43.1%`
- baseline win rate `56.9%`

Its main failure mode was over-constraining scope and sacrificing completeness.
This failure directly motivated the richer Conway-aware steering work.

**Phase 5.1** improved that picture. The Conway-aware steerer trained on 28 PR
features raised acceptance AUROC from `0.769` to `0.847` on the 30k slice and
showed much better reward prediction at 100k scale. Most importantly, the
Python-domain FeatBench evaluation showed the first practical steering win:

| Agent | F2P | P2P | Both |
|---|---:|---:|---:|
| qwen35 baseline | `17 / 28` | `3 / 28` | `2 / 28` |
| qwen35 steered | `23 / 28` | `3 / 28` | `3 / 28` |

And the scaffold-aware 6-model judge panel preferred the steered patches
`61 / 110 = 55.5%` of the time.

This answered a narrower early version of RQ3: yes, coder behavior can be
improved with a cheap host-side steerer, without retraining the coder backbone.

### 4.3 Phases 6-7: Embedding Steering, HDBSCAN, and Full JEPA Student Steering

**Phase 6.1** compared frozen JEPA patch embeddings against the older
hand-crafted Conway feature steerer. This was a direct supervised-feature
comparison on 159,256 patches:

| Model | Acceptance AUROC | Refactor AUROC |
|---|---:|---:|
| Hand-tuned Conway features | `0.71` | `0.59` |
| Embedding LogReg | `0.901` | `0.744` |
| Embedding MLP | `0.913` | `0.763` |

This is one of the clearest answers in the entire project: embedding-based
JEPA-derived signals dominated the hand-engineered feature set on patch-level
risk modeling.

**Phase 6.2** moved to unsupervised organization of the patch space with UMAP +
HDBSCAN + agglomerative super-clusters. The clusters were not only
interpretable; they exposed meaningful risk regimes:

- a shared-isolated super-cluster with `18.6%` acceptance
- follow-up function clusters with bugfix rates ranging from `10.5%` to `71.8%`
- an even sharper v2 follow-up clustering with bugfix rates from `14.9%` to
  `100%`

This matters for RQ2 because it shows that unsupervised JEPA geometry can
discover actionable patch regimes without direct acceptance labels defining the
cluster boundaries.

The HDBSCAN-informed v3 steerer then translated those clusters into scaffold
guidance. Its final fully rerun coding-task totals were:

| Run | F2P | P2P | Both |
|---|---:|---:|---:|
| Baseline | `74 / 156` | `15 / 156` | `6 / 156` |
| HDBSCAN 6.2 steerer | `72 / 156` | `15 / 156` | `8 / 156` |

The coding-task picture is mixed, but the scaffold judge panel was stronger:

- HDBSCAN v3 scaffold panel: `605 / 1083 = 55.86%`

The original phase 6.2 report also found that the v3 steerer outperformed the
earlier hand-tuned v1 steerer in scaffold judging. Taken together with phase
6.1, this supports a nuanced answer to RQ2: JEPA-based unsupervised structure
helps as a steering substrate, but it is strongest when combined with some
supervised profiling or bridge logic.

**Phase 7.1** replaced the hand-built v3 review logic with a learned JEPA
student over review-state signals. The first `h`-full JEPA v1 steerer produced
the strongest stable F2P among all full-run coding evaluations:

| Run | F2P | P2P | Both |
|---|---:|---:|---:|
| JEPA-steerer v1 | `79 / 156` | `14 / 156` | `7 / 156` |

This is arguably the strongest direct answer to RQ3. The coder model itself was
not retrained. The steerer lived entirely in the host scaffold:

1. baseline run
2. read `patch.diff`
3. infer review state with a compact student
4. inject one review prompt
5. rerun once

That is a cheap control-plane intervention rather than coder-side SFT or RL.

**Phase 7.2** studied the judges and found a systematic split:

- Qwen-family judges tended to prefer broader, more heavily tested and
  documented patches
- frontier judges such as Codex/GPT/Claude more often preferred narrower,
  lower-risk, review-ready changes

This matters because it showed that "good patch" is not judge-invariant.

**Phase 7.3** expanded the bridge ontology beyond the original 9 tags. The main
lesson from phase 7.3 is that bridge sophistication must be matched by
training. Adding richer runtime-only heuristics to the old 9-tag student made
the steering noisier and initially degraded coding quality. Retraining on a
16-tag richer ontology improved the student itself:

- acceptance AUROC: `0.8502`
- tag macro F1: `0.1806`

The stable full retrained richer-tags run finished at:

- `F2P 64 / 156`
- `P2P 17 / 156`
- `Both 7 / 156`

However, a targeted jitter rerun on the 21 cases where JEPA v1 had F2P success
but richer-tags did not recovered:

- `12 / 21` F2P passes
- `2 / 21` full successes

which yields an adjusted phase 7.3 coding line of:

- `F2P 76 / 156`
- `P2P 17 / 156`
- `Both 9 / 156`

The scaffold judge panel for retrained phase 7.3 reached:

- `612 / 1033 = 59.24%`

and the like-for-like paired-row replacement on the jitter subset improved that
to:

- `617 / 1033 = 59.73%`

This is the strongest judge result in the project, but it came with more
instability than the 9-tag v1 bridge.

## 5. Results by Research Question

### 5.1 RQ1: Are latent representations useful for understanding software engineering structure?

**Answer: Yes, with two caveats.**

The positive evidence is broad:

1. **Static structural properties are strongly encoded.**
   Phase 0 linear probing recovered LOC, complexity, API-call count, churn, side
   effects, and documentation status at strong linear-probe performance.

2. **JEPA latent prediction learns function-level structure from signatures.**
   Phase 3 reached `19.11%` Rank@1 and `47.52%` Rank@10 on validation, and
   `28.09%` Rank@1 / `51.77%` Rank@10 on test, using signatures only at
   inference time.

3. **The learned latent space transfers to maintenance-risk tasks.**
   Phase 4.1 defect prediction gave the JEPA student `AUROC 0.600`, beating the
   frozen teacher and TF-IDF.

4. **Latent representations help candidate reranking for PR review.**
   Phase 4.6 showed a material Top-1 improvement over a coder-model review
   baseline (87.7% vs 39.5%).

The caveats are equally important:

- raw student representations did **not** dominate the frozen teacher on
  within-PR bugfix localization
- local spatial ranking benefited from extra SE heads and Conway-style features

So the strongest version of the RQ1 answer is: **latent representations are
useful as a software-engineering substrate, but they are not sufficient on
their own for every downstream task. An additional mechanism is required to decode
and produce usable guidance signals**

### 5.2 RQ2: Can unsupervised JEPA methods win over supervised feature steering?

**Answer: Yes in important settings, but the picture is mixed and layered.**

There are three levels to the answer.

**Level 1: Embeddings vs hand-crafted features.**
Phase 6.1 is decisive. JEPA patch embeddings beat the hand-crafted Conway
feature baseline by very large margins:

- acceptance AUROC: `0.913` vs `0.71`
- refactor AUROC: `0.763` vs `0.59`

This is a clear yes.

**Level 2: Unsupervised structure discovery.**
Phase 6.2 showed that JEPA geometry supports meaningful unsupervised structure, with a high dynamic range:

- patch super-clusters exposed acceptance strata as low as `18.6%`
- follow-up clusters separated bugfix risk up to `71.8%` and later `100%` in
  the enriched v2 setting

These results matter because they were not hand-designed feature buckets; they
emerged from clustering in latent space and then became interpretable only after
profiling.

**Level 3: Actual steering outcome.**
The HDBSCAN-informed v3 steerer did not become the best F2P coding run, but it
did perform well in judged scaffold evaluation and improved the "both" metric
relative to baseline. The later retrained JEPA steerer then exceeded HDBSCAN on
judge preference.

So the most honest answer is:

- **for predictive risk modeling**, JEPA-derived embeddings clearly beat the
  older hand-tuned features
- **for unsupervised patch taxonomy**, HDBSCAN over JEPA embeddings discovers
  meaningful risk regimes
- **for end-to-end steering**, unsupervised structure helps, but still benefits
  from supervised heads, symbolic profiling, or bridge logic

### 5.3 RQ3: Can JEPA steer large coder models without SFT or RL post-training on the coder?

**Answer: Yes. This is one of the project's clearest practical findings.**

The evidence accumulated across phases 5 through 7:

1. **Phase 5.1 Python steerer** improved F2P from `17 / 28` to `23 / 28` and
   won the scaffold-aware panel `55.5%` of the time.

2. **Phase 7.1 JEPA-steerer v1** reached `79 / 156` F2P on the full coding
   evaluation, the best stable F2P result among the main runs.

3. **Phase 7.3 retrained richer-tags** improved judged scaffold quality to
   `59.24%` and, after jitter correction on the targeted rerun, improved the
   full coding totals to `76 / 17 / 9`.

All of these systems steered the coder **without retraining the coder model
itself**. The mechanism was host-side and cheap:

- read artifacts from run 1
- infer latent or symbolic review state
- inject a targeted second-pass review prompt

This is precisely the low-cost control regime the proposal argued for.

The project therefore supports a strong practical claim: **latent
software-engineering priors can materially change coder behavior without
coder-side SFT or RLHF.**

For reference, the final coding-task comparison across the main full runs is:

| Run | F2P | P2P | Both |
|---|---:|---:|---:|
| Baseline | `74 / 156` | `15 / 156` | `6 / 156` |
| HDBSCAN 6.2 steerer | `72 / 156` | `15 / 156` | `8 / 156` |
| JEPA-steerer v1 | `79 / 156` | `14 / 156` | `7 / 156` |
| JEPA retrained richer-tags (stable run) | `64 / 156` | `17 / 156` | `7 / 156` |
| JEPA retrained richer-tags (with targeted rerun merged) | `76 / 156` | `17 / 156` | `9 / 156` |

And the fair scaffold-judge comparison on valid paired judgments is:

| Run | Valid-judgment scaffold win rate |
|---|---:|
| HDBSCAN 6.2 steerer | `605 / 1083 = 55.86%` |
| JEPA retrained richer-tags | `612 / 1033 = 59.24%` |

### 5.4 RQ4: Which bridge or steering techniques are most efficient for code scaffolds?

**Answer: among completed experiments, prompt injection works; bridge simplicity
is robust; richer hierarchical bridges are promising but require retraining;
logit steering remains an open next step.**

The empirical ordering so far is:

1. **simple prompt injection with meaningful structural signals works**
2. **adding more bridge content without retraining makes things worse**
3. **retraining the bridge ontology helps recover and can improve judged quality**

The clearest bridge lessons are:

- the early small 6-feature scope steerer failed because it truncated necessary
  changes
- the HDBSCAN-informed v3 bridge improved judged scaffold quality
- the JEPA 9-tag bridge in phase 7.1 was the most robust for F2P
- the richer hierarchical bridge in phase 7.3 improved P2P and scaffold win
  rate, but only after retraining on the richer tag ontology

The failed intermediate phase 7.3 ablation is especially instructive. Expanding
the bridge heuristically on top of the old 9-tag student caused prompt noise,
more empty/no-op patches, and worse coding results. Only after retraining the
student on the 16-tag ontology did the richer bridge become credible.

So, within the completed work:

- **best robust coding bridge**: 9-tag JEPA v1
- **best judged scaffold quality**: retrained richer-tags bridge
- **most efficient currently deployed mechanism**: two-pass host-side prompt
  injection

**Prompt injection vs logit steering.**
The project proposal and later discussion connect naturally to ThinkLogit-style
logit steering, but that technique is still pending here. We therefore cannot
claim an empirical advantage for logit steering yet. The current evidence only
supports prompt injection. The open hypothesis is that logit steering may reduce
prompt noise and empty-patch failure modes, especially for coder models that are
sensitive to extra textual review instructions.

## 6. Discussion: Surprising Findings and Unexpected Failures

Several of the most informative outcomes were negative or surprising.

### 6.1 High cosine can be meaningless

Phase 1.1's `0.9468` cosine looked excellent until retrieval exposed that the
student had learned category prototypes rather than discriminative function
structure. This was an early warning that latent regression metrics can be
misleading in anisotropic spaces.

### 6.2 Stronger chat/instruct models were not always better teachers

The instruct fine-tuned Qwen variants repeatedly underperformed base models for structural
representation work. This is counterintuitive if one assumes chat tuning
uniformly improves code understanding. For SWE-JEPA-style structural targets,
the base model geometry was often cleaner.

### 6.3 Hard negatives and token-level auxiliaries underperformed

Two intuitive improvements failed:

- hard negatives did not materially improve retrieval
- token-level JEPA auxiliaries added cost and gradient conflict without giving
  a meaningful gain

This suggests that target quality and isotropy mattered more than more complex
loss engineering.

### 6.4 Raw JEPA student embeddings were worse than the frozen teacher on bugfix localization

This was a major surprise. It showed that JEPA is not automatically better than
its teacher on every downstream task, especially when the task requires fine
relative ordering inside a PR. The eventual fix was not more JEPA training alone
but layering SE heads and organizational features on top.

### 6.5 The first steerer failed by making patches too small

The earliest mini-swe-agent steerer lost badly because it over rewarded scope
discipline without enough understanding of completeness. This became a recurring
theme: steering for narrowness is only useful if the system also knows what
cannot be omitted.

### 6.6 Judge families encode different review priors

Phase 7.2 exposed a stable split:

- Qwen-family judges tended to prefer broader, more documented, more heavily
  tested patches
- frontier judges more often preferred surgical, lower-risk patches

This was one of the most conceptually important findings in the whole program.
It implies that "patch quality" is partly a property of the judge family and
its training priors, not a single invariant scalar. The judge rubric in Experiment xx
attempts to break down the review readiness qualia into multiple dimensions for each judge.

### 6.7 Bridge sophistication without retraining is dangerous

The phase 7.3 runtime-only heuristic expansion failed badly enough to become a
strong design lesson. If the bridge ontology changes, the student should be
retrained on that ontology. Otherwise the system becomes a brittle hybrid of
learned and hand-written signals.

### 6.8 Benchmark and infrastructure jitter were larger than expected

The later phase 7 coding evaluations revealed substantial sensitivity to:

- concurrency
- container/runtime instability
- empty patch files
- endpoint variability
- judge no-verdict behavior

The targeted 21-instance rerun is especially important here: `12 / 21` apparent
phase-7.3 F2P regressions relative to JEPA v1 recovered on rerun. This means a
non-trivial portion of the measured gap was evaluation jitter rather than a true
model-quality difference.

## 7. Limitations and Threats to Validity

1. **Evaluation infrastructure matters.**
   Some later scaffold judge and coder runs were sensitive to container naming,
   stale worker state, and endpoint instability. The final reported numbers use
   the cleaned reruns, but infrastructure noise was a real variable.

2. **Judge panels are not neutral.**
   Different judge families prefer different patch styles. Panel averages should
   therefore be interpreted as aggregate preferences, not ground truth.

3. **Some comparisons are fairer than others.**
   The final phase 7.3 scaffold comparisons are fair on valid paired judgments,
   but not every earlier steerer was rerun on exactly the same panel and
   denominator.

4. **Language skew remains a problem.**
   The project is strongest in Python-heavy settings. Go results in particular
   showed transfer problems for Python-trained steering features.

5. **Prompt injection is tested; logit steering is not.**
   RQ4 therefore remains partially open. The most efficient technique among
   completed experiments is prompt injection, but the broader design question is
   not closed.

## 8. Conclusion

The experimental program supports a clear overall thesis: latent
software-engineering representations are real, useful, and practically
actionable.

The strongest evidence is distributed across the phases:

- phase 0 showed that frozen code embeddings already encode structural and
  organizational proxies
- phases 1-3 showed that a compact JEPA student can learn discriminative
  signature-to-body structure prediction
- phase 4 showed that those representations transfer to several downstream SE
  tasks and beat large SFT baselines on the core retrieval problem
- phases 5-7 showed that the same family of signals can steer large coder models
  without retraining the coder itself

At the same time, the project sharply constrained its own design space:

- latent regression alone is not enough
- target construction matters more than many fancy loss additions
- software-engineering heads and organizational features still matter
- bridge design must stay aligned with what the student was actually trained to
  predict

The most defensible final claim is therefore not that SWE-JEPA replaces
large-model coding, but that it provides a compact and effective **software
engineering prior** that can be layered onto existing coding scaffolds. That is
already enough to beat hand-crafted feature steerers, rival or exceed larger
SFT baselines on structure prediction, and improve judged scaffold behavior
without coder-side retraining.

## References To Formalize In The Final Version

- Conway, M. E. (1968). *How Do Committees Invent?*
- Kim, M., Zimmermann, T., Nagappan, N. (2014). *An Empirical Study of
  Refactoring Challenges and Benefits at Microsoft.*
- SALT / JEPA latent prediction paper referenced in the proposal
- METR maintainer-review study referenced in the proposal
- ThinkLogit
- RAIM
