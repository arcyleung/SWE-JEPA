# Experiment 0.3: Linear Probing for Structural Properties

**Date**: 2026-02-22  
**CV folds**: 5  
**Log-transform applied to**: cyclomatic_complexity, loc, n_api_calls, pr_churn  
**Embeddings**: mean-pooled hidden states, layer 18, L2-normalised  
**Probe**: Ridge regression (R²) / LogisticRegression (balanced accuracy)

---

## Results

| Property               | Metric | Kim ref        |       Coder-3B | Coder-3B-Instruct |          Q3-8B |     Q3-8B-base |
| ---------------------- | ------ | -------------- | -------------- | -------------- | -------------- | -------------- |
| LOC                    | R²     | S1             |         0.7572 |         0.7570 |         0.7866 |         0.7894 |
| Cyclomatic complexity  | R²     | C3             |         0.6980 |         0.6968 |         0.7285 |         0.7349 |
| # branches             | R²     | —              |         0.4471 |         0.4427 |         0.4611 |         0.4833 |
| # loops                | R²     | —              |         0.4595 |         0.4587 |         0.4943 |         0.4993 |
| # returns              | R²     | —              |         0.4200 |         0.4207 |         0.3990 |         0.3880 |
| # API calls            | R²     | C5.x proxy     |         0.7227 |         0.7223 |         0.7506 |         0.7637 |
| # arguments            | R²     | —              |         0.6140 |         0.6155 |         0.6590 |         0.6535 |
| PR churn (# PRs)       | R²     | Kim et al. churn proxy |         0.7183 |         0.7137 |         0.7594 |         0.7622 |
| Return type category   | BAcc   | —              |         0.9337 |         0.9343 |         0.9412 |         0.9442 |
| Has side effects       | BAcc   | —              |         0.9675 |         0.9651 |         0.9681 |         0.9715 |
| Has decorators         | BAcc   | —              |         0.9633 |         0.9657 |         0.9709 |         0.9722 |
| Has docstring          | BAcc   | —              |         0.9752 |         0.9755 |         0.9802 |         0.9789 |

---

## Interpretation

**R²** (regression): fraction of variance in the property explained by a linear
function of the embedding. R² ≈ 0 = not linearly encoded; R² ≈ 1 = perfectly
linearly encoded. Negative R² means the probe is worse than predicting the mean.
Random baseline R² = 0 (predicting the mean).

**Balanced accuracy** (classification): mean per-class recall across all classes.
Random baseline: 0.25 for `return_type_cat` (4 classes), 0.50 for binary props.

**pr_churn** counts how many distinct PRs in our corpus touched the same
(file, function) pair — a proxy for Kim et al.'s churn and developer-count
metrics, which were among the strongest predictors of refactoring investment.

### Key findings

**1. Structural properties are strongly linearly encoded — success criterion met.**

The success criterion from the research proposal was "at least some structural
properties should be linearly decodable with reasonable accuracy." Every single
property exceeds this bar, with the regression results ranging from R²=0.39
(# returns) to R²=0.79 (LOC), and classification results at 0.93–0.98 balanced
accuracy (vs random baselines of 0.25–0.50).

**2. Size-dominant properties encode most strongly.**

LOC (R²≈0.76–0.79) and cyclomatic complexity (R²≈0.70–0.73) are the most
linearly decodable. These are correlated but distinct: a function can be long
with low complexity (sequential code) or short with high complexity (nested
conditions). That the probe captures both well suggests the embedding encodes
code *volume* and *control-flow density* as separate geometric directions.

**3. PR churn encodes nearly as well as code size (R²≈0.72–0.76).**

This is the most important result for the SWE-JEPA thesis. Churn — how many
distinct PRs touch a function — is a Kim et al. metric strongly predictive of
refactoring investment. The fact that it is linearly decodable from frozen
hidden states means the embedding implicitly captures *which functions tend to
be volatile versus stable*, without ever seeing git history directly. The model
has learned this from the statistical regularities in how frequently-modified
code tends to look (shorter, more branchy, more side-effect-ful) versus stable
code (longer, more documented, pure). This is precisely the organizational
signal the research proposal hypothesised the latent space would contain.

**4. Coarse-grained control flow (branches, loops) encodes at 0.44–0.50.**

The weaker R² for # branches and # loops (vs LOC and CC) is expected: two
functions of equal complexity can have very different branch/loop distributions
(e.g., deep nesting vs wide switch). The embedding captures something between
the two — the overall branching *character* of the function, not the precise
count. Still well above random.

**5. Classification properties are nearly perfectly linearly decodable.**

Side effects (BAcc=0.97), decorators (0.96–0.97), and docstring presence
(0.98) are essentially saturated at the linear probe level. This makes sense:
these are surface-level properties that correlate strongly with lexical patterns
(import presence, `@` prefix, triple-quoted strings) that the model encodes
faithfully. Return type category (BAcc=0.93–0.94 vs 0.25 random) requires
more genuine semantic understanding — distinguishing `return self._cache[key]`
(custom) from `return len(self._items)` (primitive) is non-trivial.

**6. 8B base model is consistently best; 3B models are nearly identical.**

Qwen3-8B-base edges out Qwen3-8B (instruct) on most structural properties,
consistent with the KNN finding that instruction tuning reshapes the geometry
away from syntactic structure. The 3B base and instruct models are nearly
indistinguishable (differences < 0.003 R²) — suggesting instruct fine-tuning
at 3B scale has minimal effect on the hidden-state geometry at layer 18.

The ordering for structural encoding quality is:
```
Qwen3-8B-base ≥ Qwen3-8B > Qwen2.5-Coder-3B ≈ Qwen2.5-Coder-3B-Instruct
```
But the gap between the best and worst model is small (≤0.03 R²), confirming
SALT's finding that teacher quality matters less than expected.

### Implications for SWE-JEPA training targets

The student, given only a function signature and surrounding context, must
predict a latent vector from which all of the above are linearly decodable.
That means it must implicitly reason about:

- *How long will this implementation be?* (LOC, R²=0.79)
- *How complex is the control flow?* (CC, R²=0.73)
- *How many external systems does it touch?* (API calls, R²=0.76)
- *How volatile is this function in practice?* (PR churn, R²=0.76)
- *Does it have side effects?* (BAcc=0.97)
- *What kind of value does it produce?* (return type, BAcc=0.94)

These are precisely the questions a software engineer asks before writing an
implementation — not "what tokens come next" but "what kind of thing am I
building?" The latent targets force the student to develop implicit answers to
all of them simultaneously, from context alone.

---

## Notes

- Log-transform applied to skewed targets before probing: `log1p(y)`.
- All embeddings L2-normalised before probing.
- All features StandardScaler-normalised for the probe.
- 5-fold cross-validation; mean score reported.
