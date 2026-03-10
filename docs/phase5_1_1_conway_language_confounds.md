# Experiment 5.1.1 — Language/Framework-Level Conway Signal Confounds

**Status**: Side excursion — not part of the mainline 5.1 → 5.2 experiment sequence.
**Purpose**: Robustness analysis of the 28 Conway signals against language and framework
conventions that create systematic false positives or false negatives. Required before
extending `prs_copy` with Go, Kotlin, and other language corpora.

---

## Motivation

Several signals in the Exp 5.1 feature set fire on language syntax conventions rather
than genuine Conway friction indicators. If the corpus becomes more language-diverse
(e.g., adding Go PRs), these confounds will systematically bias the steerer toward or
away from entire language ecosystems rather than detecting real review risk.

The goal of this excursion is to:
1. Catalogue known per-language confounds in the current 28 features
2. Propose normalization strategies (language-conditional thresholds, language dummies,
   per-feature language-stratified prevalence correction)
3. Design additional features that require language-level context to interpret correctly
4. Establish a readiness checklist before onboarding new languages to `prs_copy`

---

## Known confounds in the current feature set

### `has_pub_func` — Rust false positive (high severity)

**Pattern**: `pub fn \w+` — fires on any new Rust public function.

**Confound**: In Rust, `pub fn` is the standard visibility annotation for *any* non-private
function exposed beyond the current module. ~65% of all Rust PRs in the 100k dataset
trigger this feature. The signal was designed to detect Go's capitalization convention
(`func UpperCase`) and Java/Kotlin `public` method declarations, which are genuinely
rare and represent explicit API surface expansion decisions.

**Effect**: The model conflates "Rust PR" with `has_pub_func=1`, learning a partial
language identity mapping. At inference on agent-generated Rust code this is self-consistent
(the agent also uses `pub fn`), but the causal interpretation is broken — `has_pub_func`
in Rust does not carry the same "deliberate public API expansion" signal as in Go.

**Proposed fix**: Split into `has_pub_func_go` (capitalized identifier, Go files only)
and `has_pub_func_java_kotlin` (`public` keyword, `.java`/`.kt` files), and exclude
Rust `pub fn` from the feature entirely. Keep a separate `has_rust_pub_fn` feature
with a Rust-specific baseline rate correction.

---

### `has_try_catch` — Go false negative (medium severity)

**Pattern**: `^\+.*(try|catch|except|rescue)\s*[\({\[]`

**Confound**: Go uses explicit `if err != nil { return err }` error handling — there is
no `try/catch` syntax. This means `has_try_catch` is structurally zero for all Go PRs
regardless of how carefully the error handling is written. A Go PR that wraps every
error with `fmt.Errorf("...: %w", err)` (the idiomatic chain) scores identically to
one that swallows all errors via `_ = riskyOperation()`.

**Effect**: Once Go PRs are added, the model will underestimate error-handling quality
for Go code. The `error_contract_score` compound feature inherits this gap.

**Proposed fix**: Add Go-specific error handling signals:
- `has_go_err_wrap`: `fmt.Errorf(".*%w"` — idiomatic error wrapping (positive signal)
- `has_go_err_discard`: `_ = ` on a line calling a function that returns `error` —
  error discard (strongly negative, equivalent to bare_except)
- `has_go_err_check`: `if err != nil` — explicit check (positive, but nearly universal
  in well-written Go so use prevalence-adjusted lift)

These should feed into `error_contract_score` via a Go-aware branch.

---

### `has_bare_except` — Python-only, zero for all other languages (low severity, by design)

**Pattern**: `^\+\s*except\s*:` — Python bare except clause.

**Confound**: Not a confound per se — this is intentionally Python-only. But since the
100k dataset is ~30% Python, the feature carries a strong implicit language weight.
If the corpus expands with more Rust/Go PRs, the feature becomes less informative
proportionally.

**Proposed fix**: Add language-equivalent "silent error swallow" signals for other
languages, then merge into a unified `has_silent_error_swallow` binary:
- Python: `except:` or `except Exception: pass`
- Java/Kotlin: `catch (Exception e) {}` (empty catch)
- JavaScript/TypeScript: `.catch(() => {})` or `.catch(e => {})` (empty promise rejection)
- Go: `_ = ` error discard (already proposed above)
- Rust: `let _ = ` or `.unwrap_or_default()` on `Result` without logging

---

### `has_hardcoded_cred` — Framework credential patterns missed (medium severity)

**Pattern**: `(password|secret|api_key|token|passwd)\s*=\s*["'][^"']{4,}["']`

**Confound**: Modern frameworks use environment variable helpers (`os.getenv`,
`process.env`, Spring `@Value("${...}")`) or secret managers. A PR that replaces a
hardcoded secret with `os.getenv("API_KEY")` will correctly fire `has_hardcoded_cred=0`,
which is good. But many framework-specific patterns are missed:

- Spring Boot: `@Value("${spring.datasource.password}")` could be hardcoded in
  `application.properties` — not caught by the regex which looks at `.java` source
- Rails: `config/secrets.yml` with literal values — needs manifest-level inspection
- Helm charts: `values.yaml` with literal `password:` fields — currently only `is_draft`
  catches config files at all

**Proposed fix**: Add manifest-aware credential checks in `_DEP_FILE` / `_CONFIG_FILE`
handling: scan YAML/TOML/JSON diff hunks for `password:`, `secret:`, `token:` keys with
non-variable values separately from source code.

---

### `imp_external` / `trust_boundary_crossings` — Go module proxy ambiguity (medium severity)

**Pattern**: Tree-sitter Go import parsing classifies any non-stdlib import as external.

**Confound**: In Go, internal organization packages are identified by their module path
prefix (e.g., `github.com/myorg/...`). A PR adding `github.com/myorg/auth` is an
intra-org (relative trust) import, while `github.com/aws/aws-sdk-go` is a genuine
external (trust-boundary crossing). The current tree-sitter parser cannot distinguish
these without knowing the organization's module namespace.

**Effect**: All inter-package Go imports from the same org count as `imp_external=1`,
inflating `trust_boundary_crossings` for large monorepos that naturally import internal
packages.

**Proposed fix**: Add per-repo `go.mod` module prefix detection: parse the `module`
declaration from go.mod hunks in the diff. Imports with the same prefix as the module
declaration are `imp_relative`; all others are `imp_external`. This requires a one-pass
pre-scan of the diff for manifest files before the per-file import analysis.

---

### `has_version_suffix` — false positive in test/fixture files (low severity)

**Pattern**: `(def |class |func |fn )\w*(v\d+|V\d+|_v\d+)`

**Confound**: Test files legitimately use versioned function names as test fixtures
(`test_api_v2_response`, `mock_v1_payload`). These carry none of the deprecation/migration
risk the feature was designed to detect.

**Effect**: PRs with comprehensive test suites that test multiple API versions score
`has_version_suffix=1` even when the versioning is entirely internal to test infrastructure.

**Proposed fix**: Restrict pattern to non-test files, or require that the versioned
function appears in a non-test file path.

---

## Design by Contract / Self-Checking Mechanisms

DbC features (assertions, precondition checks, invariant guards) are promising
Conway proxies because explicit contracts signal defensive programming discipline —
exactly the kind of practice that reduces followup review churn. However, they are
confounded by language build semantics:

| Language | Mechanism | Production behavior |
|----------|-----------|-------------------|
| Python | `assert expr` | **Disabled** with `python -O` — not enforced in optimized builds |
| Java | `assert condition` | **Disabled** by default (requires `-ea` JVM flag) |
| Kotlin | `require(condition)` | Enforced — throws `IllegalArgumentException` |
| Rust | `debug_assert!(expr)` | **Stripped** in `--release` builds |
| Rust | `assert!(expr)` | Enforced in all builds |
| Go | No native assert — uses explicit `if` + `panic` | Enforced |
| JavaScript | No native assert — `console.assert` is non-throwing | Not enforced |

A `has_assert` feature built on naive pattern matching would give Python `assert` the
same weight as Rust `assert!`, despite the former being a debugging aid and the latter
a hard runtime invariant. The signal is only meaningful when conditioned on:
1. Which construct (assert vs require vs invariant library)
2. The project's build configuration (release mode, JVM flags)
3. Whether the assertion is in production code vs test code

**Current conclusion**: DbC signals are not included in the current feature set due to
this build-mode confound. A cleaner proxy for "defensive programming discipline" is the
combination of `has_try_catch`, `has_raise_from`, and `has_input_validate` (schema
validation libraries like Pydantic enforce at runtime regardless of build mode).

**Future work**: If `prs_copy` is extended with build configuration data (Dockerfile,
CI YAML), a language+build-aware `has_enforced_contract` feature becomes tractable.

---

## Language onboarding readiness checklist

Before adding a new language's PRs to `prs_copy` and the steerer training set:

- [ ] Add tree-sitter grammar to `extract_conway_patch_features.py` language table
- [ ] Define stdlib set for import classification
- [ ] Audit which existing features are structurally zero for the new language (like
      `has_bare_except` for Go) and add language-equivalent signals
- [ ] Check `has_pub_func` pattern — does the language use a distinct visibility
      keyword/convention that requires a separate feature?
- [ ] Check `imp_external` — does the language have an intra-org module namespace
      that requires prefix-aware classification?
- [ ] Recompute per-feature prevalence on a language-stratified sample to detect
      new dominant confounds
- [ ] Run the steerer's acceptance head on language-stratified folds to verify
      AUROC does not degrade for the new language slice

### Go readiness status

| Check | Status | Notes |
|-------|--------|-------|
| Tree-sitter grammar | Ready — `go` supported in `tree_sitter_language_pack` | |
| Stdlib set | Ready — defined in `_STDLIB["go"]` | |
| `has_bare_except` substitute | Missing | Need `has_go_err_discard` |
| `has_try_catch` substitute | Missing | Need `has_go_err_wrap` and `has_go_err_check` |
| `imp_external` intra-org fix | Missing | Need `go.mod` prefix detection |
| `has_pub_func` | OK — pattern fires on capitalized function names, not `pub` keyword | Go-specific behavior is appropriate here |
| Prevalence audit | Not done | Blocked on prs_copy Go data availability |

`prs_copy` does not currently contain Go PRs. This checklist applies when they are added.

---

## Implementation plan (when triggered)

1. **`extract_conway_patch_features.py` updates**:
   - Add `has_go_err_wrap`, `has_go_err_discard`, `has_go_err_check` patterns
   - Add `go.mod` prefix scan for intra-org import reclassification
   - Add `has_empty_catch` (Java/JS/TS equivalent of bare_except)
   - Merge into `has_silent_error_swallow` unified signal
   - Restrict `has_version_suffix` to non-test file paths
   - Split `has_pub_func` into `has_pub_func_go`, `has_pub_func_java_kotlin`

2. **`train_pr_steerer_rl_v51.py` updates**:
   - Add language dummy features (`lang_python`, `lang_go`, `lang_rust`, `lang_js`)
     to allow the model to learn language-conditional baselines
   - Or: train per-language submodels and ensemble at inference
   - Evaluate: language-stratified CV AUROC for each new language slice

3. **`docs/phase5_1_rl_steerer_conway.md` updates**:
   - Update feature table with new/modified signals
   - Update friction lift table with language-stratified breakdown

---

## References

- Exp 5.1 training results: `docs/phase5_1_rl_steerer_conway.md`
- Conway's Law (Melvin Conway, 1968): "Organizations which design systems are constrained
  to produce designs which are copies of the communication structures of these organizations."
- Design by Contract (Bertrand Meyer, 1992): *Applying "Design by Contract"*, Computer 25(10)
