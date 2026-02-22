Based on the full matrix results, including min/max of similarities this time: Should I select the model with higher or lower variance/ dynamic range? And how could that choice impact the downstream decoder's performance of this SWE-JEPA idea?

● Choose higher variance (wider dynamic range) — the Coder-3B Base.
  Here's the reasoning:
  ---
  Why dynamic range matters for SWE-JEPA targets
  The teacher's role is to provide target representations the student must predict. The decoder takes a context encoding + masked region position and regresses toward the teacher's embedding. So the quality of the teacher signal determines what the student is forced to learn.

  The collapse risk with low variance (Qwen3-8B)

  Qwen3-8B has:
  - min=0.728, max=0.951, mean=0.830, range=0.223

When the floor is 0.728, all function embeddings live in a tight cluster. The decoder can exploit this: predicting the centroid of the function embedding space gives a permanently low MSE loss, even if it has learned nothing about individual functions. This is the classic representation collapse failure mode — the student learns "this is a cache class" rather than "this is specifically the hashing utility." The high floor is a signal that 8B is saturating representations with class-level context, leaking shared information into every function embedding.

  Why high variance (Coder-3B Base) is the right signal

  Coder-3B Base has:
  - min=0.655, max=0.954, mean=0.790, range=0.299

  Two key properties make this superior:

  1. The variance is semantically meaningful, not noise.
  - get↔put = 0.954: correctly high — both are cache access operations touching the same internal state
  - _hash_key↔__init__ = 0.655: correctly low — pure string transformation vs. stateful initialization, genuinely distinct semantics
  - The spread maps to real code structure, so the student is being rewarded for learning real semantic distinctions

  2. It prevents decoder degeneracy.
  With a range of 0.30, a centroid-predicting decoder incurs meaningful loss on outlier pairs like _hash_key. The decoder is forced
  to learn function-specific features to reduce loss, which is exactly what you want.

  ---
  Downstream decoder impact summary

  Property: Centroid shortcut
  Coder-3B Base (range 0.30): Penalized — _hash_key is far out
  Qwen3-8B (range 0.22): Viable — floor is 0.73
  ────────────────────────────────────────
  Property: Gradient signal
  Coder-3B Base (range 0.30): Rich, varies by semantic similarity
  Qwen3-8B (range 0.22): Weak, compressed cluster
  ────────────────────────────────────────
  Property: What student learns
  Coder-3B Base (range 0.30): Function-specific semantics
  Qwen3-8B (range 0.22): Class-level context
  ────────────────────────────────────────
  Property: Risk
  Coder-3B Base (range 0.30): Harder to train initially
  Qwen3-8B (range 0.22): Silent collapse — loss looks good but representations aren't useful
  ────────────────────────────────────────
  Property: Alignment with phase 0
  Coder-3B Base (range 0.30): Consistent (layer 18 best differentiation)
  Qwen3-8B (range 0.22): Contradicts phase 0 finding

  ---
One caveat to watch

The get↔put similarity of 0.954 is high enough that the decoder might also collapse within that subgroup. When you move to a real dataset (Experiment 0.2), watch whether functionally similar functions (e.g., multiple list traversal functions) all collapse to the same representation. If they do, you may need to add a variance regularization term to the student loss or use a contrastive auxiliary objective on top of the regression target — but that's a phase 2 concern. For now, Coder-3B Base at layer 18 gives the right inductive bias.