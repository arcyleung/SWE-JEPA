# SWE-JEPA

**Latent-space predictive architecture for code understanding, inspired by SALT and V-JEPA.**

Full research proposal: [`docs/code-jepa-research-proposal.md`](docs/code-jepa-research-proposal.md)

---

## Motivation

Current AI coding assistants are good at token-level generation but blind to the structural properties of code that determine long-term quality: modularity, volatility, organizational ownership, testability. These are not properties of individual functions in isolation — they emerge from how code relates to its surrounding system, authoring organization, and its history of change.

We take inspiration from **SALT** (arxiv:2509.24317), which showed that for video representation learning: (1) a cheap frozen teacher provides sufficient latent targets, (2) students trained on those targets can outperform their teachers, and (3) teacher quality matters far less than the information asymmetry built into the learning task.

**SWE-JEPA** applies the same principle to code: freeze a pretrained code LLM as a teacher, extract its mid-layer hidden states at function body positions, and train a student to predict those latent targets from only the function signature and context. The hypothesis is that forcing the student to bridge this gap — from sparse interface to dense latent — compels the emergence of abstract software engineering reasoning without explicit chain-of-thought.