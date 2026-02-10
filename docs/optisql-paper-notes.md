# OptiSQL (arXiv:2601.13695) — Reading Notes

## Problem & Motivation
- The paper studies **executable SQL generation from table images**, arguing that real-world tables often appear as visual artifacts (scanned documents, PDFs, webpages) rather than clean structured text. Optical representations can capture table structure/layout more compactly than linearized text tokens.
- Key goal: test whether **compact optical tokens** can support SQL generation under strict token budgets, without relying on explicit textual schema/content access.

## Core Idea (OptiSQL)
- **OptiSQL** is a vision-driven framework that converts a table image into a **fixed-length sequence of optical tokens** using an OCR-oriented visual encoder, then conditions an autoregressive decoder to generate SQL.
- The default setting freezes the visual encoder (**FROZENENC**) and fine-tunes only the decoder to isolate **representational sufficiency** of optical tokens (i.e., does the tokenization preserve enough info?).
- An ablation **FULLFT** fine-tunes both encoder and decoder to measure the trade-offs between accuracy and robustness.

## Task Definition & Objective
- Input: table image `I` + natural-language question `q`. Output: executable SQL query `s` whose execution returns the correct answer.
- No textual schema or table contents are given at inference time; all table-related information must be inferred from the image.
- Efficiency goal: compare optical token count `|V|` with equivalent text linearization `|T|`, and study performance when `|V| << |T|`.

## Model Components
- **Visual Encoder**: OCR-oriented encoder generates optical tokens from the table image (fixed-length). Frozen by default.
- **Decoder Input**: concatenates optical tokens with question tokens, then projects into shared embedding space.
- **Autoregressive Decoder**: predicts SQL token-by-token until EOS, optionally with syntax constraints (core model relies on learned representations).

## Evaluation Metrics
- **Execution Accuracy (EXAcc)**: runs predicted SQL against the database and compares results to gold query execution.
- **Canonical Exact Match (EX-Can)**: normalizes SQL strings (case, whitespace, AND order at same level, etc.) before exact match. Avoids aggressive rewrites that can change semantics.
- **Robustness**: drop in EXAcc under visual perturbations (e.g., style changes, header masking).

## Dataset & Experimental Setup (High-Level)
- Experiments are run on a **visualized version of Spider 2.0-Snow**, where tables are rendered into images.
- Optical token budgets are explicitly controlled (e.g., 64, 100, 256, 400) to study accuracy–efficiency trade-offs.
- Diagnostic settings include **NOIMAGE** (remove optical tokens) and **WRONGTABLE** (mismatched table images) to verify visual grounding.

## Results (Key Takeaways)
- OptiSQL achieves strong execution accuracy **while using far fewer table tokens** (e.g., 256 optical tokens vs ~3,500 text tokens for table input), showing that compact optical tokens preserve enough structure/content for SQL generation.
- FROZENENC provides good accuracy and better robustness, while FULLFT slightly improves clean accuracy but **hurts robustness** (suggesting overfitting to superficial visual patterns).
- Accuracy increases with larger token budgets but exhibits diminishing returns beyond ~256 tokens, with latency rising roughly linearly.

## Perturbation & Robustness Findings
- **StyleShift**: change rendering style (fonts, spacing, borders) while preserving content — tests invariance to superficial appearance.
- **HeaderMask**: partially occlude headers — typically more damaging than style changes, highlighting reliance on schema cues.
- Robustness trends are relatively stable across token budgets; compression mainly limits capacity rather than changing perturbation sensitivity.

## Scope & Positioning
- The paper frames OptiSQL as an **efficiency-oriented alternative** rather than a direct competitor to text-to-SQL systems that assume ground-truth schemas.
- Optical tokens provide a compact interface that avoids explicit OCR reconstruction and can reduce error propagation vs. OCR pipelines.

## Implementation Notes (Appendix Highlights)
- Token accounting focuses on **table-only token counts** (question/output tokens excluded) to isolate compression benefits.
- Canonicalization rules are conservative: normalize case/spacing and reorder flat AND conditions, but **do not** rewrite joins or nested boolean structures.

## What This Means for Reproduction
- Need: table rendering pipeline to create image inputs; OCR-oriented visual encoder producing optical tokens (likely DeepSeek-OCR style); SQL decoder fine-tuning setup with frozen encoder; evaluation in SQLite for EXAcc + canonicalization for EX-Can; robustness perturbation generator (style shift, header mask).
