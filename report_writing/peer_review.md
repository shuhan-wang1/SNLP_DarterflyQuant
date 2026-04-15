# Peer Review: *Distribution-Aware Rotation Calibration for LLM Quantisation: Sliced Wasserstein Losses and Learnable Butterfly Rotations*

Reviewer: senior peer reviewer, UCL COMP0087 (2025/26)
Date: 2026-04-15
Source under review: `report_writing/tacl2021v1-template.tex` (1022 lines)

---

## 1. Executive Summary

The paper proposes replacing DartQuant's Whip loss with a 1-D Sliced Wasserstein Distance (SWD) loss, pairs the loss to the quantiser (SWD-Unif ↔ INT4/GPTQ, SWD-Gauss ↔ NF4/bitsandbytes), and supplies an appendix argument that fixed Hadamard $R_3/R_4$ are provably suboptimal under the RoPE-induced covariance, motivating learnable butterfly Givens rotations as future work. The core technical critique of Whip is correct and genuinely pleasing: the gradient-ratio calculation is tight, and the pairing principle is a useful conceptual contribution. SWD-Unif shows consistent but small (1.3–2.3% PPL) wins over Whip on three Llama-3 scales; the SWD-Gauss/NF4 branch is honestly reported as a negative result.

**Top 3 strengths.** (i) Gradient diagnosis of Whip vs SWD is crisp and the numeric bound (≈980× at $|y|=5$) lands. (ii) Negative SWD-Gauss result is reported in abstract, intro, §4.6 and §6 — no burying. (iii) RoPE/Hadamard appendix is a self-contained, well-structured theoretical contribution.

**Top 3 weaknesses.** (i) **No `references.bib` file exists on disk** — every `\cite*` key is unresolved, the paper will not compile to a correct PDF. (ii) Several headline "wins" sit at or below the authors' own 0.1-PPL / 0.5-acc no-separation threshold (tab:exp2_results in full). (iii) Question 3 is answered only theoretically; the paper's title promises "Learnable Butterfly Rotations" but provides zero experiments — a framing mismatch that a TACL reviewer will call out.

**Verdict for TACL-style venue:** **Major revision.** The loss contribution is real but incremental; the butterfly claim is not experimentally supported; the single-seed/narrow-benchmark posture is below TACL norms. Not a reject, but not yet an accept.

**Verdict for COMP0087 marking:** **Minor revision with one blocker.** Fix the bibliography file before submission; otherwise a well-structured empirical report with honest negative results, clear research questions, theoretical grounding, and all seven required sections. A strong pass at minimum.

---

## 2. Section A — Technical Correctness

### A.1 Proposition 1 (Whip gradient vanishing, `tex:284–292`)

**Verdict: correct but with a silent redefinition that should be disclosed.**

The paper writes
```
L_Whip = (1/N) sum_i exp(-|y_i|),    |∂L/∂y_k| = (1/N) e^{-|y_k|} → 0.
```
This is arithmetically correct. However, DartQuant defines Whip as a **sum**, not a mean (DartQuant §4.2, Eq. (4): `Whip = sum_{i=1}^{C_in} exp(-|x_i|)`). Under the sum form, the per-activation gradient is `e^{-|y_k|}` with no `1/N` factor. The exponential-decay point (Prop.~1's qualitative content) survives either definition, but the paper's $(1/N)$-normalised form should be flagged as a choice made for fair comparison with the SWD loss which is also a mean. Currently, `tex:231` attributes the $(1/N)$ form to DartQuant without this caveat — a small but correctable misrepresentation.

### A.2 Proposition 2 (SWD gradient scaling, `tex:294–304`)

**Verdict: correct.** The derivation is standard for a sorted-sample $W_2^2$ discretisation; `app:swd_gradient` (`tex:597–618`) supplies a clean rewrite in unsorted coordinates. The gradient formula $\frac{\partial \mathcal{L}}{\partial y_k} = \frac{2}{N}(y_k - t_{\sigma^{-1}(k)})$ holds almost everywhere, which is sufficient for SGD. The numeric claim at `tex:617` ("≈980 at $|y_k|=5$") is verifiable: $2 \cdot 3.3 \cdot e^5 \approx 979.8$. Correct.

**Minor caveat the paper glosses.** The claim that SWD's advantage grows "exponentially" (`tex:307`) is a statement about the *ratio*, not the absolute SWD gradient. For a user who cares about the absolute magnitude of the corrective force, the scaling is *linear* in $|y_{(N)}|$, not exponential. The current phrasing risks overselling.

### A.3 Theorem 1 (Post-RoPE variance, `tex:641–681`)

**Verdict: correct under the stated assumptions.** The derivation from single-position variance (Eq. 4) to position-averaged variance is textbook. Dropping the $\rho \neq 0$ term *inside the proof* (by setting $\rho=0$ in the hypothesis) is self-consistent but worth noting that the theorem statement is *conditional* on $\rho=0$ pre-RoPE. Remark 2 (`tex:683–687`) acknowledges this and punts to an empirical "$\rho \approx 0$" claim. That empirical claim is unsupported — no table/figure measures $\rho$ on a real Llama model. For TACL, a 1-figure histogram of pre-RoPE intra-band correlations at a single layer of Llama-3.1-8B would be very cheap to produce and would close the loop.

### A.4 Corollary 1 (Structured variance profile, `tex:701–705`)

**Verdict: correct in the asymptotic regimes, hand-wavy in the intermediate regime.** The $C_k \approx 1$ (low freq) and $C_k \approx 0$ (high freq) endpoints are correct. The parenthetical "not necessarily strictly monotonically, as intermediate-frequency bands may exhibit minor oscillations in $C_k$" (`tex:704`) is honest but weakens the corollary's practical punch. The corollary survives qualitatively.

### A.5 Proposition 3 (Hadamard suboptimality, `tex:719–749`)

**Verdict: correct as a $d=2$ counterexample; generalisation to full $d$-dim Hadamard is implicit and unproven.** The $2\times 2$ demonstration with $\Sigma = \bigl(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\bigr) \to \bigl(\begin{smallmatrix}3&0\\0&1\end{smallmatrix}\bigr)$ is a valid counterexample. However, the paper then generalises to "the Hadamard rotation" for full-dimensional $R_3/R_4$ (`tex:750–756`) without proof that Hadamard *at scale* inherits the same pathology. Empirically it clearly does, but a TACL reviewer will flag the gap. One additional sentence is enough: the $d$-dim Hadamard is a tensor product of $2\times 2$ blocks at the first level, so any correlated pair at that level reproduces the counterexample; subsequent levels compound the imbalance.

### A.6 Lemmas 1 & 2 (Givens equalisation, butterfly mixing, `tex:763–855`)

**Verdict: Lemma 1 correct. Lemma 2 correct but uses an *additional unstated assumption*.** Lemma 2's inductive step (`tex:843–854`) asserts "the paired channels are uncorrelated ($\rho = 0$ as argued above)". This is correct *only because the theorem's hypothesis assumed $\rho_{ij}=0$ on the original channels*. The statement of Lemma 2 carries the $\rho_{ij}=0$ hypothesis (`tex:826`), so this is internally consistent — but the remark at `tex:857–861` then waves this away ("When $\rho_{ij}\neq 0$, the fixed angle no longer achieves exact equalisation") without proving that *learnable* butterfly angles recover equalisation through a generic-$\rho$ covariance. The learnable-butterfly motivation is therefore **suggestive, not proven**. This is the single biggest gap between what the title promises and what the body delivers.

### A.7 Global-RMS argument (`tex:272–277`)

**Verdict: mostly correct; the CLT step is imprecise.** The argument that a per-dimension target makes the loss nearly rotation-invariant at a Hadamard initialisation relies on two statements: (i) each output $(y)_j$ is approximately Gaussian by CLT, and (ii) all $(y)_j$ have approximately equal variance. (ii) is what matters. (i) is not strictly needed and is the weaker claim. The CLT hand-wave would be improved by pointing instead at the fact that low-coherence orthogonal rotations equalise *variances* exactly up to $O(\max_{i,j}|R_{ij}|^2) = O(1/d)$ corrections — a deterministic statement about Hadamard's incoherence, not a CLT statement about distribution shape. This is a good point poorly justified.

### A.8 Lemma (Moment-matched uniform scale, `tex:574–586`)

**Verdict: correct, trivially.**

### A.9 Summary of Section A

Of the seven formal claims, **all are correct in substance**. Two (Theorem 1, Lemma 2) carry stronger assumptions than the paper acknowledges ($\rho=0$). One (Prop. 3) is proven only in $d=2$. The Whip gradient formula silently renormalises DartQuant's definition. None of these are fatal, and all are fixable with short paragraph-level edits.

---

## 3. Section B — Novelty & Positioning

### B.1 Is the framing of GPTQ/AWQ/SmoothQuant accurate?

**GPTQ (`tex:190, 352`):** Paper says GPTQ "minimises layer-wise weight quantisation error using approximate second-order information". Correct — matches GPTQ §4, especially the Hessian-based Optimal Brain Quantisation descent. However, the paper **uses GPTQ as the weight quantiser** in the INT4 branch, and this is non-trivial: Experiment 2 (`tex:439–441`) correctly identifies that the SWD-Gauss/NF4 shortfall confounds with bitsandbytes' lack of GPTQ-like calibration. The positioning is fair.

**AWQ (`tex:190`):** "identifies and protects the small fraction of weight channels most sensitive to quantisation error." Correct — AWQ §3.1 ("Improving LLM Quantization by Preserving 1% Salient Weights"). Accurate one-liner.

**SmoothQuant (`tex:191–193`):** Characterises SmoothQuant as migrating difficulty from activations to weights via per-channel scaling, enabling W8A8. Accurate. The paper does not attempt to compare numerically against SmoothQuant (which operates at W8A8, a different regime), which is acceptable; but the sentence "Despite these advances, these methods treat outliers as a calibration challenge to work around rather than a structural property to eliminate" (`tex:194`) is a rhetorical flourish that caricatures SmoothQuant slightly — SmoothQuant *does* eliminate outliers via the mathematically equivalent diagonal scaling, which is structurally similar to a rotation in spirit. Minor.

**Overall:** framings are accurate and fair.

### B.2 Is the positioning vs DartQuant honest?

Mostly yes, with one point to clean up.

- `tex:202`: "DartQuant avoids end-to-end backpropagation by optimising a globally shared $R_1$ via QR-Orth." Correct — DartQuant §4.1–4.3 and Algorithm 1.
- `tex:223–226`: "We retain the rotation insertion points, the QR-Orth optimiser and DartQuant's single-global $R_1$ training schedule." Correct — inherited.
- `tex:354`: "$R_2$ is trained per layer for 5 epochs." **DartQuant also trains $R_2$; DartQuant §5.1 explicitly names "learned rotation matrices $R_1$ and $R_2$".** The per-layer schedule for $R_2$ is not a new contribution but presented with the same neutrality as other inherited choices — which is honest. The paper does not over-claim here.

One subtle under-credit: DartQuant's ablation (DartQuant §5.2.1, Fig. 7a) already shows that Whip outperforms variance, kurtosis, and quantisation loss as an objective. The under-review paper replicates *the spirit* of that ablation but does not place its own SWD curve on the same figure. Reporting Whip/variance/kurtosis/SWD-Unif convergence curves in a single panel would be the tightest possible statement of novelty.

### B.3 Net incremental contribution

In one paragraph: **the paper substitutes a better-motivated loss (1-D Sliced-Wasserstein) for DartQuant's Whip loss, obtaining a 1.3–2.3% WikiText-2 PPL improvement at W4A4KV4; adds a conceptual "loss–quantiser pairing" principle whose Gaussian instantiation is honestly reported as failing at the backend-confound level; and supplies a self-contained theoretical appendix establishing that fixed-Hadamard $R_3/R_4$ cannot equalise variance under any non-zero intra-band RoPE correlation, motivating learnable butterfly Givens rotations as future work that is implemented but not evaluated.** Is this strong enough for TACL? The loss swap alone is below TACL bar. The combined package (loss + pairing principle + theoretical Hadamard suboptimality proof + negative NF4 result) is borderline. A TACL reviewer would want either (a) butterfly experiments to match the title, or (b) broader benchmark coverage (MMLU, LAMBADA, GSM8K, C4 with different stride, non-Llama families) to make the empirical case airtight. As submitted, it reads more like a workshop / EMNLP-short contribution than a full TACL journal paper.

For COMP0087, which is an 8-page empirical project, the contribution is **more than sufficient**: three coherent, well-motivated components with honest reporting.

---

## 4. Section C — Methodological Rigor

### C.1 Calibration setup (`tex:350–354`)

- 128 WikiText-2 samples, seq len 2048: matches DartQuant §5 exactly. Comparable.
- $R_1$ 10 epochs, $R_2$ per-layer 5 epochs, SGD lr=1e-3, momentum 0.9: not explicitly compared to DartQuant's hyperparameters. DartQuant uses "learning rate $\eta$" (Algorithm 1) without disclosing a value in the main body. The paper should explicitly state whether the optimiser/LR/momentum are inherited verbatim or retuned. Currently ambiguous — a TACL reviewer will flag this.
- No seed count disclosed (single seed, acknowledged). No sensitivity analysis on calibration-set size or sequence length. This is a common omission in the area but weak for a journal submission.

### C.2 Evaluation suite (`tex:345, 356`)

WikiText-2 + C4 perplexity at stride=2048 is the DartQuant setup. Adequate for PPL. Zero-shot: HellaSwag, ARC-Challenge, WinoGrande only.

**This is notably thin vs DartQuant.** DartQuant Table 2 reports LAMBADA, HellaSwag, ARC-C, ARC-E, BoolQ, MMLU, PIQA, SIQA, WinoGrande — 9 tasks. The paper tests 3. Consequences:

- Cannot claim robustness "across benchmarks" — only across three short-context tasks.
- For a "distribution-aware framework", missing LAMBADA (long-context dependency, sensitive to activation tails) and MMLU (knowledge retention, sensitive to weight preservation) is a real gap.
- GSM8K is a fair request given the paper emphasises instruction-tuned models (`tex:447–511`); none of the listed tasks probe reasoning.

**Strong suggestion:** add at least LAMBADA and PIQA. These are cheap to run under `lm-evaluation-harness` and bring the suite to comparable breadth with DartQuant.

### C.3 Single-seed numbers and the no-separation threshold (`tex:356`)

The paper explicitly states "we treat differences below 0.1 PPL or 0.5 acc as not separating two configurations." Given single-seed data and no CIs, this is a reasonable stance for a student project, **but it is not defensible without a sensitivity check**. For published work, one would need at minimum 3 seeds on one (model, regime) pair to justify the 0.1/0.5 threshold empirically. As-is, the threshold is asserted, not measured.

**Table-by-table audit against the 0.1 PPL / 0.5 acc threshold:**

- **Tab. `tab:exp1_main` (W4A4KV4):** SWD-Unif vs Whip — 1B Wiki-2: 13.90 vs 14.08 (Δ0.18, **above** threshold); 3B Wiki-2: 9.53 vs 9.75 (Δ0.22, above); 8B Wiki-2: 7.73 vs 7.85 (Δ0.12, above — barely). Accuracy: at 1B Avg, SWD-Unif 47.35 vs Whip 48.00 (Δ0.65, **Whip wins** above threshold); at 3B Avg, Δ0.23 (below threshold → tie); at 8B Avg, Δ0.44 (below threshold → tie). The accuracy story is therefore: SWD-Unif does not dominate accuracy.
- **Tab. `tab:exp2_results` (W4A16KV16):** SWD-Unif vs Whip Wiki-2 Δ: 0.06 (1B), 0.01 (3B), 0.02 (8B). **All three are below the 0.1 threshold.** The paper claims SWD-Unif "consistently achieves the lowest perplexity" at `tex:365`, but under its own threshold, W4A16KV16 is a tie on Wiki-2. This should be stated plainly.
- **Tab. `tab:exp3_acc`** row "SWD-Unif 8B Base 64.31 vs Instruct 63.97": Δ0.34, below threshold → tie, but presented with **bold** on the Base number (`tex:503`). Cosmetic — should not be bolded.

**Net:** the W4A4KV4 PPL claim (1.3–2.3%) is above threshold on Wiki-2 for all three scales — this is the paper's strongest result and is legitimate. But the paper frequently words "SWD-Unif consistently achieves the lowest perplexity" (`tex:365`, `tex:513`) without qualifying that under its own threshold, W4A16KV16 reduces to a tie. Soften to "SWD-Unif matches or exceeds Whip, with measurable W4A4KV4 wins and W4A16KV16 ties".

### C.4 Comparability (SWD-Unif uses INT4/GPTQ, SWD-Gauss uses NF4/bitsandbytes)

The paper itself flags this (`tex:352, 409, 439–441`). The acknowledgement is direct and repeated (abstract, §4.2, §4.6, §6). Is it sufficient? **Almost.** The honest answer is that Experiment 2 cannot separate the pairing principle from the backend confound — and the paper says so. What's missing is a single sentence explicitly stating that **the SWD-Gauss/NF4 negative result should not be interpreted as evidence against the pairing principle**. Currently the reader has to assemble this conclusion from `tex:441` ("A fair test of the pairing principle on the NF4 side requires..."). A one-line up-front disclaimer in §4.2 would help.

### C.5 Evaluation methodology nits

- `tex:356` says stride equals sequence length 2048. This is the "non-overlapping" stride, which is known to inflate PPL relative to stride=1. Consistent with DartQuant — acceptable.
- No reporting of FLOPs, inference latency, or memory savings: these are the actual deliverables of a PTQ paper. DartQuant reports both (GPU-hour and memory in Table 3). The under-review paper offers no comparable numbers. For TACL this is a notable omission.

---

## 5. Section D — Experimental Honesty

- **SWD-Gauss/NF4 negative result**: reported in abstract (`tex:146–149`), in the bullet list (`tex:170`), in §4.2 heading (`tex:406`) and discussion (`tex:438–441`), and in §6 conclusion (`tex:531`). **Not buried.** This is one of the paper's stronger honesty moves.
- **FP16 baselines** (Table 1, etc.): labelled FP16 (correct per recent commit in gitlog), values plausible for Llama-3 scales (Wiki-2 9.75 for 3.2-1B, 7.81 for 3.2-3B, 6.24 for 3.1-8B match public numbers).
- **"Wins" within threshold**: as per §C.3 above, two individual cells are bolded despite being ties (`tab:exp2_results` 1B and 3B Wiki-2 C4 rows; `tab:exp3_acc` 8B SWD-Unif Base column). Fix: de-bold anything whose Δ is below the declared threshold.
- **Hype vocabulary**: word scan — "novel" 0 uses; "state-of-the-art" 0 uses; "groundbreaking" 0 uses; "substantially" used once at `tex:398` ("SWD-Gauss trails substantially") with numeric justification ($+20$–$30\%$). Good discipline.
- **One possibly over-reaching sentence**: `tex:171` "**The INT4 instantiation achieves the strongest perplexity we observe**" — true, but presented as a positive result for the pairing principle when in fact SWD-Unif/INT4 is the *baseline-beater*, and the pairing principle was supposed to predict a *parallel* win for SWD-Gauss/NF4, which it did not. The pairing principle is therefore only half-validated. Currently stated correctly in the same bullet ("The NF4 instantiation...underperforms INT4/GPTQ at matched precision"), so on balance honest.

---

## 6. Section E — Structure & Format Compliance

### E.1 Required sections (per `requirements.pdf` p.3)

Checked in order:

1. Abstract — `tex:145–149` ✓
2. Introduction — `tex:157–180` ✓
3. Related Work — `tex:185–214` ✓
4. Methods — `tex:219–337` ✓
5. Experiments — `tex:342–513` ✓
6. Results & Discussion — **merged into §4** via explicit "Experiment" subsections and a final "Discussion" paragraph at `tex:513`. Requirements.pdf permits merging. ✓
7. Conclusions + Future Work — Future Work is §5 (`tex:518–523`), Conclusion is §6 (`tex:528–531`). Both present.

**Ordering oddity:** Future Work (§5) precedes Conclusion (§6). This is unconventional — most venues put Future Work inside or after Conclusion. Not a formal violation, but a TACL editor will notice. Suggest either (a) merge §5 into §6 or (b) put §5 after §6.

### E.2 Conclusion substantive?

Yes. `tex:528–531` is a single paragraph of ~150 words and does the required work: recaps contribution, states main positive result with number, states negative result with caveat, flags future-work open problem. Not empty/perfunctory. This is an improvement over the state flagged in `polish_prompt.md`.

### E.3 Page budget plausibility

Total tex content is 1022 lines. Main body (lines 1–532, before `\appendix`): ~400 lines of LaTeX content plus 4 tables (Tab. 1–4), zero figures. In TACL 11pt A4 one-column style, this is roughly:

- Abstract: ~0.5 pp
- Introduction (incl. bullet contributions + 3 questions): ~1.5 pp
- Related Work: ~1 pp
- Methods (incl. Prop. 1, Prop. 2, Remark, pairing): ~2.5 pp
- Experiments + 3 tables: ~2.5 pp
- Future Work + Conclusion: ~0.5 pp

**Estimate: 8.0–8.5 pages of main body.** On the hairy edge. The three candidates to move to appendix if overflow:

1. **Experiment 3 (Instruction-Tuning interaction, `tex:447–513`)** — orthogonal to the three research questions, self-acknowledged as "additional analysis". Full tables Tab. 3 and Tab. 4 in main body; could cleanly move the full-breakdown tables to appendix (they already are in `app:exp3`) and reduce to a 1-paragraph main-body summary.
2. **The §3 "Critical design choice: global RMS" paragraph (`tex:272–277`)** — important but long; could compress to 3 sentences in main body and push the CLT argument into appendix.
3. **Remark 1 (QR-Orth compatibility, `tex:310–314`)** — useful but not a core claim; belongs in appendix.

### E.4 LaTeX hygiene

- **BLOCKER: no `references.bib` file exists in `report_writing/`.** `grep` for `*.bib` across the project returns nothing. The `\bibliography{references}` directive at `tex:537` will fail to resolve. **All 22+ citation keys in the document are currently unresolved.**
  - Keys used: `quarot`, `spinquant`, `dartquant`, `gptq`, `awq`, `smoothquant`, `llmint8`, `quip`, `quipsharp`, `qlora`, `cuturi2013sinkhorn`, `wasserstein`, `deshpande2018`, `kolouri2019`, `lv2024wasserstein`, `dao2020kaleidoscope`, `dao2022monarch`, `su2024roformer`, `li2021riemannian`, `trockman2021`, `blondel2020`, `blom1958`, `wilk1968`, `dkw1956`, `golub2013matrix`, `butterflyquant`, `dao2019`.
  - The repository has PDFs for four of these (`dartquant`, `gptq`, `awq`, `smoothquant`) but no BibTeX entries. A compile run would emit 27+ `Warning: Citation undefined` messages.
- No TODO/FIXME/XXX markers in the .tex. Clean in that respect.
- No obvious dangling `\ref`s observed at scan.
- British spelling mostly consistent: "quantisation" throughout (good), "optimisation" (good), "behaviour" (good), "minimises" (good). One slip: **"normalised"** appears at `tex:276, 606, 611` — British. But `tex:270` `\hat{\sigma} = \RMS_{\text{global}}` uses American "root-mean-square"? Actually that's a neutral term. No real inconsistencies found.

### E.5 Typographic nits

- `tex:367` caption reads "Table~\ref{tab:exp1_main} above compares…" but the table is placed *after* the paragraph, not above — inherited from float placement. Drop "above" to be safe.
- `tex:409` "we investigate whether the poor results of SWD-Gauss **is caused** from a distributional mismatch" — grammar: should be "are caused by" or "is caused by" (consistent subject). Slight.
- `tex:903` "Table~\ref{tab:app_exp2_full} details the full per-benchmark results…" — fine.

---

## 7. Section F — Writing Quality & Clarity

### F.1 Three weakest paragraphs

**Weakest #1: `tex:272–277` (Critical design choice: global RMS)**

> "Per-dimension normalisation renders the loss approximately rotation-invariant for rotations with low coherence (e.g., Hadamard-initialised rotations): for such rotations, each output dimension $(\by)_j = \sum_k R_{jk} x_k$ is a weighted sum of many input channels with roughly equal weights $|R_{jk}| \approx 1/\sqrt{d}$, so by the Central Limit Theorem each output dimension is approximately Gaussian with similar variance when $d$ is large."

This is a 3-sentence argument fused into a single long sentence that (a) invokes the CLT when it only needs variance equalisation (see §A.7), (b) conflates "approximately Gaussian" with "approximately rotation-invariant", and (c) never states what the signal-destruction failure mode actually looks like in practice. Needs breaking into three sentences.

**Weakest #2: `tex:409` (Experiment 2 motivation)**

> "To address the performance gap observed in Experiment~1, we investigate whether the poor results of SWD-Gauss is caused from a distributional mismatch (Uniform vs Gaussian) or the quantiser implementation itself."

"Is caused from" is ungrammatical; should be "arises from". "The quantiser implementation" is ambiguous: does it mean NF4 codebook or bitsandbytes backend? These are different things. The whole sentence should be restructured.

**Weakest #3: `tex:509` (Attribution paragraph)**

> "We attribute this scale-dependent behaviour to two factors. First, instruction tuning sharpens the output distribution, increasing sensitivity to quantisation-induced perturbations on open-domain text. Second, at smaller scales, instruction tuning consumes representational capacity critical for surviving aggressive quantisation, whereas 8B models retain sufficient redundancy to benefit from instruction-aligned reasoning on structured tasks like ARC-Challenge."

Both "factors" are post-hoc speculation with no supporting experiment, no citation, and no measurement of "output distribution sharpness" or "representational capacity". This reads as narrative, not analysis. Either add a measurement (e.g. KL-divergence between base and instruct output distributions at matched prefix) or soften to "We speculate that…" and retract "consumes representational capacity critical for surviving aggressive quantisation".

### F.2 Unsupported quantitative claims

- `tex:398` "trails substantially ($+20\%$–$30\%$ PPL vs.\ SWD-Unif)" — *supported* by numbers, keep.
- `tex:452` "$+14\%$–$35\%$ higher WikiText-2 PPL" — supported, keep.
- `tex:686` "Empirically, pre-RoPE activations in LLMs exhibit near-zero cross-channel correlation within each frequency band" — **unsupported; no citation, no measurement in this paper.** Either add a citation or a one-figure appendix measurement.
- `tex:509` "sharpens the output distribution" — **unsupported**, no citation.
- `tex:509` "consumes representational capacity critical for surviving aggressive quantisation" — **unsupported speculation**.

### F.3 Antecedent ambiguity

- `tex:214` "DartQuant introduced the QR-Orth optimiser, which we show is compatible with the SWD loss". "We show" refers to Remark 1 — fine.
- `tex:441` "The remaining gap between SWD-Gauss/NF4 and SWD-Unif/INT4 in Table~\ref{tab:exp2_results} is consistent in magnitude with this difference in weight reconstruction quality". "This difference" is clear from context — acceptable, but rephrase to "the difference" for crispness.
- `tex:398` "This confirms the theoretical prediction from Proposition~\ref{prop:swd_grad}". "This" refers to the paragraph above, but the referent is loose; make explicit as "The consistent SWD-Unif advantage over Whip confirms…".

### F.4 Stylistic notes

- British vs American: consistent British (noted in §E.4).
- Mathematical notation is clean; bold vector convention $\mathbf{x}, \mathbf{R}$ used consistently.
- Math/prose ratio is high in §3 and the appendix — appropriate for the paper's aim.

---

## 8. Prioritised Action List

1. **[BLOCKER]** **Create/recover `references.bib` and verify `bibtex` + `pdflatex` compile to a clean PDF.** No .bib file exists in `report_writing/`. Every `\citep{...}`/`\cite{...}` key is unresolved. This is the single most important pre-submission task. Until this is fixed, the paper does not compile and no marker/reviewer can read citations.
2. **[BLOCKER]** **Verify main-body page count on a live compile.** Current estimate is 8.0–8.5 pp — at or over the 8 pp cap. If over, execute the Experiment 3 demotion to appendix plan in §E.3.
3. **[MAJOR]** **Re-audit every bolded number against the declared 0.1 PPL / 0.5 acc no-separation threshold.** At least three cells in `tab:exp2_results` and `tab:exp3_acc` are bolded despite being ties. De-bold them and soften the claim at `tex:365` and `tex:513` to "SWD-Unif matches or exceeds Whip; W4A4KV4 wins are above threshold, W4A16KV16 ties are within threshold."
4. **[MAJOR]** **Add pre-RoPE correlation measurement to close the $\rho\approx 0$ gap (§A.3 and `tex:686`).** A one-panel histogram of intra-band $\rho$ values on Llama-3.1-8B at one layer would promote the whole Hadamard-suboptimality argument from "theoretical under assumption" to "theoretical with empirical support". Cheap.
5. **[MAJOR]** **Either run butterfly experiments or soften the title.** The title advertises "Learnable Butterfly Rotations" but the paper offers only a theoretical motivation and an implementation reference. For TACL, this is a framing mismatch. Options: (i) drop the butterfly phrase from the title to e.g. "…Sliced Wasserstein Losses for Rotation Calibration"; (ii) at least one ablation row on one model scale. A 1B-scale butterfly-vs-Hadamard PPL row would be defensible.
6. **[MAJOR]** **Broaden benchmark coverage: add LAMBADA and PIQA.** Both are cheap, both appear in DartQuant's Table 2, and both probe regimes (long-context, commonsense physics) that the current three tasks miss. Failure to do so leaves the "distribution-aware framework" claim under-tested.
7. **[MAJOR]** **Run at minimum 3 seeds on one (model, regime) pair to justify the 0.1/0.5 threshold.** Currently the threshold is asserted without measurement. 3 seeds × 1 model at W4A4KV4 is ~1 GPU-day.
8. **[MAJOR]** **Acknowledge the Whip mean-vs-sum normalisation (§A.1).** Add one sentence at `tex:231` stating explicitly that the paper uses the mean-normalised Whip form for comparability with SWD; DartQuant's original definition is the sum form.
9. **[MINOR]** **Extend Prop. 3 from $d=2$ to general $d$** with a one-sentence tensor-product argument (§A.5).
10. **[MINOR]** **Reorder §5 and §6.** Put Conclusion before Future Work, or merge.
11. **[MINOR]** **Fix the three weakest paragraphs (§F.1).** Rewrite for clarity; drop unsupported speculation at `tex:509` or explicitly mark as speculation.
12. **[MINOR]** **Remove antecedent ambiguities and grammatical slips** listed in §E.5 and §F.3.
13. **[MINOR]** **Report FLOPs / memory / calibration GPU-hours** to match DartQuant's Table 3 reporting standard.

---

## 9. Open Questions for the Authors

1. **Bibliography status.** Is there a `references.bib` file that was not committed to the repository, or is the bibliography genuinely missing? If the former, please commit it before submission. If the latter, plan for ~1 day to compile the entries.
2. **$R_1$/$R_2$ hyperparameters vs DartQuant.** Are SGD lr=1e-3 and momentum=0.9 inherited verbatim from DartQuant, or retuned on your WikiText-2 calibration set? If retuned, over what grid? If inherited, can you cite the DartQuant value explicitly?
3. **Butterfly implementation status.** The codebase contains `ButterflyRotation` and `ButterflyFactored` prototypes. Why are zero experiments reported? Is this a compute constraint or a correctness concern? If the former, can at least one small-scale (1B, W4A4KV4) row be added before the 2026-04-17 deadline?
4. **Pre-RoPE correlation $\rho$.** What is the empirical distribution of intra-band $\rho$ on the Llama-3 models you evaluate? A single histogram would convert Theorem 1's $\rho=0$ assumption from unsupported to supported.
5. **NF4 with GPTQ-equivalent backend.** Is there a realistic path (e.g. AutoGPTQ NF4 support, or a custom NF4 GPTQ pass) to run the fair SWD-Gauss/NF4 test the paper identifies as missing? Even a 1B-scale comparison would close the pairing-principle loop.

---

Review saved to `C:/Users/shuhan/Desktop/UCL/snlp/int4_quantization_darkquant/report_writing/peer_review.md`.
