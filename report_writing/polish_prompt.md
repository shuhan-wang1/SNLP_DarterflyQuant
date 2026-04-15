# Report Polish Prompt — TACL Submission, Edan-Toledo Style Mimicry

You are acting as a senior ML researcher and TACL copy-editor. Your task is to polish the existing LaTeX report at `report_writing/tacl2021v1-template.tex` so that, after revision, it reads like a self-contained TACL submission authored in the voice of **Edan Toledo** (InstaDeep → Meta FAIR; first-author of *CoDreamer*, co-author of *SPO*, *Beyond-the-Boundaries-of-PPO*, and *AIRA*). Do not invent new empirical results. Do not rewrite the LaTeX template packages or geometry. Work inside the existing file.

---

## 1. Paper under revision

- **File:** `report_writing/tacl2021v1-template.tex`
- **Title:** *Distribution-Aware Rotation Calibration for LLM Quantization: Sliced Wasserstein Losses and Learnable Butterfly Rotations*
- **Core thesis:** DartQuant's Whip loss has exponentially vanishing gradients for the very outliers it targets; we replace it with a Sliced Wasserstein Distance (SWD) loss that scales linearly with outlier magnitude, pair it principally with the quantizer (SWD-Unif↔INT4, SWD-Gauss↔NF4), train an independent per-layer $R_1$, and prove that fixed Hadamard matrices are suboptimal under RoPE-induced covariance, motivating future learnable butterfly rotations.
- **Current state of the draft:**
  - Abstract, Introduction, Related Work, Methodology, Experiments 1–3, Future Work, Appendices A–C — all substantially drafted.
  - **§6 Conclusion is empty.** Must be written.
  - Some paragraphs are over-long, some transitions are abrupt, and British/American spelling is inconsistent.
  - Related Work currently uses `\citep` / `\citet` but the `.bib` file is not yet populated — flag this but do not fabricate references.

---

## 2. Hard constraints (from `report_writing/requirements.pdf`)

1. Use the TACL 2021 v1 LaTeX style **unchanged** — no font size, margin, or geometry edits.
2. **Main body ≤ 8 pages**, references and appendix excluded. If current compiled output overflows, move material to the appendix rather than truncating it.
3. Required sections, all present: (1) abstract, (2) introduction, (3) related work, (4) methods, (5) experiments, (6) results & discussion, (7) conclusions with future-work discussion. The present draft merges (5)+(6); keep them merged but ensure the discussion *obligation* is met inside the experiments section.
4. The report must be empirical and answer a clear research question.
5. Single PDF submission. File name is the group name.

---

## 3. Target voice — Edan Toledo style guide

Four of Toledo's papers have been downloaded to `report_writing/edan_toledo_references/`:
- `2411.00666_beyond_ppo.pdf` — *Beyond the Boundaries of Proximal Policy Optimization* (Tan, **Toledo**, Ellis, Foerster, Huszár, 2024)
- `2402.07963_spo.pdf` — *SPO: Sequential Monte Carlo Policy Optimisation* (Macfarlane, **Toledo**, Byrne, Duckworth, Laterre, NeurIPS 2024)
- `2406.13600_codreamer.pdf` — *CoDreamer* (**Toledo**, Prorok, CoCoMARL RLC 2024) — the paper where Toledo is first author; best fingerprint.
- `2507.02554_aira_mle_bench.pdf` — *AI Research Agents for Machine Learning (AIRA)* (**Toledo** et al., 2025)

Before editing, re-read at least the abstract + introduction + first methodology subsection of each. Then enforce the following stylistic rules, each of which is directly attested in those papers:

### 3.1 Openings are declarative, not ornamental
- **Yes:** "Sample efficiency is a critical challenge in Reinforcement Learning." (CoDreamer)
- **Yes:** "Proximal policy optimization (PPO) is a widely-used algorithm for on-policy reinforcement learning." (Beyond-PPO)
- **No:** "In recent years, with the rapid growth of…". Delete every "In recent years" and "With the advent of" you see.
- Rewrite the first sentence of the **abstract** and the first sentence of **§1 Introduction** to state the field fact and the problem in one line each.

### 3.2 Framework framing, not trick framing
Toledo consistently reframes a contribution as a *decomposition* or *framework* rather than a single heuristic. Examples:
- "outer-PPO, a framework wherein these update vectors are applied using an arbitrary gradient-based optimizer." (Beyond-PPO)
- "We formalize AI research agents as search policies that navigate a space of candidate solutions." (AIRA)

**Apply:** recast the method section around the phrase "a distribution-aware rotation calibration framework", and describe SWD-Unif / SWD-Gauss / per-layer $R_1$ as instantiations of the framework, not a menu of tricks.

### 3.3 Explicit numbered research questions at the end of §1
Toledo frequently closes the introduction with enumerated *questions* the paper answers:

> "**Question 1.** Is the unity learning rate always optimal?
> **Question 2.** Is the independence (…) of each outer update step always optimal?
> **Question 3.** Is initializing the inner loop (…) always optimal?" (Beyond-PPO, p.1)

**Apply:** introduce three questions that map to our three contributions:
1. Does a principled distribution-matching loss outperform Whip under identical rotation machinery?
2. Does the correct loss–quantizer pairing dominate the choice of quantizer family?
3. Does per-layer $R_1$ — enabled by DartQuant's cheap calibration — recover accuracy that a global $R_1$ leaves on the table?

### 3.4 Bulleted contributions immediately after the questions
Every Toledo paper ends the intro with an active-verb bullet list ("We propose…", "We optimize…", "We perform…", "We evaluate…", "Given the stated empirical results we conclude…"). The current draft already has this bullet list — tighten each bullet to ≤ 2 sentences and ensure the last bullet acknowledges a *negative* finding (the SWD-Gauss/NF4 result is honest negative evidence we must surface, not bury).

### 3.5 Honest negative reporting
Toledo states failures explicitly:
- "No method improves over baseline on MinAtar." (Beyond-PPO)
- "However, the performance improvements are relatively marginal." (CoDreamer)

**Apply:** the Experiment 2 finding that NF4/bitsandbytes underperforms GPTQ/INT4 *regardless of loss* is a negative result for our own SWD-Gauss pitch. State it plainly. Do not soften it with "interesting future direction" hedges.

### 3.6 British English throughout
Toledo (InstaDeep/London) consistently writes *optimisation, utilise, parameterise, normalise, minimise, generalisation, behaviour, favourable*. The current draft mixes "optimization/optimisation". Choose British spelling and replace globally. Exceptions: keep mathematical operators (`\RMS`, `INT4`, `NF4`) and quoted foreign terms untouched.

### 3.7 Bold inline definitions, not floated definitions, for first-use terms
Toledo marks first-use terms with `\textbf{Term}.` at the start of a paragraph (see AIRA's "**Fitness Function.**", "**Selection Policy.**", etc., and Beyond-PPO's "**Trust region**"). The current `\paragraph{…}` markers already do this — keep them, but ensure every jargon term first appears in a bolded-head paragraph.

### 3.8 Epistemic markers: "We emphasise / We note / We observe / We find"
Toledo signals what matters with short epistemic markers:
- "We emphasize that we do not seek to identify the most performant configuration possible…" (Beyond-PPO)
- "We note that…", "We find that…".

**Apply:** add one "We emphasise that…" in §3 to scope the contribution (e.g., "We emphasise that we do not modify DartQuant's rotation-insertion topology; our changes are confined to the loss and the granularity of $R_1$ training."). Add "We find that…" framings in the experiments.

### 3.9 Precise quantitative claims
Every empirical claim must carry a number, a CI, or a range. Examples:
- Yes: "5–10% on both Brax and Jumanji" (Beyond-PPO)
- No: "significant improvement".

Audit every sentence in §5 that uses "significantly", "notably", "substantially"; attach a number or delete the adverb.

### 3.10 Short paragraphs, explicit antecedents
3–5 sentences per paragraph. Every "This X" must have a noun `X` that points to the *immediately preceding* clause. Break any paragraph above 7 sentences.

### 3.11 Method-section rhythm: problem → observation → action
Toledo's §3/§4 paragraphs follow this tri-beat:
1. State the problem in one sentence.
2. "We observe that… / However, …" — one-sentence diagnosis.
3. "We therefore / We propose…" — one-sentence intervention.

**Apply:** rewrite the opening paragraph of §3.1 (SWD loss), §3.3 (pairing), and §3.4 (per-layer) to hit this rhythm.

### 3.12 Evaluation methodology transparency
Toledo names methodology papers (Agarwal et al. 2021 for rliable, IQM, probability-of-improvement, etc.). Our Experiments section should:
- Explicitly name the perplexity estimator (stride, sequence length).
- Explicitly name the LM-eval-harness version for the three zero-shot benchmarks.
- State seed count and whether numbers are single-seed or averaged (if single-seed, say so).

### 3.13 No hype adjectives
Strike: *novel, groundbreaking, superior, comprehensive, cutting-edge, state-of-the-art* (use "the best reported result" or a number instead).

---

## 4. Concrete rewrite tasks (in priority order)

1. **Abstract (≤ 250 words).** Rewrite to open with a declarative fact about PTQ + outliers, then: (a) the vanishing-gradient observation about Whip, (b) the SWD proposal with the loss–quantizer pairing, (c) per-layer $R_1$, (d) the Hadamard-suboptimality theorem and butterfly future work, (e) the main empirical number (e.g., WikiText-2 PPL 7.73 vs 7.85 on Llama-3.1-8B, W4A4KV4). Ensure the negative SWD-Gauss/NF4 result is mentioned, not hidden.

2. **§1 Introduction.** Enforce §3.1–§3.4 above: declarative opening, framework framing, three numbered Questions, tightened bullet contributions, negative-finding bullet last.

3. **§2 Related Work.** Currently strong. Only two changes: (a) fix `\citep`/`\citet` so the `references.bib` compiles — flag any missing keys as `% TODO(bib): key XYZ` rather than inventing citations; (b) tighten the final sentence of each paragraph so it ends with a "but / however" that sets up the next paragraph.

4. **§3 Methodology.** Recast as a single distribution-aware framework (see §3.2). Ensure §3.1 opening paragraph uses the problem → observation → action rhythm. Keep the propositions and remarks as-is; they are already Toledo-shaped.

5. **§4 Experiments + §5 Results/Discussion.** Merge into one section titled **Experiments** (as the draft already does) but ensure a final `\paragraph{Discussion.}` block inside §4.6 that:
   - States the negative result on SWD-Gauss/NF4 plainly.
   - Ties the base-vs-Instruct finding back to the loss-quantizer pairing principle.
   - Ends with one sentence of scope limitation (seed count, model family, context length).

6. **§6 Conclusion (currently empty).** Write ~150 words. Three beats:
   - One sentence summarising the framework.
   - One sentence summarising the positive result (SWD-Unif + INT4 + per-layer $R_1$).
   - One sentence honest about the negative result (NF4/bitsandbytes).
   - One sentence on future work (learnable butterfly rotations, already motivated in §5).

7. **§5 Future Work.** Keep one paragraph only. Toledo's future-work sections are short and concrete; cut any paragraph that repeats the appendix proof.

8. **Page budget.** After rewrite, compile and check page count. If > 8 pages, move proofs/tables to the appendix in this order: (i) the full per-benchmark tables (already in appendix — good), (ii) the Givens/butterfly proofs, (iii) the gradient-comparison numeric example.

---

## 5. Hard rules — things you must not do

- Do not change empirical numbers.
- Do not invent citations or BibTeX entries.
- Do not add new experiments or new baselines.
- Do not change the LaTeX template preamble or class options.
- Do not write "In recent years", "With the advent of", "groundbreaking", "novel" in reference to our own work.
- Do not delete the proofs in Appendix B; they are core to the future-work claim.
- Do not exceed 8 pages in the main body.
- Do not mix American and British spelling; pick British.

---

## 6. Verification checklist (run before reporting done)

- [ ] Abstract ≤ 250 words, opens declaratively, mentions the negative result.
- [ ] §1 closes with numbered Questions + bullet contributions, including a negative-finding bullet.
- [ ] Every `\paragraph{…}` heading is bolded and its paragraph is ≤ 7 sentences.
- [ ] Every "significantly / substantially / notably" has a number or is deleted.
- [ ] "optimisation" consistent throughout; no stray "optimization".
- [ ] §6 Conclusion is written.
- [ ] Compile is clean; overfull/underfull hbox warnings logged; page count ≤ 8.
- [ ] `references.bib` TODOs flagged, not fabricated.
- [ ] The four downloaded Toledo PDFs are **not** cited in our bibliography (they are style references, not scientific antecedents).

---

## 7. Deliverable

Return (a) the revised `tacl2021v1-template.tex` as a single edit pass, and (b) a short changelog listing every section touched and a one-line justification for each change, e.g.:

> `§1.¶1` — rewrote opening sentence to declarative form (Toledo §3.1).
> `§3.1.¶1` — restructured to problem → observation → action rhythm (Toledo §3.11).
> `§6` — added 4-sentence conclusion covering framework / positive / negative / future work.
