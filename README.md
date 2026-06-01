# SiGMA — Spectral Geometry of Multi-model Agreement

Code and data for the paper **"When Consensus Hides Hallucination in Multi-LLM Systems."**

SiGMA queries nine commercial LLMs once each at low temperature on three
question-answering benchmarks (TruthfulQA, SimpleQA, FreshQA), embeds the nine
answers to each question into a single response cloud, and characterizes the
spectral and volumetric geometry of that cloud across five consensus states
(AR, MR, Sp, MW, AW). The central finding is a cross-dataset sign flip: on
TruthfulQA the all-wrong (AW) cloud contracts onto a geometry statistically
non-separable from correct consensus, while on SimpleQA and FreshQA it expands.

> Manuscript under review. Citation and DOI will be added on publication.

---

## Repository contents

The pipeline runs in five stages. Scripts are named per dataset; the table
below uses TruthfulQA as the example, and SimpleQA / FreshQA versions follow the
same naming.

| File | Stage | Language | Role |
|---|---|---|---|
| `script_TruthfulQA_Stage1.py` | 1 | Python | Collect the 9 model answers via OpenRouter |
| `script_grade_TruthfulQA_Stage2.py` | 2 | Python | Grade each answer against the reference, write verdicts |
| `script_Stage3_TruthfulQA.m` | 3 | MATLAB + Python | Embed answers and questions with all-mpnet-base-v2 |
| `script_Stage4_TruthfulQA.m` | 4 | MATLAB | Per-state geometry, per-state means, p-values, rebound |
| `script_Stage5_TruthfulQA.m` | 5 | MATLAB | Sparse-decomposition check (s_max, co-endorsement) |

Data files (provided so Stages 1–2 can be skipped):

| File | Produced by | Contents |
|---|---|---|
| `<DATASET>_responses.xlsx` | Stage 1 | Raw 9-model answers; sheets `preview`, `full_answers`, `long` |
| `<DATASET>_grades.xlsx` | Stage 2 | One TRUE/FALSE `verdict` per (question, model) |

Datasets included: `TruthfulQA`, `SimpleQA`, `FreshQA`, and the `TruthEval`
scarcity-check pilot.

---

## Requirements

**Python (Stages 1–2, and the embedder Stage 3 calls)**
- Python 3.10+
- `pip install openai tenacity pandas openpyxl requests sentence-transformers`
- OpenRouter API key in the environment variable `OPENROUTER_API_KEY`
  (Stages 1–2 only).

**MATLAB (Stages 3–5)**
- MATLAB R2021b or later (Statistics and Machine Learning Toolbox for `ranksum`).
- A Python environment with `sentence-transformers` installed and reachable from
  MATLAB (Stage 3 calls it through the Python bridge).
- Stage 5 additionally requires `my_ACSD.m` and `I_CD.m` on the MATLAB path.

---

## Pipeline

Stages run in order; each consumes the previous stage's output. The embedding
step is Stage 3.

### Stage 1 — Collect answers (Python)

```bash
export OPENROUTER_API_KEY=sk-or-...          # set your own key
python script_TruthfulQA_Stage1.py
```

Queries the nine models once per question at `T=0.2`, retries failures, and is
resumable (re-running fetches only missing model-question pairs). Writes
`<DATASET>_responses.jsonl` as answers arrive and `<DATASET>_responses.xlsx` at
the end with sheets `preview`, `full_answers`, and `long`.

### Stage 2 — Grade answers (Python)

```bash
python script_grade_TruthfulQA_Stage2.py
```

Grades each answer against the benchmark reference set with an LLM judge and
writes `<DATASET>_grades.xlsx` with one TRUE/FALSE `verdict` per
(question, model).

### Stage 3 — Embed answers and questions (MATLAB)

```matlab
% edit the python path and work_dir at the top of the script, then:
run script_Stage3_TruthfulQA.m
```

Reads the `long` sheet of `<DATASET>_responses.xlsx` and embeds with
`all-mpnet-base-v2` (768-dim), L2-normalizing every vector. Produces:

- `emb_raw_mpnet_<DATASET>.mat`
  - `Y` : 9x1 cell, each `[nQ x 768]`, one matrix per model, L2-normalized
  - `present_mask` : `9 x nQ` logical (false where a model gave no answer)
  - `meta` : source, model order, qids, normalization
- `emb_question_mpnet_<DATASET>.mat`
  - `Q` : `[nQ x 768]` question embeddings, L2-normalized
  - `qids`, `meta`

Fixed model order:
`{deepseek, gemini, haiku, qwen, gpt-mini, granite, llama, grok, stepfun}`.
Questions missing a model's answer get a zero vector with `present_mask=false`;
Stage 4 drops those so every retained cloud is a complete nine-model cloud.

**Embedding setup.** Stage 3 calls `sentence-transformers` through MATLAB's
Python bridge. Point MATLAB at a Python environment that has it installed by
editing the line at the top of the script:

```matlab
pyversion('C:\path\to\your\python.exe');   % env with sentence-transformers
```

The first run downloads the `all-mpnet-base-v2` weights. The helper
`get_sentence_embeddings(textCellArray, modelName)` wraps the encoder and must
be on the MATLAB path.

### Stage 4 — Per-state geometry (MATLAB)

```matlab
run script_Stage4_TruthfulQA.m
```

Loads the two embedding files, applies the presence filter (all nine models)
and the 300-word length filter, computes the per-question cloud descriptors,
bins each question by `n_wrong` into one of five consensus states, and prints
per-state means, per-state standard deviations, Mann-Whitney p-values
(AW vs MW and AW vs AR), and the rebound `Delta = mean(AW) - mean(MW)`.
Saves `stage4_by_state_<DATASET>.mat`.

Descriptors: `log_vol`, `D_geo`, `R_bar`, `log10_kappa`, `PR`, `H_spec`,
`phillips` (geometric/archetypal volume, paper symbol beta), `hamzah`
(mean pairwise cosine, paper symbol rho), `marin` (single-model query-answer
alignment, paper symbol alpha).

### Stage 5 — Sparse-decomposition check (MATLAB)

```matlab
run script_Stage5_TruthfulQA.m
```

Fits a per-question sparse dictionary to the mean-subtracted residual of the
nine answer embeddings and reports the consolidation index `s_max`, the
co-endorsement matrix, and model clustering. A complementary check reported
separately from the spectral geometry. Requires `my_ACSD.m` and `I_CD.m` on the
path; writes results under `out/`.

---

## Reproducing the results

1. Stages 1–2 are network-dependent and cost a few USD per dataset. To skip
   them, use the provided `<DATASET>_responses.xlsx` and `<DATASET>_grades.xlsx`.
2. Run Stage 3 per dataset to build the embeddings.
3. Run Stage 4 per dataset for the per-state geometry and statistics.
4. Run Stage 5 per dataset for the sparse-decomposition check.

The portable result is the **sign** of the cross-dataset effect (contraction on
TruthfulQA, expansion on SimpleQA and FreshQA), which is robust to encoder and
roster; absolute descriptor magnitudes are encoder- and roster-dependent.

---

## Notes

- **API key.** Stage 1 reads `OPENROUTER_API_KEY` from the environment. Do not
  commit a real key; set it in your shell before running.
- **Paths.** Each MATLAB script begins with a hard-coded `cd` to a local working
  directory. Edit it to your clone path.
- **Stage 5 dependencies.** `my_ACSD.m` must be on the MATLAB path.
