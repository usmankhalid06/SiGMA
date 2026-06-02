# SiGMA — Spectral Geometry of Multi-model Agreement

Code and data for the paper **"When Consensus Hides Hallucination in Multi-LLM Systems."**

SiGMA queries nine commercial LLMs once each at low temperature on three
question-answering benchmarks (TruthfulQA, SimpleQA, FreshQA), embeds the nine
answers to each question into a single response cloud, and characterizes the
spectral and volumetric geometry of that cloud across five consensus states
(AR, MR, Sp, MW, AW). The central finding is a cross-dataset sign flip: on
TruthfulQA the all-wrong (AW) cloud contracts onto a geometry statistically
non-separable from correct consensus, while on SimpleQA and FreshQA it expands.
As an operational consequence, a query-type router (Stages 6–7) flags
misconception-prone questions from their text alone and routes them to
agreement-independent verification, reducing consensus-gate false-accepts on the
TruthfulQA AW clouds.

> Manuscript under review. Citation and DOI will be added on publication.

---

## Repository contents

The pipeline runs in seven stages. Scripts are named per dataset; the table
below uses TruthfulQA as the example, and SimpleQA / FreshQA versions follow the
same naming.

| File | Stage | Language | Role |
|---|---|---|---|
| `script_TruthfulQA_Stage1.py` | 1 | Python | Collect the 9 model answers via OpenRouter |
| `script_grade_TruthfulQA_Stage2.py` | 2 | Python | Grade each answer against the reference, write verdicts |
| `script_Stage3_TruthfulQA.m` | 3 | MATLAB + Python | Embed answers and questions with all-mpnet-base-v2 |
| `script_Stage4_TruthfulQA.m` | 4 | MATLAB | Per-state geometry, per-state means, p-values, rebound |
| `script_Stage5_TruthfulQA.m` | 5 | MATLAB | Sparse-decomposition check (s_max, co-endorsement) |
| `script_Stage6_classify_mythprone.py` | 6 | Python | Label each question myth-prone (content-only) via OpenRouter |
| `script_Stage7_router_eval.m` | 7 | MATLAB | Router vs consensus-gate evaluation (false-accepts, verification load) |

Data files (provided so Stages 1–2 and Stage 6 can be skipped):

| File | Produced by | Contents |
|---|---|---|
| `<DATASET>_responses.xlsx` | Stage 1 | Raw 9-model answers; sheets `preview`, `full_answers`, `long` |
| `<DATASET>_grades.xlsx` | Stage 2 | One TRUE/FALSE `verdict` per (question, model) |
| `<DATASET>_mythprone.xlsx` | Stage 6 | One binary `myth_prone` label per question (`question_id`, `question`, `myth_prone`, `raw`) |

Datasets included: `TruthfulQA`, `SimpleQA`, `FreshQA`, and the `TruthEval`
scarcity-check pilot.

---

## Requirements

**Python (Stages 1–2, Stage 6, and the embedder Stage 3 calls)**
- Python 3.10+
- `pip install openai tenacity pandas openpyxl requests sentence-transformers`
- OpenRouter API key in the environment variable `OPENROUTER_API_KEY`
  (Stages 1–2 and Stage 6).

**MATLAB (Stages 3–5, Stage 7)**
- MATLAB R2021b or later (Statistics and Machine Learning Toolbox for `ranksum`
  and `quantile`).
- A Python environment with `sentence-transformers` installed and reachable from
  MATLAB (Stage 3 calls it through the Python bridge).
- Stage 5 additionally requires `my_ACSD.m` and `I_CD.m` on the MATLAB path.
- Stage 7 reads the Stage 4 `.mat` files and the Stage 6 `<DATASET>_mythprone.xlsx`
  files; no extra toolbox beyond the above.

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
alignment, paper symbol alpha). The cluster-based consensus baselines
(`H_clu`, `c_maj`, `K_clu`) are also computed here from the nine answer
embeddings.

### Stage 5 — Sparse-decomposition check (MATLAB)

```matlab
run script_Stage5_TruthfulQA.m
```

Fits a per-question sparse dictionary to the mean-subtracted residual of the
nine answer embeddings and reports the consolidation index `s_max`, the
co-endorsement matrix, and model clustering. A complementary check reported
separately from the spectral geometry. Requires `my_ACSD.m` and `I_CD.m` on the
path; writes results under `out/`.

### Stage 6 — Myth-prone query classifier (Python)

```bash
export OPENROUTER_API_KEY=sk-or-...          # set your own key
python script_Stage6_classify_mythprone.py
```

Built on the same engine as Stage 1 (OpenAI SDK to OpenRouter,
`ThreadPoolExecutor`, `tenacity` retries, resumable). For each dataset it reads
the `full_answers` sheet of `<DATASET>_responses.xlsx` and sends **one LLM call
per question, on the question text only** (never the answers or the grades),
asking whether the question admits a popular canonical wrong answer. The label
is therefore content-only and cannot depend on the all-wrong outcome it is later
used to flag, which keeps the router non-circular.

Writes `<DATASET>_mythprone.jsonl` as labels arrive and
`<DATASET>_mythprone.xlsx` at the end, with columns `question_id`, `question`,
`myth_prone` (1/0), and `raw` (the model's reply). Processes all three datasets
in order (TruthfulQA, SimpleQA, FreshQA) and prints the `myth_prone` rate per
dataset. The judge model and concurrency are set at the top of the script.

### Stage 7 — Router vs consensus-gate evaluation (MATLAB)

```matlab
run script_Stage7_router_eval.m
```

Loads `stage4_by_state_<DATASET>.mat` (grade-derived consensus states and the
consensus signal `hamzah` = rho) and `<DATASET>_mythprone.xlsx`, and joins them
on `question_id`. Defines two policies, with thresholds calibrated on AR only so
neither peeks at the all-wrong cell it is evaluated on:

- **Consensus gate** — accepts an answer when rho clears a per-dataset threshold
  calibrated to pass 90% of correct-consensus (AR) clouds.
- **Router** — accepts only when the gate fires **and** the query is not
  myth-prone; all myth-prone queries are sent to agreement-independent
  verification regardless of cloud tightness.

On the AW clouds the consensus answer is wrong, so an accept is a dangerous
false-accept. Prints, per dataset and pooled, the false-accept rate of each
policy, the resulting verification load, and the share of AW clouds the
content-only classifier flagged (its recall on the dangerous cells).

---

## Reproducing the results

1. Stages 1–2 are network-dependent and cost a few USD per dataset. To skip
   them, use the provided `<DATASET>_responses.xlsx` and `<DATASET>_grades.xlsx`.
2. Run Stage 3 per dataset to build the embeddings.
3. Run Stage 4 per dataset for the per-state geometry and statistics.
4. Run Stage 5 per dataset for the sparse-decomposition check.
5. Run Stage 6 to label questions myth-prone (or skip it using the provided
   `<DATASET>_mythprone.xlsx`).
6. Run Stage 7 for the router vs consensus-gate evaluation.

The portable result is the **sign** of the cross-dataset effect (contraction on
TruthfulQA, expansion on SimpleQA and FreshQA), which is robust to encoder and
roster; absolute descriptor magnitudes are encoder- and roster-dependent. The
router (Stages 6–7) is a proof-of-concept that the characterization can be
operationalized, not a calibrated production solution.

---

## Notes

- **API key.** Stages 1–2 and Stage 6 read `OPENROUTER_API_KEY` from the
  environment. Do not commit a real key; set it in your shell before running.
- **Paths.** Each MATLAB script begins with a hard-coded `cd` to a local working
  directory. Edit it to your clone path.
- **Stage 5 dependencies.** `my_ACSD.m` and `I_CD.m` must be on the MATLAB path.
- **Stage 7 filename.** If the script is saved as `scrtip_Stage7_router_eval.m`
  (a typo), rename it to `script_Stage7_router_eval.m` to match this README.
