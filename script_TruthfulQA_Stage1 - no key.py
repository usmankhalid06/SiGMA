"""
script_pilot.py
Pilot data collection: query 9 LLMs on TruthfulQA via OpenRouter.

Roster (9 labs): DeepSeek, Google, Anthropic, Alibaba/Qwen, OpenAI, IBM Granite,
                 Allen AI (OLMo), Meta (Llama), Mistral.
  - Original 6 already collected; the resume logic skips them, so re-running
    this only fetches the 3 NEW models x N questions (cheap).

- Loads TruthfulQA from its public CSV (no datasets/pyarrow needed).
- Saves every response to pilot_responses.jsonl the moment it arrives.
- Resumes automatically; retries failed/empty answers.
- Reasoning effort = minimal, sent via extra_body. If a model rejects the
  reasoning param, the call automatically retries WITHOUT it. (OLMo-Instruct,
  Llama 4 Scout, and Mistral all fall back cleanly, keeping the run
  near-deterministic and tier-comparable to the original 6.)
- AT THE END: writes a readable pilot_responses.xlsx (preview / full_answers / long).

Run (in your clean conda env, Anaconda Prompt):
    conda activate pilot
    pip install openpyxl
    cd /d D:\\Courses\\Postdoc\\2026\\NewMethod
    set OPENROUTER_API_KEY=sk-or-v1-your-key
    python script_pilot.py
"""

import os
import io
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------

API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
if not API_KEY:
    API_KEY = "Enter your key here"   # do NOT hardcode a real key; set OPENROUTER_API_KEY instead

MODELS = [
    # --- original 6 (already collected; resume skips these) ---
    "deepseek/deepseek-v4-pro",
    "google/gemini-3.5-flash",
    "~anthropic/claude-haiku-latest",
    "qwen/qwen3.7-max",
    "~openai/gpt-mini-latest",          # OpenAI family (cheap mini)
    "ibm-granite/granite-4.1-8b",       # weak link
    # --- 3 new labs for the 9-model scaling (VERIFY exact slugs on openrouter.ai) ---
    "x-ai/grok-4.3",                    # xAI: served replacement for OLMo (OLMo has no live endpoints on OpenRouter)
    "meta-llama/llama-4-scout",         # Meta: "why not Llama?" defense
    "stepfun/step-3.7-flash",           # StepFun: replaces Mistral (new lab, served)
]

N_QUESTIONS = 817             # 5 for the test, then 1000 for the full run
MAX_TOKENS = 2000              # high cap: reasoning models spend tokens thinking before answering
TEMPERATURE = 0.2
CONCURRENCY = 8
OUTPUT_FILE = "TruthfulQA_responses.jsonl"
EXCEL_OUT   = "TruthfulQA_responses.xlsx"

TRUTHFULQA_CSV_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"

# ----------------------------------------------------------------------
# 2. SETUP
# ----------------------------------------------------------------------

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
_write_lock = threading.Lock()

def append_record(record: dict):
    with _write_lock:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_already_done() -> set:
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("answer"):
                        done.add((rec["question_id"], rec["model"]))
                except Exception:
                    continue
    return done

def load_truthfulqa(n):
    print("Downloading TruthfulQA CSV ...")
    r = requests.get(TRUTHFULQA_CSV_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    questions = []
    for i, row in df.iterrows():
        if i >= n:
            break
        questions.append({
            "question_id": int(i),
            "question": str(row["Question"]),
            "best_answer": str(row.get("Best Answer", "")),
            "correct_answers": str(row.get("Correct Answers", "")),
            "incorrect_answers": str(row.get("Incorrect Answers", "")),
        })
    return questions

def _short(m):
    return (m.replace("~", "").replace("mistralai/", "").replace("google/", "")
             .replace("anthropic/", "").replace("deepseek/", "")
             .replace("ibm-granite/", "").replace("qwen/", "").replace("openai/", "")
             .replace("allenai/", "").replace("meta-llama/", ""))

def write_excel():
    if not os.path.exists(OUTPUT_FILE):
        print("No data file yet; nothing to export.")
        return
    rows = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("No rows to export.")
        return
    try:
        from openpyxl.utils import get_column_letter
        from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
    except ModuleNotFoundError:
        print("\n*** openpyxl NOT installed — no Excel written. Run: pip install openpyxl ***")
        return

    df = pd.DataFrame(rows)
    df["model_short"] = df["model"].map(_short)
    ok = df[df["answer"].notna()].copy()

    wide_full = ok.pivot_table(index=["question_id", "question"], columns="model_short",
                               values="answer", aggfunc="first").reset_index()
    ok["preview"] = ok["answer"].str.slice(0, 140).str.replace("\n", " ", regex=False) + "…"
    wide_prev = ok.pivot_table(index=["question_id", "question"], columns="model_short",
                               values="preview", aggfunc="first").reset_index()

    with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
        wide_prev.to_excel(writer, sheet_name="preview", index=False)
        wide_full.to_excel(writer, sheet_name="full_answers", index=False)
        df.sort_values(["question_id", "model"]).to_excel(writer, sheet_name="long", index=False)

        wb = writer.book
        header_fill = PatternFill("solid", fgColor="4472C4")
        header_font = Font(bold=True, color="FFFFFF")
        thin = Side(style="thin", color="CCCCCC")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        for sheet_name, rowh in (("preview", 60), ("full_answers", 200)):
            ws = wb[sheet_name]
            ncols = ws.max_column
            for c in range(1, ncols + 1):
                cell = ws.cell(row=1, column=c)
                cell.fill = header_fill; cell.font = header_font
                cell.alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
            ws.column_dimensions["A"].width = 6
            ws.column_dimensions["B"].width = 40
            for c in range(3, ncols + 1):
                ws.column_dimensions[get_column_letter(c)].width = 50
            for r in range(2, ws.max_row + 1):
                for c in range(1, ncols + 1):
                    cell = ws.cell(row=r, column=c)
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                    cell.border = border
                ws.row_dimensions[r].height = rowh
            ws.freeze_panes = "C2"

    ncols = len([c for c in wide_full.columns if c not in ("question_id", "question")])
    print(f"\nEXCEL WRITTEN: {os.path.abspath(EXCEL_OUT)}")
    print(f"  preview & full_answers: {len(wide_full)} questions x {ncols} model columns")

# ----------------------------------------------------------------------
# 3. API CALL  (tries with reasoning; falls back to no-reasoning if rejected)
# ----------------------------------------------------------------------

@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=2, min=2, max=30),
       retry=retry_if_exception_type(Exception), reraise=True)
def call_model(model: str, question: str):
    msgs = [{"role": "user",
             "content": f"Answer the following question concisely and factually.\n\nQuestion: {question}"}]
    # StepFun's valid reasoning levels are high/medium/low (it rejects "minimal");
    # use "low" so it doesn't burn the whole token budget thinking and return empty.
    effort = "low" if model.startswith("stepfun/") else "minimal"
    try:
        resp = client.chat.completions.create(
            model=model, messages=msgs,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            extra_body={"reasoning": {"effort": effort}},
        )
    except Exception as e:
        # Some models reject the reasoning param — retry once without it.
        if "reasoning" in str(e).lower():
            resp = client.chat.completions.create(
                model=model, messages=msgs,
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            )
        else:
            raise
    answer = resp.choices[0].message.content
    if not answer or not answer.strip():
        raise ValueError("empty answer returned")
    provider = getattr(resp, "provider", None)
    return answer, provider

# ----------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------

def main():
    if API_KEY == "PASTE_YOUR_KEY_HERE" or not API_KEY:
        print("ERROR: No API key set. Set OPENROUTER_API_KEY in your environment.")
        return

    questions = load_truthfulqa(N_QUESTIONS)
    print(f"Loaded {len(questions)} questions.")
    done = load_already_done()
    print(f"Already collected: {len(done)} pairs. Resuming...")

    jobs = [(q, m) for q in questions for m in MODELS if (q["question_id"], m) not in done]
    print(f"Jobs remaining: {len(jobs)} (of {len(questions) * len(MODELS)} total)")

    if jobs:
        completed = failed = 0
        start = time.time()

        def worker(job):
            q, m = job
            try:
                answer, provider = call_model(m, q["question"])
                append_record({"question_id": q["question_id"], "question": q["question"],
                               "model": m, "answer": answer, "provider": provider,
                               "best_answer": q["best_answer"],
                               "correct_answers": q["correct_answers"],
                               "incorrect_answers": q["incorrect_answers"]})
                return "ok"
            except Exception as e:
                append_record({"question_id": q["question_id"], "question": q["question"],
                               "model": m, "answer": None, "error": str(e)[:300],
                               "best_answer": q["best_answer"],
                               "correct_answers": q["correct_answers"],
                               "incorrect_answers": q["incorrect_answers"]})
                return "fail"

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = [ex.submit(worker, job) for job in jobs]
            for fut in as_completed(futures):
                if fut.result() == "ok":
                    completed += 1
                else:
                    failed += 1
                tot = completed + failed
                if tot % 5 == 0 or tot == len(jobs):
                    el = time.time() - start
                    rate = tot / el if el > 0 else 0
                    rem = (len(jobs) - tot) / rate if rate > 0 else 0
                    print(f"  {tot}/{len(jobs)}  ok={completed} fail={failed}  ~{rem/60:.1f} min left")

        print(f"\nDone collecting. ok={completed} fail={failed}.")
        if failed:
            print("Some failed (often transient). Re-run to retry only the failures.")
    else:
        print("Nothing to collect — all present.")

    print(f"Raw data: {os.path.abspath(OUTPUT_FILE)}")
    write_excel()

if __name__ == "__main__":
    main()