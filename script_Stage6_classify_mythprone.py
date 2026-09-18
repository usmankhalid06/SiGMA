
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------
API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
if not API_KEY:
    API_KEY = "Place your API key here"   # or set OPENROUTER_API_KEY (as you did for Stage 1)

DATA_DIR = r"D:\Courses\Postdoc\2026\SIGMA\Matlab Codes"   # where *_responses.xlsx live and outputs go
DATASETS = ["TruthfulQA", "SimpleQA", "FreshQA"]

JUDGE       = "~openai/gpt-mini-latest"   # same slug your grader used (works on your account)
MAX_TOKENS  = 2000                         # one-word answer; generous headroom for reasoning
TEMPERATURE = 0.0
CONCURRENCY = 8

PROMPT = (
    "Does the following question concern a topic on which a popular but FALSE "
    "belief, misconception, myth, urban legend, misquotation, or a single "
    "canonical wrong answer is widely held, such that many people (or models "
    "trained on web text) would confidently give the same wrong answer?\n"
    "Answer NO if it is an obscure single fact, a recent/time-sensitive fact, or "
    "a topic with no single popular wrong answer.\n"
    "Reply with exactly one word: YES or NO.\n\n"
    "Question: {q}"
)

# ----------------------------------------------------------------------
# 2. SETUP
# ----------------------------------------------------------------------
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
_write_lock = threading.Lock()


def jsonl_path(ds): return os.path.join(DATA_DIR, f"{ds}_mythprone.jsonl")
def xlsx_path(ds):  return os.path.join(DATA_DIR, f"{ds}_mythprone.xlsx")
def resp_path(ds):  return os.path.join(DATA_DIR, f"{ds}_responses.xlsx")


def append_record(ds, record: dict):
    with _write_lock:
        with open(jsonl_path(ds), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done(ds) -> set:
    done = set()
    if os.path.exists(jsonl_path(ds)):
        with open(jsonl_path(ds), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("myth_prone") is not None:
                        done.add(int(rec["question_id"]))
                except Exception:
                    continue
    return done


def load_questions(ds):
    df = pd.read_excel(resp_path(ds), sheet_name="full_answers")
    df = df[["question_id", "question"]].drop_duplicates("question_id")
    return [{"question_id": int(r["question_id"]), "question": str(r["question"])}
            for _, r in df.iterrows()]


def parse(ans):
    if ans is None:
        return None
    a = str(ans).strip().lower()
    if a.startswith("yes"):
        return 1
    if a.startswith("no"):
        return 0
    if "yes" in a and "no" not in a:
        return 1
    if "no" in a and "yes" not in a:
        return 0
    return None


# --- same call pattern as your Stage 1 call_model (reasoning + fallback) ---
@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=2, min=2, max=30),
       retry=retry_if_exception_type(Exception), reraise=True)
def call_judge(question: str):
    msgs = [{"role": "user", "content": PROMPT.format(q=question)}]
    try:
        resp = client.chat.completions.create(
            model=JUDGE, messages=msgs,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            extra_body={"reasoning": {"effort": "minimal"}},
        )
    except Exception as e:
        if "reasoning" in str(e).lower():
            resp = client.chat.completions.create(
                model=JUDGE, messages=msgs,
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            )
        else:
            raise
    ans = resp.choices[0].message.content
    if not ans or not ans.strip():
        raise ValueError("empty answer returned")
    return ans


def write_excel(ds):
    if not os.path.exists(jsonl_path(ds)):
        return
    rows = []
    with open(jsonl_path(ds), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return
    df = pd.DataFrame(rows).drop_duplicates("question_id").sort_values("question_id")
    df = df[["question_id", "question", "myth_prone", "raw"]]
    df.to_excel(xlsx_path(ds), index=False)
    rate = df["myth_prone"].mean()
    print(f"[{ds}] done: {int(df['myth_prone'].notna().sum())}/{len(df)} classified, "
          f"myth_prone rate = {rate:.2f}  ->  {xlsx_path(ds)}")


def run(ds):
    questions = load_questions(ds)
    done = load_done(ds)
    jobs = [q for q in questions if q["question_id"] not in done]
    print(f"[{ds}] {len(questions)} questions, {len(done)} already done, {len(jobs)} to do")

    if jobs:
        completed = failed = 0
        start = time.time()

        def worker(q):
            try:
                raw = call_judge(q["question"])
                append_record(ds, {"question_id": q["question_id"], "question": q["question"],
                                   "myth_prone": parse(raw), "raw": raw})
                return "ok"
            except Exception as e:
                append_record(ds, {"question_id": q["question_id"], "question": q["question"],
                                   "myth_prone": None, "raw": f"ERROR: {str(e)[:200]}"})
                return "fail"

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = [ex.submit(worker, q) for q in jobs]
            for fut in as_completed(futures):
                if fut.result() == "ok": completed += 1
                else:                    failed += 1
                tot = completed + failed
                if tot % 25 == 0 or tot == len(jobs):
                    el = time.time() - start
                    rate = tot / el if el > 0 else 0
                    rem = (len(jobs) - tot) / rate if rate > 0 else 0
                    print(f"  [{ds}] {tot}/{len(jobs)}  ok={completed} fail={failed}  ~{rem/60:.1f} min left")

    write_excel(ds)


def main():
    if not API_KEY or API_KEY == "PASTE_YOUR_KEY_HERE":
        print("ERROR: set OPENROUTER_API_KEY in your environment (or paste it into API_KEY).")
        return
    for ds in DATASETS:
        run(ds)


if __name__ == "__main__":
    main()
