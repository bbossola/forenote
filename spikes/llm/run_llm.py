"""Spike: llama-cpp-python with Gemma 4 (notes spec, section 6.3).

Usage: uv run python spikes/llm/run_llm.py MODEL.gguf [--ctx N] [--threads N]
Prints one JSON object per check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT = ROOT / "tests" / "fixtures" / "ami" / "transcripts" / "ES2010c.jsonl"
TEMPLATE_MARKS = ("<start_of_turn>", "<end_of_turn>", "<|turn", "<turn|>")
ACTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"owner": {"type": ["string", "null"]}, "text": {"type": "string"}},
        "required": ["owner", "text"],
    },
}


def rss_mib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def transcript_text(max_minutes: float) -> str:
    out = []
    with open(TRANSCRIPT, encoding="utf-8") as f:
        for raw in f:
            row = json.loads(raw)
            if row["t_start_ms"] > max_minutes * 60_000:
                break
            s = row["t_start_ms"] // 1000
            who = "Me" if row["speaker"] == "me" else "Others"
            out.append(f"[{s // 60:02d}:{s % 60:02d}] {who}: {row['text']}")
    return "\n".join(out)


def emit(**kw) -> None:
    print(json.dumps(kw), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--minutes", type=float, default=30.0)
    a = ap.parse_args()

    import llama_cpp
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    before = rss_mib()
    t0 = time.perf_counter()
    llm = llama_cpp.Llama(
        model_path=str(a.model), n_ctx=a.ctx, n_threads=a.threads,
        n_threads_batch=a.threads, n_batch=512, verbose=False,
    )
    template = llm.metadata.get("tokenizer.chat_template")
    emit(check="load", model=a.model.name, llama_cpp=llama_cpp.__version__,
         arch=llm.metadata.get("general.architecture"), ctx=a.ctx, threads=a.threads,
         load_s=round(time.perf_counter() - t0, 1), rss_mib=round(rss_mib() - before),
         has_template=template is not None)
    if template is None:
        return 1

    bos = llm._model.token_get_text(llm.token_bos())
    eos = llm._model.token_get_text(llm.token_eos())
    fmt = Jinja2ChatFormatter(template=template, eos_token=eos, bos_token=bos)

    def render(text: str) -> str:
        return fmt(messages=[{"role": "user", "content": text}]).prompt

    sample = render("XYZ")
    emit(check="template", head=sample[: sample.index("XYZ")], tail=sample[sample.index("XYZ") + 3 :])

    prefix = (
        "You write meeting notes. Use only facts from the transcript.\n\n"
        f"Transcript:\n{transcript_text(a.minutes)}\n\n"
    )

    def run(task: str, max_tokens: int, grammar=None) -> dict:
        tokens = llm.tokenize(render(prefix + task).encode("utf-8"), add_bos=False, special=True)
        t = time.perf_counter()
        first = None
        n = 0
        text = ""
        for chunk in llm.create_completion(tokens, max_tokens=max_tokens, temperature=0.2,
                                           stream=True, grammar=grammar):
            if first is None:
                first = time.perf_counter() - t
            n += 1
            text += chunk["choices"][0]["text"]
        total = time.perf_counter() - t
        first = first if first is not None else total
        return {
            "prompt_tokens": len(tokens),
            "first_token_s": round(first, 2),
            "gen_tokens": n,
            "gen_tok_s": round((n - 1) / (total - first), 1) if n > 1 and total > first else 0.0,
            "stopped_early": n < max_tokens,
            "template_text_in_output": any(m in text for m in TEMPLATE_MARKS),
            "text": text,
        }

    r1 = run("Write a summary of the meeting in 3 to 6 sentences.", 500)
    r1["prompt_tok_s"] = round(r1["prompt_tokens"] / r1["first_token_s"], 1)
    emit(check="first_task", **r1)
    r2 = run("Expand this bullet with facts from the transcript in 1 to 3 sentences: «remote control buttons».", 200)
    emit(check="second_task_same_prefix", **r2)
    grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(ACTION_SCHEMA), verbose=False)
    r3 = run("List the action items as JSON. Each item has owner (a name or null) and text.", 500, grammar)
    try:
        json.loads(r3["text"])
        r3["json_ok"] = True
    except json.JSONDecodeError:
        r3["json_ok"] = False
    emit(check="json_task_same_prefix", **r3)
    emit(check="total", rss_mib_end=round(rss_mib() - before))
    return 0


if __name__ == "__main__":
    sys.exit(main())
