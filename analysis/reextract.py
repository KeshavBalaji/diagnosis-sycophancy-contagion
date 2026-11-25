"""Re-run extraction on an existing single-agent run's replies with another extractor.

Writes <run>.ext-<model>.jsonl, one record at a time as each finishes; rerunning resumes (skips records already
written). Records that fail (e.g. --max-cost reached) are skipped and picked up on the next run.
"""
import argparse, asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from single_agent import UNCLEAR, Endpoint, extract, rank, read_run


def turns(r):
    yield r["turn0"]
    yield from r["successive"]
    yield from r["independent"][1:]
    yield r["sure"]
    yield from r["toward_correct"]


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--concurrency", type=int, default=32, help="extraction calls in parallel")
    p.add_argument("--max-cost", type=float, default=0.5)
    p.add_argument("--provider", default=None, help="OpenRouter host to pin (no fallbacks), e.g. Z.AI")
    a = p.parse_args()
    if a.base_url in ("openai", None):
        a.base_url = None
    ext = await Endpoint(a.base_url, a.model, a.reasoning_effort).resolve()
    ext.provider = a.provider
    Endpoint.max_cost, Endpoint.max_tokens = a.max_cost, 4000
    out = a.run.with_name(f"{a.run.stem}.ext-{a.model.split('/')[-1]}.jsonl")
    prev = read_run(out) if out.exists() else []
    out.write_text("".join(json.dumps(r) + "\n" for r in prev))  # drop a half-written line
    done = {(r["case_id"], r["seed"]) for r in prev}
    todo = [r for r in read_run(a.run) if (r["case_id"], r["seed"]) not in done]
    sem = asyncio.Semaphore(a.concurrency)
    errors = []

    async def one(r, t):
        async with sem:
            ddx, meta = await extract(ext, t["reply"])
        top = next((x for x in ddx if x != UNCLEAR), UNCLEAR)
        t.update(differential=ddx, diagnosis=top, correct=top == r["label"],
                 label_rank=rank(r["label"], ddx), distractor_rank=rank(r["distractor"], ddx),
                 extract_cost=meta["cost"], extract_version=meta["model_version"])

    with open(out, "a") as f:
        async def record(r):
            try:
                await asyncio.gather(*(one(r, t) for t in turns(r)))
            except Exception as e:
                errors.append(e)
                return
            r["independent"][0] = r["successive"][0]
            r["extractor"] = ext.model
            f.write(json.dumps(r) + "\n")
            f.flush()

        await asyncio.gather(*(record(r) for r in todo))
    n = len(read_run(out))
    print(f"wrote {out}: {n} records ({len(done)} resumed, {len(todo) - len(errors)} new) | cost ${Endpoint.spent:.4f}")
    if errors:
        print(f"{len(errors)} records not done (rerun to resume), e.g. {type(errors[0]).__name__}: {errors[0]}")


asyncio.run(main())
