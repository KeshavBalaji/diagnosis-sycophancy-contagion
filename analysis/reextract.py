"""Re-run extraction on an existing run's replies with another extractor; writes <run>.ext-<model>.jsonl."""
import argparse, asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from single_agent import Endpoint, extract, rank


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
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-cost", type=float, default=0.5)
    a = p.parse_args()
    if a.base_url in ("openai", None):
        a.base_url = None
    ext = await Endpoint(a.base_url, a.model, a.reasoning_effort).resolve()
    Endpoint.max_cost, Endpoint.max_tokens = a.max_cost, 4000
    rs = [json.loads(l) for l in open(a.run)]
    sem = asyncio.Semaphore(a.concurrency)

    async def one(r, t):
        async with sem:
            ddx, meta = await extract(ext, t["reply"])
        t.update(differential=ddx, diagnosis=ddx[0], correct=ddx[0] == r["label"],
                 label_rank=rank(r["label"], ddx), distractor_rank=rank(r["distractor"], ddx),
                 extract_cost=meta["cost"], extract_version=meta["model_version"])

    await asyncio.gather(*(one(r, t) for r in rs for t in turns(r)))
    for r in rs:
        r["extractor"] = ext.model
    out = a.run.with_name(f"{a.run.stem}.ext-{a.model.split('/')[-1]}.jsonl")
    with open(out, "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out} | cost ${Endpoint.spent:.4f}")


asyncio.run(main())
