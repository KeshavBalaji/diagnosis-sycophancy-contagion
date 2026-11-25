"""Poll the final runs every 60 s: progress, cost, health. Kills all runs on a stop condition."""
import json, os, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "runs"
TARGET = {"single": 625, "panel_primary": 300, "panel_severe": 25, "relay_primary": 300, "relay_severe": 25}
SHORT = {"single": "single", "panel_primary": "panel-p", "panel_severe": "panel-s", "relay_primary": "relay-p", "relay_severe": "relay-s"}
MAX_TOTAL = 15.5


def walk(x, acc):
    if isinstance(x, dict):
        for k in ("cost", "extract_cost"):
            if isinstance(x.get(k), (int, float)):
                acc["cost"] += x[k]
        if "finish_reason" in x:
            acc["replies"] += 1
            acc["cut"] += x["finish_reason"] == "length"
            if "diagnosis" in x:
                acc["dx"] += 1
                acc["unclear"] += x["diagnosis"] == "unclear"
        for v in x.values():
            walk(v, acc)
    elif isinstance(x, list):
        for v in x:
            walk(v, acc)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


pids = {n: int(p) for n, p in (l.split() for l in open(ROOT / "final.pids"))}
stall = {n: 0 for n in pids}
last = {n: 0 for n in pids}


def kill_all(reason):
    for p in pids.values():
        if alive(p):
            os.kill(p, signal.SIGTERM)
    print(f"STOPPED ALL RUNS: {reason}. Finished cases are saved; resume with runs/launch_final.sh after fixing.", flush=True)
    sys.exit(1)


while True:
    time.sleep(60)
    tot = {"cost": 0.0, "replies": 0, "cut": 0, "dx": 0, "unclear": 0}
    parts, warns = [], []
    for n in pids:
        f, log = ROOT / f"final_{n}.jsonl", ROOT / f"final_{n}.log"
        acc = {"cost": 0.0, "replies": 0, "cut": 0, "dx": 0, "unclear": 0}
        done = 0
        if f.exists():
            for line in open(f):
                try:
                    walk(json.loads(line), acc)
                    done += 1
                except json.JSONDecodeError:
                    pass
        for k in tot:
            tot[k] += acc[k]
        logtxt = log.read_text(errors="ignore") if log.exists() else ""
        retries = logtxt.count("Retrying request")
        up = alive(pids[n])
        stall[n] = stall[n] + 1 if (up and done == last[n]) else 0
        last[n] = done
        state = "" if up else (" DONE" if done >= TARGET[n] else " EXITED")
        parts.append(f"{SHORT[n]} {done}/{TARGET[n]} ${acc['cost']:.2f}{state}")
        if retries:
            warns.append(f"{SHORT[n]} {retries} retries")
        if "Traceback" in logtxt and not up:
            warns.append(f"{SHORT[n]} crashed (see runs/final_{n}.log)")
        if " failed (rerun" in logtxt:
            warns.append(f"{SHORT[n]}: " + next(l for l in logtxt.splitlines() if " failed (rerun" in l)[:120])
        if stall[n] >= 3:
            warns.append(f"{SHORT[n]} no progress for {stall[n]} min")
    running = sum(alive(p) for p in pids.values())
    cut = tot["cut"] / max(tot["replies"], 1)
    unc = tot["unclear"] / max(tot["dx"], 1)
    print(f"[{time.strftime('%H:%M')}] " + " | ".join(parts) + f" | total ${tot['cost']:.2f} | cut-off {cut:.1%}, "
          f"no diagnosis {unc:.0%} | {running}/5 running" + ("" if not warns else " | WARN: " + "; ".join(warns)), flush=True)
    if tot["cost"] > MAX_TOTAL:
        kill_all(f"total cost ${tot['cost']:.2f} > ${MAX_TOTAL}")
    if tot["replies"] >= 200 and cut > 0.05:
        kill_all(f"{cut:.1%} of replies cut off at the token limit")
    if tot["dx"] >= 200 and unc > 0.30:
        kill_all(f"{unc:.0%} of replies have no usable diagnosis")
    for n in pids:
        log = (ROOT / f"final_{n}.log")
        if log.exists() and (m := [l for l in log.read_text(errors="ignore").splitlines() if " failed (rerun" in l]):
            if int(m[0].split()[0]) >= 10:
                kill_all(f"{SHORT[n]}: {m[0][:80]}")
    if running == 0:
        print("ALL RUNS FINISHED.", flush=True)
        break
