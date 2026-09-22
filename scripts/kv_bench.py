#!/usr/bin/env python3
# KV stress tool: simulates concurrent long-context workloads (large prompt + long output)
# usage: python3 kv_bench.py --concurrency 5 --prompt-tokens 4000 --max-tokens 500 [--name xxx]
import argparse, json, time, threading, urllib.request, re, statistics, sys

DEFAULT_BASE_URL = "http://localhost:8000/v1"
BASE = DEFAULT_BASE_URL          # overridden by --base-url in main()


def metrics_url():
    """Derive the Prometheus /metrics endpoint from the OpenAI-compatible base URL."""
    return BASE.rsplit("/v1", 1)[0].rstrip("/") + "/metrics"

def get_model():
    """Fetch the real model name from /v1/models instead of hard-coding it."""
    try:
        with urllib.request.urlopen(BASE + "/models", timeout=5) as r:
            d = json.loads(r.read().decode())
            return d["data"][0]["id"]
    except Exception:
        return "Qwen3.8-27B-NVFP4"

MODEL = None  # resolved in main() once --base-url is parsed

def snap():
    try:
        with urllib.request.urlopen(metrics_url(), timeout=5) as r:
            t = r.read().decode()
    except Exception:
        return None
    def g(name):
        vals = re.findall(rf"^{re.escape(name)}\{{[^}}]*\}} ([0-9.e+]+)", t, re.M)
        return sum(float(v) for v in vals) if vals else 0.0
    return {
        "run": g("vllm:num_requests_running"),
        "wait": g("vllm:num_requests_waiting"),
        "gen": g("vllm:generation_tokens_total"),
        "preempt": g("vllm:num_preemptions_total"),
        "kv": g("vllm:kv_cache_usage_perc"),
    }

def gen_prompt(tokens, salt=""):
    # Filler text used to pad the prompt to the requested token count. NOTE: the "1 char ≈ 1 token"
    base = "Model test document. This is a filler paragraph used for load testing; it discusses the scheduling behaviour of local inference engines and the KV cache management strategy."
    per = len(base)  # calibration property of the ORIGINAL CJK filler; this English filler tokenises at ~4 chars/token, so --prompt-tokens is now an overestimate
    reps = max(1, tokens // per)
    # Salt goes at the very front: it makes the content differ from the first token on and bypasses prefix-cache merging (simulates independent histories per request)
    return salt + "".join([base] * reps)

def send_req(prompt, max_tokens, res, idx):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read().decode())
        dt = time.time() - t0
        comp = d["usage"]["completion_tokens"]
        res[idx] = {"ok": True, "seconds": dt, "tokens": comp,
                    "tok_s": comp / dt, "ttft": None}
    except Exception as e:
        res[idx] = {"ok": False, "error": str(e)[:120]}
        return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--name", default="run")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="OpenAI-compatible base URL of the vLLM server (default: %(default)s)")
    args = ap.parse_args()

    global BASE, MODEL
    BASE = args.base_url.rstrip("/")
    MODEL = get_model()

    print(f"[{args.name}] concurrency={args.concurrency} prompt≈{args.prompt_tokens}tok max_out={args.max_tokens}tok")
    m0 = snap()
    if m0:
        print(f"  start: running={m0['run']:.0f} waiting={m0['wait']:.0f} kv={m0['kv']:.1%} preempt={m0['preempt']:.0f}")
    else:
        print("  start: /metrics not reachable - this engine publishes no Prometheus counters")

    prompts = [gen_prompt(args.prompt_tokens, salt=f"\n\n[request {i} unique content: analysis and execution plan for sub-task {i}, including a detailed argument for the solution and notes on implementing it in code.]") for i in range(args.concurrency)]
    res = [None] * args.concurrency
    threads = []
    for i in range(args.concurrency):
        t = threading.Thread(target=send_req, args=(prompts[i], args.max_tokens, res, i))
        threads.append(t)
        t.start()
        time.sleep(0.5)

    # the monitor thread samples every 2 s
    stop = threading.Event()
    samples = []
    def monitor():
        while not stop.is_set():
            s = snap()
            if s: samples.append(s)
            time.sleep(2)
    mt = threading.Thread(target=monitor, daemon=True)
    mt.start()

    for t in threads:
        t.join()
    stop.set()

    m1 = snap()
    ok = [r for r in res if r and r["ok"]]
    fail = [r for r in res if r and not r["ok"]]
    if m1:
        print(f"  end: running={m1['run']:.0f} kv={m1['kv']:.1%} preempt={m1['preempt']:.0f}")
        if m0:
            print(f"  preemption delta: {m1['preempt'] - m0['preempt']:.0f} (0 = no KV swapping)")
    else:
        print("  end: /metrics not reachable - this engine publishes no Prometheus counters")
    if samples:
        peak_kv = max(s["kv"] for s in samples)
        print(f"  KV pool peak occupancy: {peak_kv:.1%}")
    if ok:
        speeds = [r["tok_s"] for r in ok]
        print(f"  ok {len(ok)}/{args.concurrency}: mean {statistics.mean(speeds):.1f} tok/s | "
              f"min {min(speeds):.1f} | max {max(speeds):.1f} | median {statistics.median(speeds):.1f}")
        for i, r in enumerate(ok):
            print(f"    req{i}: {r['tok_s']:.1f} tok/s ({r['tokens']} tok / {r['seconds']:.1f}s)")
    if fail:
        for i, r in enumerate(fail):
            print(f"    req{i}: FAIL {r.get('error')}")
    _pd = f"{m1['preempt']-m0['preempt']:.0f}" if (m0 and m1) else "n/a"
    _pk = f"{peak_kv:.1%}" if samples else "n/a"
    print(f"  [RESULT:{args.name}] preempt_delta={_pd} "
          f"mean={statistics.mean([r['tok_s'] for r in ok]) if ok else 0:.1f} "
          f"peak_kv={_pk} ")

if __name__ == "__main__":
    main()
