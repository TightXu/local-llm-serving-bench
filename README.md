# Serving a 27B-class model on one RTX 5090: what three inference engines actually measured

One consumer GPU, one machine, and a question that kept coming back: what does it really cost to
serve a 27B-class model at home with long context and several requests in flight at once, and which
of the settings that get repeated online actually pay off? This repository is the answer in numbers.
It holds measurements, not software. Everything here came from running vLLM, NInfer and LM Studio on
a single RTX 5090 (32 GB, sm_120) under WSL2, and then writing down what came back, including the
results that contradicted the advice I had been following and the things I could not verify.

## What this is

A small, self-contained measurement log for local LLM serving on one consumer GPU, plus the tools
that produced it: an interactive launcher for `vllm serve` and a concurrency and KV-pressure script
that reads vLLM's own `/metrics` instead of guessing. There is no model code here, no fork of any
engine, and no model weights. The models are somebody else's work (see
[Credits and models](#credits-and-models)). What is mine is the setup, the measurements, the tools,
and the mistakes.

## Why it exists

Long-context local serving has a habit of looking fine in a benchmark and falling apart in use: a
request that decoded at 50 tokens/s drops to 0.5, and the reason turns out not to be the GPU but the
KV cache being oversubscribed, which makes the scheduler evict and recompute the longest request. I
wanted numbers I could act on: which KV cache dtype is actually worth it, whether CUDA graphs matter,
how deep a context a 32 GB card really fits, and where the engine I had been using was quietly
costing me time. The answers were specific enough to be useful and surprising enough to publish.

## Who it is for

Anyone running a large model on one consumer GPU, especially a 5090-class Blackwell card (sm_120)
under WSL2, where kernels for the newest compute capability are the first thing to break. Also
anyone deciding between a dense model and an MoE model of the same class, or between a serving
engine and a desktop inference application. If you are choosing a KV cache dtype, start with
[the KV cache dtype comparison](DESIGN.md#kv-cache-dtype-8-candidates-measured); if you are chasing
a throughput cliff, start with
[concurrency, context depth and CUDA graphs](DESIGN.md#concurrency-context-depth-and-cuda-graphs).
Everything should be read as *this machine, this build*; see
[Method and its limits](DESIGN.md#method-and-its-limits).

## The three engines, one line each

- **vLLM 0.28.0** - an open-source serving engine built around continuous batching and a paged KV
  cache, aimed at many requests in flight at once. It runs inside WSL2 and publishes Prometheus
  counters, which is where its numbers are read from.
- **NInfer** - an engine built from its own source tree that serves its own artifact format. It runs
  inside WSL2, takes a fixed KV capacity at start-up instead of deriving one from a utilisation
  fraction, supports speculative decoding, and returns per-request timings in the response body.
- **LM Studio** (application 0.4.23, bundled llama.cpp CUDA build 2.32.0) - a closed-source desktop
  application on the Windows side. It offloads weights to the GPU without continuous batching,
  exposes no CUDA graph switch, and has no timings API at all, so its numbers can only come from its
  own server log. It appears here only as a comparison point.

Only one engine can be resident at a time - each configuration needs roughly 28-31 GB of the 32 GB
card - so the three were started, measured and stopped one after another, never side by side.

## One measured round across the three engines

Same request surface for all three: one request in flight, temperature 0, 256 output tokens, median
of three measured requests after one discarded warm-up. Tokens/s, server-reported decode. The last
two vLLM columns are the same engine at the same points with and without speculative decoding.

| Depth asked for | Depth reached (prompt tokens) | vLLM, no spec decoding | vLLM, 3 draft tokens | NInfer, 3 draft tokens | LM Studio, draft head in the weights |
|---|---|---|---|---|---|
| 0 | 76 - 117 | 67.3 | 151.1 | 262.5 | 91.0 |
| 8,192 | 4,531 - 4,571 | 63.3 | 123.8 | 253.0 | 85.3 |
| 32,768 | 17,882 - 17,922 | 58.3 | 59.5 | 251.7 | 134.3 |
| 98,304 | 53,507 - 53,547 | 45.5 | 30.1 | 230.9 | 72.5 |

**Three statements survive this table and nothing further does:**

- **NInfer was the fastest engine at every depth**, by 3.9x to 5.1x against vLLM without
  speculative decoding, and it is the only engine whose speed barely moves with depth (262.5 at the
  shallowest point against 230.9 at the deepest).
- **Speculative decoding is a depth-dependent lever on the vLLM side, and it turns into a loss:**
  151.1 / 123.8 / 59.5 / 30.1 against 67.3 / 63.3 / 58.3 / 45.5, so +125% / +96% / +2% / **-34%**.
  Drafting adds work that scales with the context being attended over while the tokens it saves per
  step stay small; by the deepest point it costs about a third of the throughput it was meant to
  buy. The loss is stable to within about 4% across its three samples, so it is a reproducible
  property of this configuration rather than one slow run - and NInfer, same model family, same
  card, held 230.9 tokens/s at that depth with speculation on and an acceptance rate of 1.00, so
  this is an implementation behaviour here, not a law about the technique. That 1.00 needs reading
  carefully: it comes from the engine's own request logs, on a prompt whose output is one sentence
  repeated 20 times, which is about as predictable as text gets, and the engine's own documentation
  records 61-62% for the same 3-draft scheme on ordinary text. It is a property of this prompt at
  least as much as of the engine.
- **The fastest cell is not the shortest context.** LM Studio's best point is its third depth
  (134.3 against 91.0 at near-zero context), where its draft acceptance reached 0.99: decode speed
  under speculative decoding follows the acceptance rate, and context length is only one of the
  things that moves it.

**And five things make this four columns rather than one comparison:**

- **The speculative-decoding state and scheme differ.** NInfer ran with 3 draft tokens, LM Studio
  ran with a draft head shipped inside its weights, and vLLM appears twice because it ran both ways.
  The acceptance rate is engine-internal in each case, so it explains an engine's own speed and
  cannot be ranked across engines.
- **The weights and the KV cache dtype are not the same.** LM Studio served a Q4_K_M GGUF artifact
  with `q4_0` KV while the other two served pre-quantised NVFP4 (`int4_per_token_head` and `fp8`), so
  the engines were not even holding the same amount of context state per token - and part of any
  difference is weight format, which these runs cannot separate.
- **The depth labels are off by 45%.** The fill corpus is sized by characters, not tokens: the
  98,304-token request landed at 53,547 prompt tokens. All three engines received the same bytes, so
  the points stay comparable with each other, but the depth axis must not be read at face value.
- **Single stream, n = 3, no concurrency, and vLLM has no prefill figures at all**: it publishes no
  prefill counter and the client-side combination was never implemented, so its prefill cells are
  empty because they were never taken, not because prefill was slow.

Differences below the run-to-run band of roughly 10% are unresolved by this method; every
engine-to-engine difference published here is far outside it. The read-out rules for every cell, the
raw per-point records and the full list of what cannot be compared across the columns are in
[DESIGN.md](DESIGN.md), which is also where the numbers above come from. Every figure here has a
definition you can check, and several of them must not be compared with each other directly - that is
what DESIGN is for.

## Numbers from the earlier single-engine rounds

These come from the dense 27B-class model on vLLM unless stated otherwise, and are the ones the
tooling's own estimates are calibrated against.

| Question | Measured |
|---|---|
| KV pool cost per token, the engine's `auto` default against `fp8` | 37,948 against 37,703 bytes per token (37.06 against 36.82 KiB) - fp8 bought almost nothing |
| Longest single context in a pool of 3.0-3.6 GiB | ~86K tokens (engine default) / ~93K (fp8) / ~176K (4-bit per-token-head) |
| CUDA graphs off against on, single stream, 4K prompt, 500 output | 13.9 to 48.2 tokens/s (+247%), for 0.11-0.18 GiB of graph capture at that KV dtype (start-up log values; an earlier write-up of the same series used 0.15-0.51 GiB) |
| CUDA graphs off against on, 5 concurrent 48K prompts | 6.3 to 8.8 tokens/s in that round (+40%); two other graphs-on rounds of the same point read 6.3 and 13.7 - different runs, not comparable |
| KV pool peak occupancy, 5 concurrent 48K prompts, unique salts | ~51% with zero preemptions |
| The one observed collapse to 0.5 tokens/s | ~97% pool occupancy with preemptions; the pool, not the GPU |
| MoE 35B-A3B, 4 concurrent requests near zero depth | 120.9 tokens/s aggregate, 10,240 B KV per token - 69% less than the dense geometry |

The apparent ~8% capacity gain from fp8 is the one that needs stating carefully: the per-token pool
cost was almost identical (0.65% apart, and the engine's default was the higher of the two), and the
gain came from the pool itself being larger in that run. Most of the layers in this architecture are
linear attention (48 of 64 in the model measured), and their recurrent state is not governed by the
KV cache dtype setting at all, so quantising it changes less of the pool than the arithmetic suggests. The 4-bit option is a **capacity** recommendation; its
long-context output quality was **not measured here**, and that risk sits next to the capacity gain
rather than in a footnote. DESIGN chapters 3 and 4 carry the geometry, the memory ledger and the
full dtype table.

## What is in here

| Path | What it is |
|---|---|
| `DESIGN.md` | The measurement report: environment, protocol, the three-engine matrix, model geometry and the KV budget, the KV dtype comparison, concurrency and CUDA graphs, MoE against dense, and what was not verified |
| `docs/QUICKSTART.md` | The linear path from a bare WSL2 install to a serving endpoint and one benchmark request, with copyable commands |
| `docs/TOOLS.md` | The launcher and the pressure script, file by file: every parameter, every default, the VRAM estimator and what it assumes, the parameter-memory file, the failure logs |
| `scripts/` | The tools themselves: `select_and_run.py`, `kv_bench.py`, `chat.html`, `start_cg_on.sh`, `saved_params.json` |

Two directories in the working copy are deliberately **not** part of the published tree, and
`.gitignore` excludes both:

- `logs/` - the engines' own capture files, starting with the start-up sweep behind the KV cache dtype
  results. They are evidence rather than prose, but they carry machine-local paths and per-boot state
  and their labels are not in English, so they are not shipped. The sweep's numeric content is quoted,
  with its source, inside [DESIGN.md](DESIGN.md), and per-run logs are excluded for the same reason:
  they would need scrubbing before they could be published.
- `tools/` - the pre-publication gate this copy was checked with: local paths and usernames, private
  platform names, credential-shaped strings, CJK and emoji, file size. It stays out for the same class
  of reason - its pattern table lists the private names it screens for - and it remains a working-copy
  tool run before each push.

See [docs/TOOLS.md](docs/TOOLS.md) for what each file does and why it is or is not included.

## Quick start

[docs/QUICKSTART.md](docs/QUICKSTART.md) walks the whole path: hardware and driver prerequisites,
WSL2 networking, installing vLLM into the Linux environment, the two environment variables that
decide whether the newest kernels load at all, moving weights onto the Linux filesystem rather than
reading them from a Windows mount, the equivalent steps for NInfer and LM Studio, and then one
benchmark request whose output you can read line by line.

In short: `vllm serve` with the flags the measurements point to, plus an interactive launcher that
estimates VRAM before it starts, so a configuration that cannot fit fails fast instead of half a
minute into a weight load. That launcher remembers parameters per model, and the saved file is
perfectly capable of silently turning CUDA graphs off - which, not any model flag, was the single
largest lever in these measurements.

## Tools

[docs/TOOLS.md](docs/TOOLS.md) documents `scripts/` in full: the interactive launcher and its eleven
parameters, its two pre-flight checks, its VRAM estimator and the constants behind it, the
parameter-memory file and what it currently remembers, the concurrency and KV pressure script
(including why every request carries a unique salt and why its prompt sizing is an overestimate),
the single-file browser page, and the one-off start-up script that shows the minimum command for
CUDA graphs on.

## What this repository is not

It is not a benchmark of these engines - it is a record of one machine's behaviour under the
workloads I actually run. It does not measure output quality except where stated. It is not a
distribution of anyone's models: no weights are included. And it does not claim full
reproducibility: the environment was assembled over months and some details, notably the exact
virtual environment vLLM was installed into, are not recorded well enough to rebuild from these
files alone. Where that matters, the documents say so instead of guessing.

## Credits and models

- **vLLM**, the serving engine most of these measurements are about, is an open-source project
  (Apache-2.0) by the vLLM team. This repository is an independent record and is not affiliated
  with or endorsed by it.
- **FlashInfer** provides the attention kernels in this setup (Apache-2.0). Version and build
  details are in `DESIGN.md`.
- **Qwen models** were published by the Qwen team and obtained as pre-quantised **NVFP4** releases
  from **unsloth**, through Hugging Face and a ModelScope mirror. No weights are redistributed
  here; see the model cards for their licence terms, which were not read while writing this.
- **LM Studio** is a closed-source, commercially licensed application. It appears here only as a
  comparison point; nothing from it is included in this repository, and its licence terms were not
  read either.
- Found a number that contradicts your own measurements? That is the useful kind of issue: please
  include your hardware, driver, engine version and the exact flags.

## License

MIT - see [LICENSE](LICENSE). The measurements and text may be reused with attribution.
