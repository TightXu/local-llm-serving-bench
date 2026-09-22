# DESIGN - local LLM serving measurements on one RTX 5090 (vLLM, NInfer, LM Studio)

<!--
Status (2026-09-23). All nine chapters are written. Chapters 4-8 were written in a second pass
from the same body of internal source material as chapters 1-3, and chapter 6 in a third pass from
that material plus the per-point records of the switch runs; that material is not part of this
repository and is referred to there by an English shorthand, defined at the head of the chapter
that first uses it.

Sources are named inline in HTML comments next to the figures they support, so they can be
stripped before publication. A source that is not in this repository is named by its own
file name only - never by a path on the machine that produced it.

Pending material, by chapter:
  ch2 updated: the vLLM point with speculative decoding is measured and published as a second
       vLLM column, so all four columns now carry a speculative-decoding state. Still missing:
       a per-depth draft-acceptance reading on the vLLM side and the preemption counters for
       those points, so the deep-context collapse cannot be separated between draft overhead
       and pool pressure from the retained evidence. The NInfer column's acceptance figure is not
       in the per-point records either - the harness left that field empty for that engine - so it
       is published with its source (the engine's own request lines) and with the limit that goes
       with the prompt it was measured on; see the acceptance-rate section of chapter 2.
  ch6 note: the per-engine switch points of the protocol are published in chapter 6, one point per
       setting under the comparison protocol. What that chapter still lacks is named there and in the
       last chapter: no combination of two switches, no under-load pool occupancy or preemption
       counter for a switch point, and no single-variable fp8-KV point at the unchanged request
       ceiling.
  ch4 pending: the polling driver that produced the seven in-round start-up logs was not retained,
       so those points cannot be re-run as they stand. The eighth row of the table was rejected in
       an earlier round and is marked as such wherever it appears.
  ch5 pending: a same-session pair for short-context single-stream decode, so that the two read-out
       paths a chapter apart can be separated; and a retained read-out rule for the stress series.
  ch7 pending: the second geometry under the identical-request-surface protocol of chapter 2, and a
       calibrated per-token constant for it in the launcher's estimator.
  ch8 note: the source material's earlier attribution of one engine's memory behaviour was corrected
       upstream of this repository. Only the corrected version appears in chapter 8; the superseded
       explanation appears nowhere in this document.
  ch9 pending: licence terms of the quantised weights and of the closed-source comparison
       application (their model cards were not read); upstream issue numbers quoted by the source
       material were never checked against a link, so no issue number appears in the published text
       - only the mechanism each one described.
-->

Three inference engines, one consumer GPU, and the same request sent to each of them. This
document is the measurement report: what was set, what came back, and which parts of the
result the method itself cannot support. Every chapter follows the same shape -
**Method -> Result -> What it means** - and reproduces the measurement tables in full rather
than summarising them.

## How this document is organised

| # | Chapter | What it contains |
|---|---|---|
| 1 | Scope, hardware and measurement method | The three engines and their roles, the machine and software, the configuration snapshot, the request protocol, how each engine's numbers are read, and the limits of the method. **Written.** |
| 2 | Three-engine comparison | The comparison matrix: three engines across four context depths, single stream, one warm-up plus three measured requests per point, the speculative-decoding state and acceptance rate at every point, and the read-out rules. **Written.** |
| 3 | Model geometry and the KV budget | The per-token KV formula for the two model geometries measured, the pool cost a dtype switch actually controls, the memory ledger the launcher uses, the utilisation convention stated once, and the correction that the recurrent state is not governed by the KV cache dtype setting. **Written.** |
| 4 | KV cache dtype: 8 candidates measured | The eight-row comparison of KV cache dtypes (bytes per token, pooled capacity, longest single context, start-up outcome), the counter-intuitive result and its root cause, and the explicit note that the 4-bit option is a capacity recommendation whose output quality was not measured. **Written.** |
| 5 | Concurrency, context depth and CUDA graphs | The concurrency ladder, the preemption mechanism behind a long request collapsing to a fraction of a token per second, the safe pool-occupancy ceiling, and the CUDA graph on/off pair. **Written.** |
| 6 | Switch matrix: one setting changed at a time | The per-engine switch points under the comparison protocol - CUDA graphs, KV cache dtype, speculative-decoding scheme, and one model swap - each at four depths, with its change against that engine's own baseline column and the capacity, memory or prefill cost it carries; plus the readings that must not be taken from it: acceptance rates near 100% that are a property of this prompt, a KV row that moved two variables, a model swap that is not an MTP measurement, and one depth whose median is a range. **Written.** |
| 7 | MoE vs dense on one 32 GB card | Speed against context depth for both model geometries, the memory geometry that lets the MoE hold deeper contexts, and the honest limits of that comparison. **Written.** |
| 8 | What each engine actually optimises for | Why the three engines behave differently: batching and prefix handling, memory bookkeeping, prefill strategy, and which of those differences survive once the request surface is held identical. Every inference is marked as one. **Written.** |
| 9 | What we did not verify | Every figure that came from elsewhere, the output quality of the 4-bit KV option, licence terms, and the upstream issue numbers the source material quoted without a link. **Written.** |

## Scope, hardware and measurement method

<!-- status: written in this pass. Sources: 3engine-benchmark-protocol-20260922.md
(variable control, per-engine read-out definitions), 3engine-benchmark-results-20260922.md
(measured hardware, configuration snapshot, limitations), bench/vllm-v2.json,
bench/ninfer-v2.json and bench/lmstudio.json (raw per-point records),
lmstudio-aggregate-20260922.md (log-derived statistics and their own limits). -->

### What is compared, and what is not

Three inference engines ran the same model family on the same single consumer GPU: vLLM under
WSL2, NInfer under WSL2, and LM Studio on the Windows side. The comparison in the next chapter
is deliberately narrow - one request in flight at a time, four context depths, temperature 0,
a fixed output length of 256 tokens, three measured repetitions per point, median reported. It
answers one question: under identical request conditions, how fast does each engine decode and
how fast does it ingest a prompt. Concurrent throughput, output quality, and contexts outside
the interval tested here are out of scope and are listed as such in the last chapter.

Only one engine can be resident at a time. Each of the three configurations below needs
roughly 28 to 31 GB of device memory and the card has 32 GB in total, so the engines were
started, measured and stopped one after another, never side by side.

### The three engines

- **vLLM** (0.28.0) - an open-source serving engine built around continuous batching and a
  paged KV cache, aimed at many requests in flight at once. It runs inside WSL2 (it does not
  support a native Windows install) and publishes Prometheus counters at `/metrics`, which is
  where its numbers are read from.
- **NInfer** - an engine built from its own source tree (build dated 2026-09-16) that serves
  its own `.ninfer` artifact format. It runs inside WSL2, takes a fixed KV capacity at
  start-up rather than deriving it from a utilisation fraction, supports speculative decoding,
  and returns per-request timings inside the response body.
- **LM Studio** (application 0.4.23 with its bundled llama.cpp CUDA build 2.32.0) - a
  closed-source desktop application. It runs on the Windows side, offloads weights to the GPU
  without continuous batching, exposes no user-facing switch for CUDA graphs, and has no
  timings API at all, which is why its numbers have to come from its server log. It appears in
  this repository only as a comparison point; nothing from it is redistributed here.

### Hardware

| Item | Value |
|---|---|
| GPU | RTX 5090, 32 GB GDDR7, compute capability 12.0 (sm_120, consumer Blackwell) |
| CPU | Ryzen 9 9950X3D |
| Host system | Windows 11 |
| Linux environment | WSL2 on Ubuntu 24.04, for vLLM and NInfer |
| Host memory | 32 GiB |

### Software

| Component | Version as measured |
|---|---|
| vLLM | 0.28.0 |
| FlashInfer (attention kernels) | 0.6.16.post3 |
| torch | 2.13.0+cu130 |
| CUDA toolkit | 13.2 (the `nvcc` first on the default PATH was 12.0; the kernel JIT builds use the 13.2 toolkit) |
| LM Studio | application 0.4.23, llama.cpp CUDA build 2.32.0 |
| NInfer | built from source, build dated 2026-09-16 |

Versions drift, and this matters more than it sounds: a default that changes upstream, a kernel
that is replaced, or a counter that is renamed all break reproducibility while leaving the
command line untouched. Everything in this repository describes one machine and one build set
in September 2026.

### Model artifacts

| Engine | Artifact served | Notes |
|---|---|---|
| vLLM | Qwen3.8-27B-NVFP4, 21.3 GiB | pre-quantised NVFP4 release: 4-bit weights and activations with FP8 attention |
| NInfer | the same model family as a `.ninfer` artifact, 22.1 GiB | engine-native artifact format, not an interchange format |
| LM Studio | Qwen3.8-27B Q4_K_M GGUF, 16.8 GiB | a different quantisation from the other two |

All three serve the 27B dense member of one model family, and that shared identity is what makes
the comparison worth making. The quantisation is *not* identical, and that is a limit of the
comparison rather than a detail inside it: Q4_K_M GGUF on one side against NVFP4 on the other
two means part of any measured difference is weight format, not engine.

### Engine configuration snapshot

The settings below decide most of what an engine can do on this card, and they are the columns
that must be quoted next to any number taken from this repository.

| | vLLM | NInfer | LM Studio |
|---|---|---|---|
| Weight format | NVFP4, pre-quantised | NVFP4, `.ninfer` artifact | Q4_K_M GGUF |
| KV cache dtype | `int4_per_token_head` | `fp8` | `q4_0`, keys and values |
| Context ceiling (tokens) | 114,688 | 196,608 | 201,000 |
| Speculative decoding | off in the comparison series, on (3 draft tokens) in the second vLLM series | on, 3 draft tokens | on, draft head shipped inside the weights |
| CUDA graphs | on (engine default) | on (engine default) | no user-facing switch |
| Prefix caching | disabled | not exposed | not exposed |
| Memory budget setting | `gpu-memory-utilization 0.94` | `kv-capacity 196608` | GPU offload at maximum |
| Device memory in use during the run | about 30.6 GB | about 30.2 GB | about 28.8 GB |

Two of these rows need context. NInfer's context ceiling is a per-request logical limit and does
not by itself occupy memory; the pool is decided by its KV capacity setting. LM Studio's context
ceiling comes from a per-model configuration and is the reason it can hold a 200K-class context
where the other two configurations cannot: the engines account for GPU memory differently, and
that bookkeeping difference is the subject of a later chapter. The measured footprint column is
the whole device memory in use while serving, not the weights alone.

### The request protocol

One request surface for all three engines, differing only in the model identifier:

1. **Fill corpus.** One fixed text block, repeated to the requested length, byte-identical
   across the three engines. A different corpus would be a different measurement.
2. **Salt.** Every request carries a unique salt at the head of the prompt, and vLLM
   additionally ran with prefix caching disabled. Without this, repeated requests hit a prefix
   or context cache and the prefill figures measure the cache instead of the engine.
3. **Depths.** 0, 8,192, 32,768 and 98,304 tokens requested. These are nominal depths; what they
   turned into is one of the limits below.
4. **Sampling.** Temperature 0, `top_p` and `top_k` left unset, so no sampling drift is added
   on top of a comparison that is already narrow.
5. **Output length.** `max_tokens` 256 with `ignore_eos` set, and a prompt that asks for a fixed
   sentence to be repeated, so all three engines actually generate the full 256 tokens. Decode
   rate depends on output length, and a short answer is not a rate.
6. **Thinking mode.** Off for all three. vLLM and LM Studio accept a request-level directive for
   this; NInfer's equivalent is a start-up flag. That is an asymmetry we accepted rather than a
   setting we could align, and it is recorded here so nobody has to guess later.
7. **Concurrency.** Single stream, requests issued one after another, with each engine's own
   request limit set to 1 where it has one.
8. **Repetition.** One warm-up request per point, discarded, then three measured requests; the
   published figure for a point is the median of the three. The per-point records kept alongside
   this document carry the median and the sample count, not the three individual values - those
   were printed by the harness at run time and only one pair of points has them in the run
   write-up, which is stated where that pair is quoted. A peak is never used as a comparison
   value, and a single run is never used as a cell.
9. **Rejection rules.** A point is void if the completion is shorter than 32 tokens, if the
   reported rate exceeds 400 tokens/s (not possible for a 27B model on this card, hence read as
   a parse artifact), or if vLLM reports a non-zero preemption count during the request, which
   means the KV pool was oversubscribed and the point is not a clean measurement. An earlier
   round of this experiment returned 2 and 39 completion tokens because the prompt ended with a
   one-word instruction; that round was discarded in full and is kept only as evidence of the
   defect, not as data.

### How each engine's numbers were read

The three engines do not expose the same numbers, and this is where a comparison quietly goes
wrong. The rule applied here: the figure comes from the engine, not from the client. Where an
engine reports its own speed, that is the published figure. The client-side wall-clock rate is
kept as a secondary column only, because it includes network time, JSON parsing and time to
first token, none of which is engine speed.

| Engine | decode | prefill | other fields recorded |
|---|---|---|---|
| vLLM | first-order difference of the engine's own Prometheus counters across the request: `1 / (delta request_time_per_output_token_seconds_sum / delta request_time_per_output_token_seconds_count)` | not captured | delta of `generation_tokens_total`, delta of `num_preemptions_total`, `kv_cache_usage_perc` |
| NInfer | `timings.predicted_per_second` in the response body | `timings.prompt_per_second` | `draft_n`, `draft_n_accepted` |
| LM Studio | numeric fields of the server-log line that reports evaluation time in milliseconds and tokens per second | the matching prompt-evaluation numeric field | draft acceptance ratio, CUDA graph reuse count |

Two consequences of that table are worth stating plainly. LM Studio has no timings API, so its
figures can only be extracted from its server log; that makes it the weakest of the three
sources, and it is the reason the extraction is scripted rather than read off the interface.
Only numeric fields are read: no log line, no request or response body and no filesystem path
from that log appears in this repository or anywhere it points to, and the logging switch that
produced the timing fields was turned back off after the run. Separately, vLLM publishes no
independent prefill counter; the plan was to combine client time-to-first-token with the delta
of `generation_tokens_total`, that combination was not implemented, and so vLLM has decode
figures only. Its prefill cells are empty because they were never taken - not because the
prefill was slow.

### Method and its limits

1. **Nominal depth is not achieved depth.** The fill corpus is sized by characters, not tokens.
   A request for 98,304 tokens landed at 53,547 prompt tokens on two engines and 53,507 on the
   third; the 32,768 request landed near 17,900. Because all three engines receive the same
   bytes, the points remain comparable with each other, but the depth axis must not be read at
   face value. Every depth quoted in this repository is the prompt token count the engine itself
   reported, not the number that was requested.
2. **Speculative decoding is not symmetric across the three engines.** NInfer ran with 3 draft
   tokens, LM Studio ran with a draft head shipped inside its weights, and vLLM ran twice: once
   without speculative decoding and once with it, 3 draft tokens. Decode speed under speculative
   decoding is coupled to the draft acceptance rate rather than to context length alone: the
   fastest point in the whole matrix was not the shortest context but the point with the highest
   acceptance rate, and vLLM's own gain from the same feature falls to nothing by the third depth
   and turns negative at the fourth. With both vLLM series published, the matrix supports the three
   statements listed in the next chapter and nothing further; it does not support one ordering of
   the three engines.
3. **The weights are not the same.** LM Studio served a Q4_K_M GGUF artifact while the other two
   served NVFP4. "Same protocol" applies to the request and sampling surface, not to the weights.
4. **Single stream, three repetitions, no concurrency.** These figures describe one request at a
   time. They cannot be extrapolated to several requests in flight. The switch matrix the protocol
   called for - CUDA graph on and off, KV cache dtype, speculative decoding, each as its own point -
   was not run in this round; it is chapter 6.
5. **One engine's figures depend on a log format.** LM Studio's numbers exist only because its
   server log prints timing information in a parseable shape. A formatting change upstream, on a
   closed-source application no one here controls, would silently remove the source of those
   numbers.
6. **Repeated runs and reboots move the numbers.** Identical configurations have differed by
   roughly 10% from one boot to the next, and pooled KV figures by roughly 7% between runs.
   Differences smaller than that are noise here, and nothing in this repository rests on a
   difference of that size.

## Three-engine comparison

<!-- Sources: 3engine-benchmark-results-20260922.md (final three-engine table, the vLLM
speculative-decoding sweep, the limitations list), bench/vllm-v2.json, bench/vllm-mtp.json,
bench/ninfer-v2.json and bench/lmstudio.json (per-point records: n = 3, median, derived client-side
rate), 3engine-benchmark-protocol-20260922.md (read-out definitions, not restated here). Every
ratio, percentage and capacity quoted below was recomputed from the numbers in this chapter.
Still pending: a per-depth draft-acceptance reading and the preemption counters for the vLLM
speculative series, and the dtype switch at this chapter's own request ceiling, which would make the
KV row single-variable. The per-engine switch matrix itself is chapter 6. The depth axis published
here is the prompt token count each engine reported, never the nominal label it was asked for. -->

This is the chapter the rest of the document supports. Three engines, four context depths, one
request in flight at a time, temperature 0, 256 output tokens, median of three measured requests
after one discarded warm-up.

### The matrix

Decode, server-reported, tokens per second (median of 3). Two of the four columns are the same
engine at the same points and must be read as a pair: the first vLLM column is the engine without
speculative decoding, the second is the same engine with 3 draft tokens.

| Nominal depth | vLLM, no speculative decoding | vLLM, 3 draft tokens | NInfer, 3 draft tokens | LM Studio, draft head inside the weights |
|---|---|---|---|---|
| 0 | 67.3 | 151.1 | 262.5 | 91.0 |
| 8,192 | 63.3 | 123.8 | 253.0 | 85.3 |
| 32,768 | 58.3 | 59.5 | 251.7 | 134.3 |
| 98,304 | 45.5 | 30.1 | 230.9 | 72.5 |

Three statements survive this table, and nothing further does.

1. **NInfer was the fastest engine at every depth**, by 3.9x to 5.1x against vLLM without
   speculative decoding, and by 1.7x to 7.7x against vLLM with it. It is also the only engine whose
   speed barely moves with depth: 262.5 tokens per second at the shallowest point and 230.9 at the
   deepest, a loss of 12% across the whole range.
2. **Against LM Studio, vLLM's speculative column leads at the two shallow depths and trails at the
   two deep ones** (151.1 against 91.0 and 123.8 against 85.3, then 59.5 against 134.3 and 30.1
   against 72.5). Without speculative decoding vLLM trails LM Studio at every depth. So "which of
   these two is faster" has no answer here that holds across the table, and any single number
   quoted from it should be read as one cell, not as a ranking.
3. **The fastest cell is not the shortest context.** LM Studio's best point is its third depth
   (134.3 against 91.0 at the near-zero-context point, 47.7% faster), where its draft acceptance
   reached 0.99 - which is the mechanism at work, not an anomaly: decode speed under speculative
   decoding follows the acceptance rate and the cost of drafting, and context length is only one of
   the things that moves those.

### Prefill

Server-reported, tokens per second (median of 3). vLLM's cells stay empty for the reason given in
the method chapter: it publishes no independent prefill counter and the client-side combination was
never implemented. **Empty means never captured, not slow.**

| Nominal depth | vLLM | NInfer | LM Studio |
|---|---|---|---|
| 8,192 | not captured | 9,102 | 3,288 |
| 32,768 | not captured | 8,563 | 3,333 |
| 98,304 | not captured | 6,236 | 2,804 |

NInfer ingested prompts 2.2x to 2.8x faster than LM Studio at these depths. The near-zero-context
point also exists in the run records (NInfer 1,409 and LM Studio 572, on a 76-token and a 117-token
prompt) and is not published as a prefill rate: at that prompt length the figure is dominated by
fixed per-request cost and says nothing about ingestion speed.

### The vLLM speculative-decoding gain, depth by depth

This is the measurement the second vLLM series exists for, and it is what changes the reading of
the whole matrix. Same engine, same weights, same KV cache dtype, same request surface; only the
speculative-decoding state differs. Both series were run at util 0.94 with 4-bit KV, which is the
capacity profile the deepest point needs.

| Nominal depth | no speculative decoding | 3 draft tokens | change |
|---|---|---|---|
| 0 | 67.3 | 151.1 | +124.6% |
| 8,192 | 63.3 | 123.8 | +95.6% |
| 32,768 | 58.3 | 59.5 | +2.1% |
| 98,304 | 45.5 | 30.1 | -33.9% |

**The gain collapses with depth and turns into a loss.** It is not a constant of the engine and not
a property of speculative decoding in general: it is the difference between two costs that grow at
different rates. Drafting adds work that scales with the context the draft must attend over, while
the tokens it saves per step are a fixed small number. By the third depth the two cancel; at the
fourth, drafting costs about a third of the throughput it was meant to buy.

Two pieces of evidence separate that reading from "the request was stuck":

- At the deepest point the three individual samples were 29.3, 30.0 and 30.6 tokens per second with
  drafting, against 45.4, 45.5 and 46.5 without it. The loss is stable to within about 4%, so it is
  a reproducible property of the configuration rather than one slow run, and the requests completed
  normally at both settings.
- NInfer, on the same model family and the same card, held 230.9 tokens per second at that depth
  with its own speculation enabled and a draft acceptance of 1.00 at every depth (from the engine's
  own request logs - the acceptance-rate section below says where that figure comes from and what it
  does not mean). The collapse is therefore a property of this engine's implementation on this
  architecture, not of the technique.

### Speculative-decoding state and acceptance rate

The state is part of the measurement, not a configuration footnote: two cells of the matrix above
are the same engine doing the same work with and without it.

| Nominal depth | vLLM, no speculative decoding | vLLM, 3 draft tokens | NInfer, 3 draft tokens | LM Studio, draft head inside the weights |
|---|---|---|---|---|
| 0 | not enabled | enabled | enabled | enabled |
| 8,192 | not enabled | enabled | enabled | enabled |
| 32,768 | not enabled | enabled | enabled | enabled |
| 98,304 | not enabled | enabled | enabled | enabled |

Acceptance rate at the same points. A cell for an engine that ran without speculative decoding is
not a measurement and is marked as such, never as zero.

| Nominal depth | vLLM, 3 draft tokens | NInfer, 3 draft tokens | LM Studio, draft head inside the weights |
|---|---|---|---|
| 0 | 0.734 (shallow points only) | 1.00 | 0.89 |
| 8,192 | 0.734 (shallow points only) | 1.00 | 0.93 |
| 32,768 | not recorded per depth | 1.00 | 0.99 |
| 98,304 | not recorded per depth | 1.00 | 0.91 |

The vLLM figure is a shallow-context reading carried in the run write-up as a mean accepted length
of 3.20 draft tokens and a draft acceptance of 73.4%; it is not split by depth there, so it is
shown against the two shallow points and nothing is invented for the deep ones. LM Studio's
acceptance ranged 0.89-0.99, and its best cell is the point where it was fastest.

**NInfer's 1.00 is not in the per-point records, and this column carries the chapter's reading, so
its provenance is stated rather than implied.** The harness's acceptance field is empty (`null`) for
that engine at every depth: it was read from the engine's response body and never exported into the
records. The figure published here comes from the engine's own per-request log lines, which for these
requests report the acceptance pair fully accepted - `mtp accepted 191/191` on every measured request
at every depth, three draft tokens per step and all three accepted. That is what the engine did under
this protocol.

It is not what that engine does in general, and the reason is in the request surface. The prompt asks
for one fixed sentence to be repeated 20 times, so the continuation is about as predictable as text
gets and a draft head has almost nothing to get wrong. **The engine's own documentation records an
acceptance rate of 61-62% for the same 3-draft scheme on ordinary text** (quoted from the material
behind this repository; not measured here). The 100% is therefore a product of this measurement
condition - a highly predictable output - and must not be read as that engine's general acceptance
level, nor as a statement that its draft scheme is better than another engine's. What the column
supports is the narrower claim that this engine's acceptance did not fall with depth *under this
prompt*, and even that is a statement about the prompt at least as much as about the engine.

**These numbers explain each engine's own speed and must not be ranked across engines.** The three
engines do not count the same thing: one reports a mean accepted length over draft tokens, one
reports accepted tokens against drafted tokens, one reports the pair its server prints. The draft
schemes also differ in kind - three fixed draft tokens against a draft head shipped inside the
weights. A higher acceptance rate is not "better" speculation; it is a different measurement.

### Depth actually reached

The prompt token count each engine reported. This is the axis every table above is read on, and it
is why the nominal column is not the whole story.

| Nominal depth | vLLM | NInfer | LM Studio |
|---|---|---|---|
| 0 | 117 | 76 | 117 |
| 8,192 | 4,570 | 4,531 | 4,571 |
| 32,768 | 17,922 | 17,882 | 17,922 |
| 98,304 | 53,547 | 53,507 | 53,547 |

All three engines received the same bytes; the 40-token difference at the deepest point is the
tokenizer, not the protocol, and it is far inside the run-to-run band. The requested 98,304 landed at
53,547 - the nominal label is off by 45%, which is why no figure in this document is quoted against
it.

### What a cell in these tables means

- The figure is the engine's own report, read by the rules given in the method chapter, not the
  client's wall-clock rate.
- Each cell is the median of three measured requests, after one warm-up request that was discarded.
  A cell is never a peak and never a single run.
- The depth a cell belongs to is the prompt token count the engine reported for the request, not the
  nominal figure in the left column.
- `not captured`, `not enabled` and `not recorded per depth` are three different things and each one
  is marked where it appears: a number that was never taken, a feature that was switched off, and a
  number that exists in the engine's own output but was not preserved per point. **None of them is
  zero**, and none of them may be read as a slow measurement.

### Sample size, spread, and the differences this method cannot resolve

- Every point is n = 3 after a discarded warm-up, and the published value is the median. The
  per-point records kept alongside this document carry the median and n; the three individual
  values were printed by the harness at run time and are carried in the run write-up for one pair of
  points only (the deepest-depth vLLM pair quoted above, spreads of about 4% and about 2%).
- The spread within a point is small. The dominant variation measured on this machine is **between
  runs**: identical configurations have differed by about 10% from one boot to the next, and pooled
  KV capacity by about 7% between runs.
- Consequences, stated as plainly as they can be: any difference below that band is unresolved by
  this method. vLLM's 67.3 against 63.3 at the two shallow depths and LM Studio's 91.0 against 85.3
  are both inside it, so neither may be read on its own as "the engine slows down with depth". The
  monotone trend each of those engines shows across all four depths, and the endpoint differences
  (32% for vLLM, 20% for LM Studio), are outside the band.
- Every engine-to-engine difference this chapter publishes is far outside it. The smallest -
  LM Studio against vLLM without speculative decoding at the two shallow depths - is 35%.

### What cannot be compared across the columns

1. **The speculative-decoding state.** Two of the four columns ran with it, over schemes that differ
   in kind, and vLLM appears once with it and once without. The matrix is usable for reading vLLM's
   two series against each other and for each engine's own trend with depth; cross-engine statements
   are limited to the ones listed above.
2. **The weights.** LM Studio served a Q4_K_M GGUF artifact while the other two served NVFP4. Part
   of any difference is weight format, not engine, and that part is not separable from these runs.
3. **The KV cache dtype.** `int4_per_token_head`, `fp8` and `q4_0` cost different numbers of bytes
   per token, so the three engines were not holding the same amount of context state per token; the
   arithmetic is in the next chapter. The deepest point is where this matters most, because pool
   pressure and per-token pool cost both scale with it.
4. **The acceptance rate.** Engine-internal, for the reasons given above.
5. **The client-side wall-clock column.** It is kept in the run records and is deliberately not
   published here as a comparison value: it includes network time, JSON parsing and time to first
   token. At the deepest point it diverges from the server figure by 6.5x on LM Studio (11.1
   against 72.5 tokens per second), which is a property of the client path, not of the engine.

### When this stops being three separate measurements and becomes a comparison

Five things have to match at every point, and any one of them that differs turns the rows back into
three separate measurements. The state of each one here is given with it.

1. the depth as the engine reported it - matches to within 40 tokens at the deepest point, a
   tokenizer difference rather than a protocol difference;
2. the output length - 256 tokens on all three engines at every point, and none of the points fell
   below the 32-token rejection threshold;
3. the sampling temperature - 0 at every point;
4. the speculative-decoding state - matches in on/off, differs in scheme; vLLM is published as a
   pair rather than as one column for this reason;
5. the number of requests in flight - 1 at every point.

### Still pending for this chapter

1. A per-depth draft-acceptance reading for the vLLM speculative series, and the preemption counters
   for those points. The deep-context collapse is reproducible, but the retained evidence does not
   separate draft overhead from pool pressure, because the preemption counter was read during the
   run and is not carried in the per-point records this repository keeps.
2. Two points that would tighten the switch matrix of chapter 6: the fp8-KV point taken at this
   chapter's own request ceiling, so that the dtype becomes the only variable in that row, and a
   combination point that puts graphs off together with speculation on.
3. A concurrency ladder, which would move this chapter from "one request at a time" to something
   that speaks to serving several requests at once.

## Model geometry and the KV budget

<!-- Sources for this chapter (shorthand used because none of these files is part of this
repository; the numbers were re-derived here, not copied):
  cn-geometry  = the internal model-geometry table and per-token formula (27B and MoE configs)
  cn-kv-dtype  = the internal write-up of the eight KV cache dtype start-up runs (2026-09-04)
  cn-moe       = the internal write-up of the MoE measurement and its start-up failures (2026-09-05)
  cn-ledger    = the internal engine-comparison write-up, section on memory-ledger differences
  scripts/select_and_run.py in this repository carries the launcher constants quoted below.
Every capacity, residual and pool figure in this chapter was recomputed from the constants shown in
it; the recomputation and every point where it disagrees with the source material are recorded in
the execution notes for this document. One published figure is not reproducible and is explicitly
not used here: a MoE pool of ~5.3 GiB paired with a ~192K single-request ceiling (that pairing
cannot hold - see the ledger examples below for the figures that do). -->

### The per-token KV formula, and the two geometries measured

Neither model here is a plain decoder stack. In both, only one layer in four carries an attention KV
cache; the rest are linear-attention layers that carry a recurrent state instead. That ratio, not the
parameter count, is what decides the size of a KV cache on this architecture:

```
KV bytes per token = full-attention layers x KV heads x head dimension x 2 x bytes per element
```

The factor 2 is one array for keys and one for values.

| | Dense 27B-class | MoE 35B-A3B-class |
|---|---|---|
| Total layers | 64 | 40 |
| Full-attention layers | 16 | 10 |
| Linear-attention (recurrent) layers | 48 | 30 |
| KV heads | 4 | 2 |
| Head dimension | 256 | 256 |
| KV bytes per token, 1 byte per element | 32,768 | 10,240 |
| KV bytes per token, 2 bytes per element | 65,536 | 20,480 |
| Relative to the dense geometry | 100% | 31.25% |

The row worth keeping is the third from the bottom: the MoE holds 31.25% of the dense model's KV per
token, so it holds 3.2x as many tokens in a pool of the same size, and that follows from the layer
count and the number of KV heads and from nothing else. The number of experts, the number of
activated parameters and the weight file size never enter this formula - worth stating because the
intuition "the bigger model needs more cache" points the wrong way here: the MoE carries the larger
weights (23.27 GiB against 21.34 GiB) and the smaller cache.

Formula only, three worked examples:

- a 100,000-token context on the dense geometry at 1 byte per element: 100,000 x 32,768 B
  = 3.05 GiB;
- the same context on the MoE geometry: 100,000 x 10,240 B = 0.95 GiB;
- a 200,000-token context on the dense geometry at 4-bit packed storage (0.5 byte per element):
  attention part 200,000 x 16,384 B = 3.05 GiB, plus one fp32 scale per (token, head, direction)
  200,000 x 512 B = 0.10 GiB.

### The formula is a lower bound: what the pool actually costs

The formula gives the attention part. The pool the engine reports costs more, and that difference is
the subject of this chapter and of the next one. Each row below is one start-up run of the internal
KV cache dtype series, all at `max-model-len 65536`, `max-num-seqs 32` and
`gpu-memory-utilization 0.92`; the measured pool cost per token is the reported pool size divided by
the reported token capacity, and the last column is that value minus the formula.

| KV cache dtype | Element width | Attention part (formula) | Measured pool cost (B/token) | Residual |
|---|---|---|---|---|
| `auto` (FP8 checkpoint) | 1 byte | 32,768 | 37,948 | +5,180 |
| `fp8` | 1 byte | 32,768 | 37,703 | +4,935 |
| `fp8_per_token_head` | 1 byte + per-token-head scale | 33,280 | 38,979 | +5,699 |
| `int8_per_token_head` | 1 byte + per-token-head scale | 33,280 | 38,873 | +5,593 |
| `int4_per_token_head` | 0.5 byte + per-token-head scale | 16,896 | 22,020 | +5,124 |
| `turboquant_k8v4` | 0.75 byte (8-bit keys, 4-bit values) | 24,576 | 30,240 | +5,664 |

The residual is between 4,935 and 5,699 bytes per token across the six rows of that table - a spread of
13% against the +-12% band the launcher itself carries - and it does not move with the dtype: the
step from 16-bit-class to 8-bit-class storage changes it by nothing, and the step to 4-bit changes
it by 8%. Three consequences follow, and the rest of this chapter rests on them.

**1. A dtype switch acts on part of the pool, so the capacity it buys is smaller than the formula
claims.** `int4_per_token_head` halves the attention part (32,768 to 16,896 bytes per token) but
cuts the measured pool cost by 41.6% (37,703 to 22,020), and the capacity gain is 2.04x (86,016 to
175,542 tokens). A pool budgeted from the formula alone would claim the full halving.

**2. On this checkpoint the engine's default is already 8-bit-class, so the usual 16-bit-to-8-bit
saving is not there to measure.** `auto` and `fp8` land 0.65% apart per token (37,948 against
37,703) and within 7% in pool size, which is inside run-to-run variation. Reading the `auto` row as
bfloat16 - as the source material does - cannot hold: the bfloat16 formula would cap that 3.04 GiB
pool at about 49,800 tokens, while the engine reported 86,016, and no amount of page alignment
explains a factor of 1.7. The checkpoint behind these runs is itself FP8-quantised (the same fact
that makes `fp8_e5m2` unusable on it, in the next chapter), so `auto` resolves to the 8-bit-class
path and the roughly 8% capacity difference between those two rows is pool-size variation rather
than quantisation.

**3. The formula's answer and the engine's report disagree by 15%.** The internal geometry table
publishes about 106.8K tokens as the dense model's single-request ceiling at 1 byte per element in a
3.26 GiB pool. That is the formula's number (3.26 GiB / 32,768 B). The same pool size, measured,
held 92,842 tokens, and the 15% between them is the residual in the table above. This document
publishes the measured capacity and treats the formula as a lower bound on pool cost per token, not
as a capacity.

### The memory ledger the launcher uses

The launcher in this repository estimates start-up memory from a ledger calibrated on that machine's
own start-up logs:

```
budget = min(util x total VRAM, free VRAM at start-up)
pool   = budget - weights - peak activations - non-torch overhead
                 - CUDA-graph reserve - MTP head (if enabled)
tokens = pool / bytes per token
```

The form of this ledger is why two servers can disagree about the same card while both are telling
the truth: the budget is a ceiling taken on the whole device rather than on the KV pool, and four
reserve terms are charged before the pool gets anything. The engine refuses to start when free
memory is below the requested ceiling, and it reports "KV cache needed > available" with an
estimated limit when the pool cannot hold the requested context. Both are fast, explicit failures
rather than faults, and both have been hit on this machine.

| Term | Value (dense geometry) | Basis |
|---|---|---|
| Total VRAM | 31.82 GiB | as the driver reports it |
| Weights | 21.34 GiB | six start-up runs |
| Weights, MoE | 23.27 GiB | one start-up log; 1.93 GiB more, so its util has to come down |
| Peak activations | 1.92 GiB | follows the batch budget; at the 32-sequence batch of the dtype series the same arithmetic implies about 2.25 GiB |
| Non-torch overhead | 1.9 GiB | CUDA context, workspaces |
| CUDA-graph reserve | 0.11-0.18 GiB for the per-token quantised family, 0.44-0.51 GiB for the 8-bit family | varies with the attention backend |
| MTP head | 0.4 GiB | about one extra layer plus its drafting activations |
| Error band on the pool | +-12% | non-torch overhead and the CUDA-graph estimate |

Worked examples, all recomputed here from the table above:

- **Dense, util 0.92, `int4_per_token_head`:** budget 29.27 GiB, pool 3.93 GiB (band 3.46-4.41),
  capacity about 169,000 to 215,000 tokens. A 100,000-token request fits with room for concurrency.
- **Dense, util 0.92, `fp8`:** pool 3.60 GiB (3.17-4.04), capacity about 90,000 to 115,000 tokens.
  One 100,000-token request sits at the edge of the band, which is why the protocol's second vLLM
  profile caps the context at 73,728: on this card, 8-bit-class KV in the dense geometry is a
  64K-class configuration.
- **MoE, util 0.92, `fp8`:** pool 1.67 GiB (1.47-1.88), capacity about 154,000 to 197,000 tokens at
  10,240 bytes per token. The one start-up limit recorded for this model, 192,832 tokens, sits at the
  top of that band, and it implies a pool of about 1.97 GiB at the roughly 10,950 bytes per token the
  same log implies. **This - not a 5 GiB pool - is the arithmetic behind the MoE's ~192K ceiling.**
- **Dense, util 0.94, `int4_per_token_head`:** pool 4.57 GiB (4.02-5.12). A 200,000-token request
  needs 200,000 x 22,020 B = 4.10 GiB, so the ceiling clears it on paper; every measured attempt at
  that size failed by about half a gigabyte (a 3.60 GiB pool against 4.10 GiB needed), and the gap
  between the two accounts is the activation term, which grows with the batch size - the measured
  3.60 GiB is the 32-sequence pool, while the estimate above uses the 12-sequence constant. Whether
  200K fits at util 0.94 therefore depends on the concurrency wanted at the same time, and this
  repository does not claim that it fits.

Two 150K figures appear elsewhere in the material behind this repository, and neither is a measured
maximum. At `int4_per_token_head` a 3.60 GiB pool holds about 175.5K tokens at the measured pool cost
of 22,020 B per token, so the 150K single-stream working point used for long-context work is a
compromise against a 200K target, not the ceiling. The MoE's remembered configuration carries a
150,000-token context limit against the computed ceiling of about 192.8K derived above. Quoting
either 150K as the capacity of the corresponding configuration would be wrong in the same direction.

### The utilisation convention, stated once

Four values of `gpu-memory-utilization` appear in the material behind this repository. They are not
four opinions about one setting; they are one setting at four different jobs, and a capacity quoted
without its util is not a measurement.

| Value | What it is |
|---|---|
| 0.92 | the steady-state default: the value the geometry table above is stated at, and the launcher's own default |
| 0.93 | the dense model's remembered setting, the value its parameter memory carries |
| 0.94 | the ceiling used in the stress tests and in the comparison run of the previous chapter - viable on this machine, not a daily setting |
| 0.90 and below | the recommendation for the MoE, whose weights are 23.27 GiB against the dense model's 21.34 GiB; the 1.93 GiB difference comes straight off the pool at the same util |

### The correction: the recurrent state is not governed by the KV cache dtype setting

The scope of a KV cache dtype switch is narrower than its name suggests on this architecture. Of the
dense model's 64 layers, 48 carry a recurrent state (a convolution state and a state-space state)
rather than a KV cache; the MoE has 30 of 40. That state is allocated in a fixed 16-bit/32-bit
format taken from the model configuration rather than from the KV cache dtype flag - a code-reading
claim carried by the source material, not re-measured here.

What the measurements above add to it is what a dtype-independent component inside the pool looks
like from the outside:

- the residual is flat across six dtype settings (4.9-5.7 kB per token), within the error band;
- the step that should save the most - 16-bit-class to 8-bit-class storage - shows no measurable
  saving at all on this checkpoint;
- the step to 4-bit returns 42% instead of the 50% a pure attention budget would claim, and the
  capacity gain is 2.04x rather than the 3.9x that comparing 32-bit with 4-bit would suggest.

One published figure does not reconcile, and is therefore not reproduced here. The source material
attributes part of the pool's non-attention content to 32 sequences' worth of recurrent state at
about 75 MiB per sequence, that is about 2.4 GiB. A 2.4 GiB charge inside the 3.26 GiB pool the
engine reported would leave room for about 28,000 attention tokens, while the engine reported
92,842; the residual measured above is about 0.46 GiB at that capacity. The figures that do
reconcile are the measured capacities and a constant residual of roughly 5 kB per token, and those
are the ones this document uses. Where exactly the recurrent state is charged is not settled by the
evidence retained here; what is settled is that the KV cache dtype switch cannot move it, and that a
budget built from theoretical bytes per element will understate the pool.

Two limits go with this. On a decoder-only model with no recurrent layers, the same switch should
return the theoretical factor and the 8-bit step should save what it promises; nothing here was
measured on such a model, so the finding above must not be carried over to one. And the per-token
constants in this chapter are for these two checkpoints on this card, at the batch sizes stated;
they are not properties of the dtypes themselves.

## KV cache dtype: 8 candidates measured

<!-- Sources: cn-kv-dtype = the internal write-up of the KV cache dtype start-up series
(2026-09-04). One artefact of that series exists in this work copy as a result summary with one
RESULT line per attempt (7 attempts); it is excluded from the published tree by .gitignore, so the
numbers below are quoted with their source rather than shipped as raw evidence. cn-geometry and the
attention formula are the ones established in the previous chapter. Every pool cost, capacity ratio and residual below
was recomputed from the reported pool size and the reported token capacity rather than copied from
the source material; where the recomputation disagrees with it, this chapter follows the
recomputation and the disagreement is recorded in the execution notes for this document. The polling
driver that produced the start-up logs was not retained: these points exist as evidence and cannot
be re-run as they stand. -->

### Method

Eight KV cache dtype candidates appear in the engine's menu on this build. Seven of them were started
in one series and one was rejected in an earlier round, and this chapter keeps those two groups
apart, because only the seven belong to the series the numbers come from.

All seven ran under identical start-up settings: `max-model-len 65536`, `max-num-seqs 32`,
`gpu-memory-utilization 0.92`, on the dense geometry and the checkpoint of the previous chapters.
Each candidate was polled for up to 300 seconds and counted as started when the engine printed its
own start-up-complete line; every candidate that reached that point was then sent one real chat
request at temperature 0 expecting a numeric answer, and the process was killed to release device
memory before the next candidate began. Six of the seven answered. The retained summary carries one
line per attempt with the outcome and the answer (it is not part of the published tree; see the
source note above), and the attempt stamps show the whole series inside
26 minutes, with the one failing candidate returning 43 seconds after its attempt began - well inside
the 300-second window, which is what a fast explicit refusal looks like rather than a hang.

Two properties of that method have to travel with the table:

- **The settings are fixed; the pool is not.** With the same settings the engine ends up with a
  different pool on each start: 3.04 GiB on the first candidate, 3.74 GiB on the last. The ledger's
  reserve terms vary between start-ups, so the comparable quantity is the pool cost per token -
  reported pool in bytes divided by reported token capacity - and not the pool size in GiB. Every
  ordering claim below is made on that per-token figure.
- **The series' own request ceiling was 65,536 tokens.** Every pool below is larger than that, so the
  "longest single context" column is the ceiling the pool supports for one request that is given the
  whole pool - computed from the reported capacity - and not a request that was actually issued at
  that length. No request longer than 65,536 tokens was sent to any of the eight candidates.

### The eight candidates

| KV cache dtype | Element width, with its scale | Measured pool cost (B/token) | Pool reported (GiB) | Longest single context (tokens) | Start-up outcome |
|---|---|---|---|---|---|
| `auto` (engine default) | 8-bit class, per-tensor | 37,948 | 3.04 | 86,016 | started; answered the test request |
| `fp8` | 1 byte, per-tensor | 37,703 | 3.26 | 92,842 | started; answered |
| `fp8_per_token_head` | 1 byte + per-(token, head) scale | 38,979 | 3.69 | 101,647 | started; answered |
| `int8_per_token_head` | 1 byte + per-(token, head) scale | 38,873 | 3.68 | 101,647 | started; answered |
| `int4_per_token_head` | 0.5 byte + per-(token, head) scale | 22,020 | 3.60 | **175,542** | started; answered |
| `turboquant_k8v4` | 0.75 byte (8-bit keys, 4-bit values) | 30,240 | 3.74 | 132,796 | started; answered |
| `fp8_e5m2` | 1 byte, per-tensor | not reachable | not reachable | not reachable | rejected at start-up |
| `nvfp4` | 4-bit | not reachable | not reachable | not reachable | rejected at start-up, in an earlier round |

Read out column by column:

- **Measured pool cost** is the reported pool in bytes divided by the reported token capacity, so it
  is the engine's own accounting read back and not a formula. Its distance from the attention-only
  formula - the residual of the previous chapter - is 4,935 to 5,699 bytes per token across the six
  rows above, and it does not step with the dtype.
- **Pool reported** is what the engine allocated at those settings. Its 3.04 to 3.74 GiB spread at
  identical settings is the start-up variation the method paragraph warns about.
- **Longest single context** is the pool's token capacity, i.e. the ceiling for one request that is
  given the whole pool. Every value here exceeds the series' 65,536-token request ceiling, so the
  column is a computed ceiling and not an exercised request length.
- **Start-up outcome** is the engine's own start line plus a real request. "Rejected" means the
  process never reached a serving state; the two rejections have different causes and are separate
  sections below.
- `not reachable` is used for candidates that never served: there is no pool, no capacity and no cost
  to report. It does **not** mean zero, and none of these cells is a slow measurement.

### The counter-intuitive result: capacity does not follow bit width

The intuition is that the element width decides how much context fits. Measured, the ordering is:

1. 0.5 byte with fine scales: **175,542 tokens**.
2. 0.75 byte, 8-bit keys with 4-bit values: 132,796 tokens.
3. 1 byte with fine scales: **101,647 tokens** - and both members of that layout group land on the
   same number, to the token.
4. 1 byte, per-tensor: 92,842 tokens.
5. The engine's own default: **86,016 tokens**, the smallest of the six.

Three findings follow, and they are what this chapter exists to report.

**Finding 1: halving the element width does not halve the pool.** `int4_per_token_head` cuts the
measured pool cost by 41.6% (37,703 to 22,020 bytes per token), not by 50%, and the capacity it buys
is 2.04x the default (86,016 to 175,542) - not the roughly 3.9x that comparing a 16-bit-class formula
(65,536 B/token) with a 4-bit one (16,896 B/token) would claim. The 41.6% and the 2.04x are the same
fact seen from two sides.

**Finding 2: the 8-bit family is one capacity class.** Four different 8-bit-class layouts, including
the default, spread across 86,016 to 101,647 tokens - 18% above the smallest value - while the single
4-bit row sits 73% above the top of that class. On this checkpoint, moving between 8-bit candidates
is not a capacity decision; the 4-bit row is the only step that changes class.

**Finding 3: the two candidates that share a layout share a capacity.** `fp8_per_token_head` and
`int8_per_token_head` differ in the element type and agree on capacity exactly, on pools that differ
by 1% (3.69 against 3.68 GiB). Byte-for-byte accounting explains that, and it is a useful check: the
capacity column is reading the layout, not noise. `turboquant_k8v4` also behaves as its 0.75-byte
width predicts, landing between the 8-bit class and the 4-bit row on both cost and capacity - the one
row where intuition and measurement agree.

### The root-cause chain

Three links, in the order the evidence supports them.

1. **The pool is not mostly attention state.** Of the checkpoint's 64 layers, 16 are full-attention
   and 48 are linear-attention layers that carry a recurrent state - a convolution state and a
   state-space state - instead of a key/value cache. That state lives in the same pool and is
   allocated in the format the model configuration asks for, not in the format the KV cache dtype
   flag asks for, so the switch never touches it. This is a code-reading claim carried by the source
   material; what this document measured is its shadow, in the next link.
2. **A dtype-independent component dominates the differences between candidates.** The residual
   between the attention formula and the measured pool cost is 4,935 to 5,699 bytes per token across
   the six rows of the table. Its width - about 13% of the largest value, about 15% of the smallest -
   is of the same order as the launcher's own +-12% band on the pool, and it does not move with the
   dtype. Its share of the pool is what decides how much a switch can return: at plain 8-bit it is
   13% of the cost per token, at 4-bit it is 23%. That is why the 4-bit row returns 42% instead of
   the 50% a pure attention budget would claim, and why the 0.75-byte row lands where it does.
3. **On this checkpoint the default is already an 8-bit class, so the saving people expect from
   8-bit KV is not available to measure.** The default reports 37,948 bytes per token against
   `fp8`'s 37,703 - a difference of 0.65% - and its pool is 7% smaller between two start-ups. Read as
   bfloat16, the formula would cap that 3.04 GiB pool at about 49,807 tokens, while the engine
   reported 86,016: a factor of 1.73 that no page-alignment rule explains. The checkpoint is an
   NVFP4 release whose attention layers are FP8-quantised, so the engine's default resolves to the
   8-bit-class path and the small capacity difference between those two rows is pool-size variation
   between start-ups rather than quantisation. **The statement that "8-bit KV saves almost nothing
   against 16-bit on this model" cannot be measured on this checkpoint at all, because there is no
   16-bit-class row to compare against.** What is measurable, and what this chapter reports, is that
   the 8-bit family is one capacity class and the 4-bit row is a different one.

The source material also attributes part of the saving to page alignment between the attention pages
and the recurrent-state pages. That arithmetic does not reproduce - the previous chapter shows why -
and it is not used here. What the retained evidence does support is the constant, dtype-blind
residual above.

### The `_per_token_head` family

The suffix is not decoration: it is the quantisation granularity, and it is why this family's rows do
not behave like plain `fp8`.

- Plain `fp8`: one scale for the whole key/value tensor.
- `*_per_token_head`: one independent fp32 scale per (token, head) pair. Measured against the
  per-tensor layout, that is the same element width plus 512 bytes per token of scale storage in the
  attention formula (32,768 to 33,280 bytes per token, the difference being the scale array: 16
  layers x 4 heads x 2 directions x 4 bytes = 512 bytes per token, computed). Against one scale per
  tensor that is finer granularity by the number of (token, head) pairs - about 4.3 orders of
  magnitude at a 4,570-token prompt and about 5.3 at a 53,500-token prompt (computed) - for 1.6% more
  bytes. The scale array is why `int8_per_token_head` cannot beat `fp8` on
  capacity even though its element type is no wider.
- So `int8_per_token_head` sells precision at the capacity of `fp8_per_token_head`, and
  `int4_per_token_head` halves the element width while keeping the fine scale - which is why it is
  the only row that changes capacity class.
- The three layouts are defined in the engine's own attention backend, at the path
  `vllm/v1/attention/backends/triton_attn.py` on this build. This document quotes the layout, not the
  implementation.
- `turboquant_k8v4` is a different backend again rather than a member of that family: its log line
  names a separate attention backend, it forces the attention kernel back one version, and it packs
  8-bit keys with 4-bit values into one slot - a layout that assumes keys are the sensitive half and
  values the cheap half. It entered the engine only at this version and, per the material behind this
  repository, was developed and tested mainly on another vendor's hardware, with no authoritative
  accuracy baseline. Its position in the quality ordering below is therefore a theoretical inference
  from its bit widths and nothing more.

Quality ordering - **theoretical inference only, nothing here measured quality**:
`fp8_per_token_head` about equal to `int8_per_token_head`, both above plain `fp8`, at or above
`turboquant_k8v4`, above `int4_per_token_head`.

### The two candidates that never started

Both are fast, explicit refusals with different causes, which is why they are reported separately
rather than lumped together as "unsupported dtypes".

- **`fp8_e5m2`** was rejected inside this series, 43 seconds after its attempt began. The engine's
  message names a supported combination, not a missing kernel: this key/value element type is not
  accepted with an FP8 checkpoint. The backend's own support list does contain the element type, so
  the refusal comes from the checkpoint - whose attention layers are already FP8-quantised - and not
  from the kernel set. On a non-quantised checkpoint this candidate would be judged on its merits.
- **`nvfp4`** was rejected in an earlier round and is **not** one of the seven attempts in the series
  above. Its rejection is a capability gate rather than a checkpoint check: the engine's 4-bit
  key/value path is enabled only for the data-centre compute family, and every attention backend on
  the consumer part this machine runs refused it. Its row appears in the table only because the
  engine's menu offers it; **no number in this chapter comes from it**, and it must never be counted
  as one of the series' seven experiments.

### The recommendation, and the half of it that is missing

The capacity recommendation is `int4_per_token_head`: 175,542 tokens, 2.04x the default, on a pool
that is the smallest of the six despite being the most compressed candidate. That is exactly what
classifies it as a **capacity** option and not a speed or quality option. Two consequences:

- The speed side has a price, and the material behind this repository puts it at depth: unpacking
  4-bit state back to a wider format happens at every decode step and costs more the more history the
  step must attend to, so the cost is paid in long-context decode, not at start-up. That is
  consistent with the depth curves measured at this dtype in the next two chapters.
- **Its output quality was not measured.** The verification the series performed was one request per
  candidate at temperature 0 with a numeric answer expected, and all six answered it. That is a smoke
  test of the code path, not a quality measurement. Nothing in this repository measures long-context
  recall or coherence at 4 bits, and the fill corpus used across this document is a repeated text
  block rather than a document with facts to retrieve, so a recall test would need material this work
  does not have. What the source material offers instead is the fine-grained scale as a design
  mitigation and a suggested method - one document, questions answered at each dtype, comparing the
  middle of a long document - which has not been run.

The candidate that would have anchored the other end of that comparison, a 16-bit-class row, cannot
run on this checkpoint. So the chapter has a measured capacity ladder and no measured quality ladder
at all.

### What this chapter does not claim

1. It does not claim these capacities are properties of the dtypes. They are properties of these
   candidates on this checkpoint, at `max-model-len 65536`, `max-num-seqs 32` and util 0.92, on this
   card. The residual that dominates the differences is a property of this hybrid architecture: on a
   decoder-only model with no recurrent layers the 8-bit step should recover the saving it promises,
   and nothing here was measured on such a model.
2. It does not claim the capacities were exercised. The pools are larger than the series' own
   request ceiling, and no single request approached a pool's full capacity.
3. It does not claim a ranking of quality, speed or stability among the six that started. One of the
   eight changes capacity class by a margin that matters, and each candidate answered exactly one
   request.
4. It does not claim the series is reproducible: its driver was not retained with its result summary.

## Concurrency, context depth and CUDA graphs

<!-- Sources: cn-concurrency = the internal write-up of the KV pool concurrency stress series and the
CUDA graph pair (2026-09-05), whose harness family is the one in this repository
(scripts/kv_bench.py); cn-moe = the internal write-up of the second geometry, which contributes the
occupancy points and is the subject of chapter 7. The pool cost per token and the ledger constants
are the measured ones of the two preceding chapters. Where a figure is a rate multiplied by a
concurrency it is marked computed; where a read-out was not retained the cell says so. These points
were taken in a different series from the comparison matrix, with different depths, output lengths
and read-out rules, and they are not part of that matrix. -->

### Method

The measurements in this chapter come from a different series from the comparison matrix, and mixing
them would be a mistake, so the difference is stated first. The comparison matrix is one request at a
time, 256 output tokens, a fixed fill corpus, three repetitions, and the engine's own counter as the
read-out. The series here is a stress series: several requests in flight, prompts of about 4,000 and
about 48,000 real prompt tokens, 400 to 500 output tokens, and per-request rates read from the
serving side while pool occupancy and the preemption counter were sampled during the run. Its purpose
was to find where the pool breaks, not to rank engines.

Configuration for the series: the dense geometry of the previous chapters, `int4_per_token_head`,
`max-model-len 100000`, `max-num-seqs 12`, `gpu-memory-utilization 0.94` - the ceiling configuration
of the utilisation table, chosen because deep-context work is what it exists for. Two CUDA graph
states were measured: off, through the engine's eager flag, and on, which is the default.

The one methodological point that decides how the table is read: **the five-request points need
distinct prompt content.** Five requests carrying the same text can be collapsed by the engine's
prefix machinery into one cached copy, and then the pool holds one request's worth of state instead
of five. The series ran both variants on purpose, and the difference is large enough to be the first
result below.

### The concurrency ladder

Rates are per request, server side, as the series read them. "Aggregate" is the per-request rate
multiplied by the number of requests in flight - **computed**, and a floor rather than a guarantee,
because it assumes the requests received equal shares of the device.

| Requests in flight | Prompt depth (real tokens) | Output | CUDA graphs | Pool peak | Preemptions | Per-request rate (tok/s) | Aggregate (tok/s, computed) |
|---|---|---|---|---|---|---|---|
| 1 | about 4,000 | 500 | off | not retained | not retained | 13.9 | 13.9 |
| 1 | about 4,000 | 500 | on | not retained | not retained | **48.2** | 48.2 |
| 3 | about 4,000 | 500 | on | not retained | not retained | 43.9 (range 42.4-45.6) | about 132 |
| 3 | about 48,000 | 400 | on | not retained | not retained | 12.2 | about 37 |
| 5 | about 48,000 | 400 | off | not retained | 0 | 6.3 | about 32 |
| 5 | about 48,000 | 400 | on | 50.8% | 0 | 6.3 | about 32 |
| 5 | about 48,000 | 400 | on | 24.6% | 0 | 13.7 | about 69 |
| 5 | about 48,000, **identical content** | 400 | on | 22.3% | 0 | 11.0 | about 55 |

Four statements are supported by this table.

1. **Depth costs far more than concurrency does.** Raising the prompt from about 4,000 to about
   48,000 tokens at three requests in flight costs 3.6x of per-request rate (43.9 to 12.2,
   computed). Raising concurrency from three to five requests at about 48,000 tokens costs no
   measurable eviction at all: zero preemptions at both points, and a worst-round pool peak of 50.8%
   against a 12-sequence slot budget, i.e. about half the pool still free.
2. **The pool is not what limits these points.** Every five-request point completed with zero
   preemptions. The rate loss is the attention work five deep requests put on one device, not an
   eviction, and it is the reason the earlier demonstration in this chapter is about configuration
   rather than about load.
3. **Identical content nearly halves the pool cost and does not help the rate.** Five identical
   48,000-token prompts held a 22.3% pool peak against 50.8% for distinct content - the engine
   merging them into one cached copy, which is precisely what the comparison chapter's per-request
   salt exists to prevent - and the per-request rate was 11.0, between the two distinct-content
   rounds. The pool column moves with the content while the rate column barely does.
4. **The same configuration, run twice, gave 6.3 and 13.7 tokens per second.** Those two cells share
   depth, concurrency, pool settings and read-out, and they differ by 2.2x - far outside the ~10%
   run-to-run band this document quotes for single-stream work, and outside anything the counters
   explain (50.8% against 24.6% pool peak, both with zero preemptions). **Consequence, stated
   plainly: the individual cells of this ladder are indicative, the counters that came with them are
   the durable evidence, and no figure in this chapter should be quoted to a decimal place.** Where
   the ladder and the counters appear to disagree, the counters win. The CUDA graph pair later in
   this chapter quotes the same two rounds from the other side, and its note says which round is
   which and why the figures cannot be compared with each other.

### The mechanism behind a fraction of a token per second

The extreme case on this machine is documented, was reproduced by its own counters, and is not a slow
model: it is the scheduler throwing work away.

Timeline of that event, as the material behind this repository records it: the configuration in use
at the time was the 200,000-token request ceiling with 12 sequence slots and 4-bit KV; three requests
were in flight, with prompts of about 103,000 and about 50,000 tokens. Their combined key/value
demand at the measured pool cost of 22,020 bytes per token is about 4.2 GiB (computed); the pool
reported for that configuration was about 3.0 GiB. The pool filled. The counters taken during the
window read 97% pool occupancy, 84 preemptions, and a waiting queue whose reason was capacity. What
follows from those counters is mechanical:

1. When the pool cannot hold every running request's state, the scheduler preempts: it takes the
   longest request's state out of the pool to make room for the others.
2. A preempted request does not resume where it stopped. Its prompt is processed again when it is
   admitted back, so the deepest request pays its prefill repeatedly.
3. With several deep requests in flight the pool stays oversubscribed, so evictions repeat, and the
   measured external symptom is a per-request rate of about half a token per second while the
   requests are technically still progressing.

**The lesson is about configuration, not about the engine's speed.** The same engine on the same card
with a 100,000-token ceiling and the correspondingly deeper pool held five 48,000-token requests with
zero preemptions. The collapse belonged to a ceiling whose paper capacity the working set did not
fit; lowering the ceiling removed it, and lowering concurrency was not needed. That is also why the
comparison chapter discards any point whose preemption counter is non-zero: a non-zero counter means
the point measured the scheduler's eviction policy as much as the engine.

One discrepancy between the two write-ups behind this section is recorded rather than smoothed over.
For a nominally similar 200,000-token-class configuration they report pools of about 3.0 GiB and about
3.6 GiB. The pool is a remainder after the ledger's reserve terms and the activation term moves with
the batch budget, so a difference of that size between two different configurations is expected in
direction, but its exact value is not pinned down by the retained evidence.

### The safe pool-occupancy ceiling

The working rule carried by the material behind this repository is that the pool's peak occupancy
should stay under about 85% for a chosen depth-x-concurrency product, and that above roughly 90% the
scheduler starts preempting and the rate collapses. The points that support it:

| Geometry | Depth x concurrency | Pool peak | Preemptions | Per-request rate (tok/s) |
|---|---|---|---|---|
| MoE | about 0 x 4 | 9.7% | 0 | 120.9 |
| MoE | 80K x 4 | 80.6% | 0 | 23.3 |
| MoE | 100K x 5 | 97.6% | 1 | 13.4 |
| dense | 48K x 5 | 50.8% | 0 | 6.3 |
| dense | 48K x 5 | 24.6% | 0 | 13.7 |

The conditions that go with the rule, because without them it reads as a law of nature:

- It rests on **three points on one geometry** (9.7%, 80.6%, 97.6%) plus two dense points, all on one
  card in one week. Two points under the line and one above it do not make a threshold; they make a
  usable heuristic.
- It is an **occupancy** rule, not a depth rule and not a concurrency rule. The quantity to keep
  under the line is the state the running set needs. The 80K x 4 point shows why: 80.6% occupancy,
  zero preemptions, and still a fivefold drop in per-request rate against the shallow point. Staying
  under the ceiling does not make deep work fast; it only keeps it from being evicted.
- It is stated at the **peak**, not at the average, because the preemption decision is made at the
  peak.
- It depends on the KV **dtype**, because the same token count occupies different bytes under
  different dtypes (previous chapter). A depth-x-concurrency product that fits at 4 bits may not fit
  at 8.
- The comparison chapter's rejection rule belongs with it: a point whose preemption counter is
  non-zero is not a clean measurement of the engine, whatever its average occupancy says.

### The CUDA graph pair

CUDA graphs are the engine's replayable-capture path for decode steps: a forward pass is captured
once and replayed for later steps of the same shape class, which removes per-step launch overhead.
The engine enables them by default; its eager flag disables them and saves the capture's memory.

| Configuration | Graphs off | Graphs on | Change |
|---|---|---|---|
| 1 request, about 4,000-token prompt, 500 output | 13.9 tok/s | 48.2 tok/s | +247% |
| 5 requests, about 48,000-token prompt, 400 output | 6.3 tok/s | 8.8 tok/s | +40% |

One note belongs under the second row, because the same configuration appears three times in this
chapter with three different numbers. The 6.3 to 8.8 pair is the graphs-off/graphs-on round of the
stress series. The ladder earlier in this chapter carries two further graphs-on rounds of the same
point - five requests at about 48,000-token prompts - reading **6.3 at a 50.8% pool peak** and
**13.7 at a 24.6% pool peak**. Those two records and this pair come from different rounds with
different pool occupancy states and **are not comparable with one another**; the 8.8 round is not
privileged over the other two, and this document does not choose between them. What the pair is used
for below is direction and order of magnitude only - graphs on lost no rate at either depth measured,
and the capture's cost stayed small against the pool - and no figure in this row may be quoted to a
decimal place (statement 4 of the ladder above states the same limit).

What the capture costs in device memory, from the start-up logs behind the previous chapters: 0.11 to
0.18 GiB for the per-(token, head) quantised family, 0.44 to 0.51 GiB for the 8-bit family. Against
a pool of 3 to 4 GiB, that is between 3% and 13% of the pool, and it buys 40% to 247% of the decode
rate at the two measured points. Three consequences:

1. **Turning graphs off to buy pool is a losing trade at both depths measured:** the pool grows by at
   most 13% while the rate can fall by 71% at short context. It is a setting for a device that cannot
   start otherwise, not a tuning option.
2. The gain shrinks with depth - +247% at short context against +40% with five deep requests - so a
   deep-context configuration loses less by disabling graphs, and it still loses.
3. **The pair exists on this engine only.** The closed-source comparison application exposes no
   user-facing switch for it, so there is no cross-engine pair to publish. Its log does carry a
   graph-reuse counter, and the log-derived statistics show it in the thousands of lines, but a reuse
   count is not an on/off measurement and must not be compared with the pair above. That application
   also performs its own kernel fusion and persistent-kernel work, so "graphs on or off" is not a
   feature one engine has and another lacks: it is a switch one engine exposes and the other does
   not.

### One number for short-context single-stream decode on the dense model

Two figures for short-context single-stream decode on this model were carried by the material behind
this repository and read like a contradiction. They are one measurement regime with a context-depth
difference, and this document publishes them as one figure.

| Reading | Configuration | Rate |
|---|---|---|
| near-zero context, over a 16-request histogram | single stream, graphs on | about 50 tok/s (mean 26.8 ms per output token) |
| about 4,000-token prompt, 500 output | single stream, graphs on, 4-bit KV | 48.2 tok/s |

Same regime - short context, single stream, graphs on, dense geometry - so they are published as
**about 48 to 50 tokens per second**, with the difference attributed to prompt depth (a 117-token
prompt against a 4,000-token one) rather than to a conflict between two read-outs.

A third reading of the same regime does not sit inside 10% of those, and it has to be stated rather
than hidden: the single-stream comparison series reads the engine's own counter and reports 67.3
tokens per second at a 117-token prompt and 63.3 at a 4,570-token prompt, with the same KV dtype and
the same graph state. That is about 40% above the two readings above, which is far outside the
run-to-run band, and the read-out path is the leading candidate for the difference: the comparison
series divides the engine's per-output-token counter deltas, while the stress series records a rate
without the rule that produced it. **So this document publishes the engine-counter reading as primary
for short-context decode and the 48-to-50 figure as the stress series' reading, and does not claim
they are the same measurement.** Separating them needs one point taken both ways in one session,
which was not done.

### Limits of this chapter

1. The ladder is not a matrix: prompt depths, output lengths and graph states vary across its rows,
   and the two rounds of one configuration differ by 2.2x.
2. Pool peaks and preemption counters exist for the five-request points and for the second geometry.
   For the single- and three-request points they were not retained, and those cells say so.
3. Every aggregate column is a computed multiplication, not a measured total.
4. Nothing here measures output quality, and nothing here quantifies the work a preemption repeats.
5. The occupancy heuristic is not a substitute for the rejection rule. A configuration that stays
   under the ceiling and still reports a non-zero preemption count did not hold its pool, whatever
   its average occupancy says.

## Switch matrix: one setting changed at a time

### Method

This chapter is the protocol's own switch matrix: the request surface of the comparison matrix, run
once per setting. One request in flight, temperature 0, a fixed output length of 256 tokens, the same
fill corpus, a per-request salt that defeats prefix reuse, four depths asked for (0, 8,192, 32,768 and
98,304), one discarded warm-up and three measured requests per point, median reported. The read-out is
the same server-reported decode rate the matrix uses; the client-side rate stays in the per-point
records and is not published as a comparison value, for the reason given in the comparison chapter.

A row is a switch because **one** setting differs from that engine's baseline column of the comparison
matrix. Two rows do not satisfy that condition, and both say so where they appear: the KV-dtype point
of the first engine also moved two capacity parameters, because that is how the protocol's second
configuration is defined, and the third engine's row is a model swap rather than a setting, because on
that application the relevant setting does not exist in the running server at all. Every other row is
single-variable, and each carries a start-up signature in its own record that shows the switch took
effect: a weight footprint that fell by 0.7 GiB when speculation was removed, a KV pool whose byte
cost halved when its dtype changed, a draft head that added 1.7 GiB of weights.

Two read-out gaps are inherited from the matrix and travel with the tables below:

- the first engine publishes no prefill counter, so its baseline prefill cells are empty (comparison
  chapter). Its two switch points carry a supplementary one-token-probe prefill reading instead, which
  is comparable between those two points and with nothing else.
- the per-point records leave the draft-acceptance field empty for two engines: one reports acceptance
  under a different name than the harness maps, and the other does not report it at all. Acceptance
  figures appear here only where an engine's own request lines carry them, with that provenance stated
  beside them.

### The baseline columns

Server-reported decode, tokens per second, median of three, as published in the comparison chapter.

| Engine | Baseline setting | 0 | 8,192 | 32,768 | 98,304 |
|---|---|---|---|---|---|
| vLLM | NVFP4 weights, `int4_per_token_head` KV, no speculative decoding | 67.26 | 63.32 | 58.26 | 45.49 |
| NInfer | fp8 KV, 3 draft tokens, 196,608-token ceiling | 262.51 | 252.97 | 251.69 | 230.94 |
| LM Studio | Q4_K_M weights with a draft head inside them, `q4_0` KV, 201,000-token context | 90.97 | 85.29 | 134.33 | 72.52 |

The third engine's 32,768 cell is the one its draft head lifted, and it is why two rows of the matrix
below have to be read carefully.

### The matrix

One row per switch, at the same four depths. Decode is server-reported, median of three; the bracketed
figure is the change against the baseline cell above it; the average is the mean of the four changes -
computed, not measured. The last column is the price, and it is the column that decides whether a
switch is worth having.

| Engine | Setting changed | 0 | 8,192 | 32,768 | 98,304 | Average | What the switch costs |
|---|---|---|---|---|---|---|---|
| vLLM | CUDA graphs off (eager flag) | 13.04 (-80.6%) | 13.21 (-79.1%) | 14.73 (-74.7%) | 14.91 (-67.2%) | **-75.4%** | Nothing is bought. Weight footprint and pool are unchanged; what comes back is a graph capture of 0.11 to 0.18 GiB at that KV dtype and a shorter start-up. Prefill and wall clock also pay. |
| vLLM | KV dtype `int4_per_token_head` -> `fp8`, with `max-model-len` 114,688 -> 73,728 and utilisation 0.94 -> 0.92 | 71.47 (+6.3%) | 70.22 (+10.9%) | 68.98 (+18.4%) | 66.13 (+45.4%) | **+20.2%** | A third of the usable context ceiling taken away (73,728 against 114,688) and half the pool (108,423 tokens against 234,970). Not a single-variable point. |
| NInfer | Speculative decoding off | 79.97 (-69.5%) | 77.47 (-69.4%) | 77.02 (-69.4%) | 71.77 (-68.9%) | **-69.3%** | 0.7 GiB fewer weights and 0.7 GiB more free memory; prefill is unchanged. The rate is the price, and there is no other side to the trade. |
| NInfer | KV dtype `fp8` -> `nvfp4` | 254.55 (-3.0%) | 254.89 (+0.8%) | 248.57 (-1.2%) | 236.30 (+2.3%) | **-0.3%** | None measured. Memory runtime falls from 7.76 to 4.92 GiB at the same 196,608-token capacity, so 2.84 GiB comes back as free memory. |
| NInfer | Draft scheme `mtp` (3 tokens) -> `dflash2` (7 tokens) | 465.95 (+77.5%) | 472.77 (+86.9%) | 461.37 (+83.3%) | 396.88 (+71.9%) | **+79.9%** | 1.7 GiB more weights, 160 MiB more host-pinned state, 81.5 MiB of free device memory left, 4.3% to 11.8% less prefill, and one depth whose median is not settled (below). |
| LM Studio | Model artifact: Q4_K_M with a draft head inside the weights -> the previous version line of the same family, Q6_K, no draft head | 62.19 (-31.6%) | 59.42 (-30.3%) | 51.53 (-61.6%) | 37.29 (-48.6%) | **-43.0%** | A weight file about 5 GB larger (23.01 GB on disk; 21.43 GiB of weights as the application reports them), device usage measured at 29,705 to 29,764 MiB against a 2,426 to 2,591 MiB idle desktop, 9% to 12% less prefill at the three deeper points, and two variables moved at once. |

Every cell above is a median of three samples whose spread stayed inside the protocol's 10% band
except one: the deepest cell of the draft-window row, which is dealt with under "What these results
cannot be read as". Nothing in these points was evicted or truncated - every request returned the full
256 output tokens, no preemption was recorded where an engine reports one, and every point's
start-up signature matches the setting it claims.

### The two readings that belong under the table

The first engine's rows need a prefill reading that its baseline column does not have, so both of its
switch points were measured a second time with a one-token probe that reports prompt throughput from
the serving side. Same probe for both, so the two columns are comparable with each other and with
nothing else.

| vLLM, supplementary prefill (tok/s) | 0 | 8,192 | 32,768 | 98,304 |
|---|---|---|---|---|
| Graphs off | 1,194.52 | 7,799.07 | 5,234.78 | 2,724.95 |
| KV dtype `fp8` | 1,661.61 | 10,655.16 | 9,127.55 | 6,468.39 |

The gap widens with depth: 1.4x at the shortest probe, 2.4x at the deepest one, which is 8.3 seconds
of prompt ingestion against 19.6 seconds at 98,304 requested. That is the mechanism behind the client
column of the fp8 row, which reads +103.1% where the server column reads +45.4%: the client rate
carries the prefill time of its own single request inside the same wall clock.

The second engine's request lines carry the draft-acceptance counts, so they are published for the
three points where they exist - including the baseline, whose own line was re-read for this table. The
counts in the last column come from those lines and include each point's discarded warm-up; the
percentages themselves agree between the two available readings of the same requests (the per-point
records and the engine's request lines) to within a tenth of a point.

| NInfer, draft acceptance | 0 | 8,192 | 32,768 | 98,304 | Drafts per request |
|---|---|---|---|---|---|
| `mtp`, 3 draft tokens (baseline) | 100.0% | 100.0% | 100.0% | 100.0% | 191 |
| `nvfp4` KV | 99.7% | 100.0% | 100.0% | 100.0% | 191 |
| `dflash2`, 7 draft tokens | 99.7% | 100.0% | 100.0% | 100.0% | 223 |

Read as a pair with the matrix: the 3-draft scheme was already accepting essentially every draft it
proposed on this prompt, so the faster row above is not a better draft head - it is the same near-total
acceptance cashed in over a longer draft window, 223 proposals per request against 191.

### vLLM: the largest single switch in this document, and one that pays only at depth

**CUDA graphs.** Turning them off is the largest switch measured anywhere in this document: decode
falls from 45 to 67 tokens per second to 13.0, 13.2, 14.7 and 14.9, an average of -75.4%, and the
profile of the loss says why. The graphs-off rates are nearly flat across depth, so the path is no
longer the one the graphs-on column is on: with graphs off the engine pays per-kernel launch overhead
that does not care how deep the context is, and the deeper the context, the more of the graphs-on rate
the fixed overhead has already given up, so the *relative* loss is smallest at the deepest point
(-67.2%) and largest at the shortest (-80.6%). The measurement's own cost is visible in wall clock:
one request took 19 to 37 seconds instead of a few.

The concurrency chapter carries a graph pair of its own, from the stress series: 13.9 against 48.2
tokens per second at a prompt of about 4,000 tokens with 500 output tokens, +247%. Different depths,
different output length, a different read-out rule and a pool state that chapter spends a page
disclaiming, so the two pairs **are not comparable figures**. What they agree on is the shape: with
graphs off the rate lands near 14 tokens per second whatever else changes, and with graphs on it moves
with depth. Off is not a tuning option here; it is the setting a device runs when it cannot start
otherwise.

**KV cache dtype.** The other direction, and a trade rather than a loss: +6.3% at the shortest context,
+10.9%, +18.4%, and +45.4% at the deepest, averaging +20.2%. The gain grows monotonically with depth,
which is what the arithmetic predicts - the attention work that reads the cache scales with the tokens
attended over, so a narrower cache reads less per step, and the shallower the context the less there is
to read. What it costs is capacity, and the cost is paid by the whole configuration rather than by the
cache alone: the request ceiling falls from 114,688 tokens to 73,728 and the pool from 234,970 tokens
(2.05x the ceiling) to 108,423 (1.47x). A configuration that needs a 100,000-token single request
cannot hold this point at all, whatever the rate says.

It is also the one row in this chapter that is not a single-variable experiment: the dtype, the ceiling
and the utilisation fraction moved together, because the protocol defines them as one alternative
configuration. Read it as configuration B against configuration A, not as "fp8 is 20% faster". A
strictly single-variable point - fp8 dtype at the unchanged 114,688-token ceiling - was not run, and at
that ceiling the narrower cache may not fit the same pool on this card.

### NInfer: one free switch, one flat loss, one large gain with a price

**Speculation off** is the counterfactual for both speculative columns of this document, and it is
flat: 80.0, 77.5, 77.0, 71.8 tokens per second across four depths, an average of -69.3%, with prefill
essentially unchanged (within 1.5% at the three deeper points). A rate that barely moves with depth is
the signature of a configuration bound by weight bandwidth rather than by attention, and it puts the
upper bound on what this artifact can do per forward pass. It also prices the baseline: whatever the
draft scheme costs, it buys 3.2x to 3.3x of the unspeculated rate at every depth measured.

**The KV dtype switch is free on this engine.** `fp8` to `nvfp4` changes decode by -3.0%, +0.8%, -1.2%
and +2.3% - an average of -0.3%, inside the run-to-run band this document quotes, so the honest reading
is "no measurable change" rather than a small loss. The reason to make the change is not the rate: the
engine was started with the same fixed 196,608-token capacity in both points, and that pool's runtime
memory falls from 7.76 GiB to 4.92 GiB, taking free device memory from 2.02 GiB to 3.70 GiB. It is worth
stating what that is and is not: the saving appears as free memory, not as more context, because
capacity here is a start-up flag rather than a fraction of whatever is free. Prefill is slightly lower
on the narrower pool (1.0% to 2.6% at the three deeper points); decode is not.

**The draft window is where the rate is.** Moving from a 3-token draft scheme to a 7-token one raises
the server-reported decode to 466, 473, 461 and 397 tokens per second, an average of +79.9%, and the
acceptance table above shows the mechanism is the window rather than the hit rate: acceptance was
already at the ceiling. The client column reads far less at depth (+7.1% at 32,768 and -5.5% at 98,304)
for the same reason as the first engine's fp8 row - the client rate carries prefill, and the wider draft
window makes prefill slower, by 4.3% to 11.8% at the three deeper points.

The price is memory, and it is nearly total: 21.7 GiB of weights against 20.0, 747 MiB of host-pinned
state against 587, and 81.5 MiB of free device memory left where the baseline had 2.02 GiB. That
footprint is also the leading candidate explanation for the one unstable cell in this chapter:
at 98,304 asked for, the three samples read 396.88, 441.07 and 222.18 tokens per second - a spread of
55.2%, five times the protocol's threshold - and the separately recorded recheck of that single depth
read 358.55, 437.42 and 429.92 (median 429.92, spread 18.4%), so the slow request reproduces. Its
signature is not engine jitter: prefill and decode were slowed by the same factor, queueing time was
10 to 22 milliseconds, and acceptance was still 223 of 223. A whole request held back at once, on a
configuration with 81.5 MiB of headroom, points at an allocation or a page-in outside the engine's own
scheduling - which is a candidate explanation, not a demonstrated one, and the slow request also
appeared in a schedule where other measurement activity on the same machine cannot be ruled out.
**What is defensible: this row's deepest cell has a true median between about 397 and 430 tokens per
second, one request in six is roughly twice as slow, and the cell must not be quoted to a decimal
place.**

### LM Studio: not a switch

On this application the speculative-decoding state is a property of the loaded artifact, not a runtime
flag, so the only way to run the no-draft-head case is to load a different model. That row therefore
moves the quantisation (Q4_K_M to Q6_K), the tokenizer and the decoding path at once, and it is the
reason it cannot be read as "the MTP head is worth 43%".

What it does show, with those limits attached:

- **All four depths are slower**, by 31.6%, 30.3%, 61.6% and 48.6%. The deepest relative loss is at
  32,768, and that cell is the least informative of the four: the baseline's own 32,768 point is the
  one its draft head lifted (acceptance 0.99, and a decode rate of 134.33 where 8,192 reads 85.29 and
  98,304 reads 72.52), so the -61.6% mixes a lost draft head with a heavier weight format and cannot
  be split.
- **Prefill falls 9% to 12% at the three deeper points**, which the missing draft head cannot explain:
  speculation touches decode only. A heavier weight format read by a prefill phase that is largely
  weight-bandwidth bound explains it, and that part of the row is a clean statement about the artifact
  swap.
- **The acceptance column is empty, and that is a result rather than a gap**: the measurement window
  logged no draft-acceptance line at all, which is the independent evidence that this artifact has no
  usable draft head.
- **The context is unchanged**: 201,000 tokens still loaded (`n_ctx_slot` 201,216), so the four depths
  correspond one to one with the baseline's and no depth had to be dropped.
- **The prompt token count moves 42 tokens the other way at every depth** (75 against 117, 4,529
  against 4,571, 17,880 against 17,922, 53,505 against 53,547): the same corpus through a different
  tokenizer. At the deeper points that is under 1% and does not matter; at the shortest point the
  baseline's prompt is 56% longer than this one's, which is why that point's prefill reading is not
  used as a comparison anywhere in this chapter even though its decode reading is.
- **There is no memory reading for the baseline**, so the device usage measured for this row (29,705
  to 29,764 MiB during the run, against a 2,426 to 2,591 MiB idle desktop, i.e. about 27 GB more than
  idle) has nothing to be compared against inside this document. What it does say on its own is that
  the heavier artifact still fits this card at this context, with about 2 GiB of the 32 GB left.

### What these results cannot be read as

1. **Not as a general speculative-decoding gain.** The fill corpus asks for one sentence repeated
   twenty times, which is about as predictable as text gets, and that is why acceptance sits at or
   near 100% in every speculative row here. The same engine, asked a short free-generation question in
   the same session, accepted 3 of 14 drafts - 21.4% - and the same 3-draft scheme is documented
   upstream at 61 to 62% on ordinary text. The +79.9% of the draft-window row and the speculative gain
   of the matrix both belong to highly predictable output; on open-ended generation they shrink, and
   this chapter contains no measurement of by how much.
2. **Not as a single-variable result for the fp8-KV row.** Dtype, ceiling and utilisation fraction
   moved together by construction. The other five rows are single-variable; that one is not.
3. **Not as the value of a draft head on the desktop application.** The swap changed the quantisation
   and the tokenizer with it, and the missing head cannot be separated from the heavier weights without
   two artifacts that differ only in the head, which this machine does not have.
4. **Not as a settled median for the deepest cell of the draft-window row.** 55.2% spread on the main
   measurement, 18.4% on the recheck of that one depth; the figure is a range, about 397 to 430, and
   the configuration it belongs to is the one that leaves 81.5 MiB free.
5. **Not as free capacity.** The `nvfp4` KV row returns 2.84 GiB of runtime memory and not one token of
   context on that engine, because capacity is fixed at start-up. Realising it as context means
   restarting with a larger capacity, which was not measured.
6. **Not as differences smaller than the run-to-run band.** Identical configurations have differed by
   about 10% between boots on this machine, so the -0.3% average of the KV-dtype row is noise, and a
   single-digit percentage anywhere in the matrix is not a resolved difference. The switches worth
   acting on here are the ones that move a rate by tens of percent.
7. **Not as a comparison of prefill between the engines.** The two supplementary prefill columns exist
   for one engine's switch points only, were taken with a probe whose salt prefix makes its prompt two
   tokens longer than the same depth's main measurement, and have no baseline counterpart.
8. **Not as anything about output quality.** A KV cache dtype and a weight quantisation both change
   what the model would answer, and neither was scored; every figure above is a rate.

### Limits of this chapter

1. Six points, four depths each, one card, one build set, one week. No switch point was repeated as a
   whole, so there is no evidence on how much a switch's own result moves between runs beyond the three
   samples inside each cell.
2. No combination points: graphs off with speculation on, fp8 KV with speculation on, the memory-saving
   dtype with the wider draft window, and draft windows other than the one the protocol named were all
   left unmeasured.
3. Pool figures quoted here are start-up readings, not occupancy observed under load, and no preemption
   counter is carried by these points. The configuration that ran out of headroom did so in free
   memory, not in the pool.
4. The third engine exposes no CUDA-graph switch and no timing API, so it contributes one row and no
   mechanism; its figures exist only because its log prints timing fields.
5. Every measurement here is single-stream and none of it says anything about several requests in
   flight, where the concurrency chapter's own series shows one configuration reading 6.3 and 13.7
   tokens per second in two rounds.

## MoE vs dense on one 32 GB card

<!-- Sources: cn-moe = the internal write-up of the second geometry's measurement and its start-up
failures (2026-09-05); cn-concurrency = the dense stress series quoted in the concurrency chapter. The
geometry, the per-token formula and every ratio below were recomputed from the constants in the
geometry chapter, not copied. The second geometry was **not** run through the identical-request-surface
matrix of this document: the two series quoted here differ in depth, concurrency, output length, KV
dtype and request ceiling, each difference is named where the figure appears, and no cross-geometry
number in this chapter is an identical-protocol comparison. -->

### Two geometries on one card

A recap of the geometry chapter, because it is the whole story of this one:

| | Dense 27B-class | MoE 35B-A3B-class |
|---|---|---|
| Total / full-attention / linear layers | 64 / 16 / 48 | 40 / 10 / 30 |
| KV heads | 4 | 2 |
| KV bytes per token at 1 byte per element | 32,768 | 10,240 |
| Relative to the dense geometry | 100% | 31.25% |
| Measured weights | 21.34 GiB | 23.27 GiB |
| Recommended utilisation | up to 0.93 as remembered | 0.90 or below |

The number of experts, the number of activated parameters and the weight file size never enter the
KV formula. The second geometry holds **31.25% of the dense per-token KV cost - 3.2x as many tokens
for the same pool bytes** - and it also carries 1.93 GiB more weights.

### Speed against context depth

Every cell below is measured; the aggregate column is computed. The two series come from the same
harness family, but not from the same settings, so read the column set with the caveats after it.

| Geometry | Prompt depth | Requests in flight | KV dtype | Per-request rate (tok/s) | Aggregate (computed) | Pool peak | Preemptions |
|---|---|---|---|---|---|---|---|
| MoE | about 0 | 4 | `fp8` | **120.9** | about 484 | 9.7% | 0 |
| MoE | 80K | 4 | `fp8` | 23.3 | about 93 | 80.6% | 0 |
| MoE | 100K | 5 | `fp8` | 13.4 | about 67 | 97.6% | 1 |
| dense | about 4K | 1 | `int4_per_token_head` | 48.2 | 48.2 | not retained | not retained |
| dense | about 48K | 3 | `int4_per_token_head` | 12.2 | about 37 | not retained | 0 |
| dense | about 48K | 5 | `int4_per_token_head` | 6.3-13.7 | about 32-69 | 24.6-50.8% | 0 |

Three readings, each with its own caveat:

1. **Both geometries lose most of their shallow-context rate by deep context.** The MoE falls 81%
   from its shallow point to 80K; the dense model falls 75% from its about-4K point to about 48K
   (both computed). The shape of the curve is the same on both.
2. **The MoE's ceiling is about 2.5x the dense model's** - 120.9 against 48.2 tokens per second - and
   it is reached with four requests in flight, so the comparison against a single-stream dense point
   is not like for like: it is a ceiling comparison, and the MoE's figure already includes whatever
   interference four requests cause. The closest like-for-like pair in the table is the deep one:
   23.3 against 12.2, about 1.9x, at different depths (80K against 48K) and different concurrency
   (4 against 3). **Neither pair is a controlled comparison, and this chapter does not present either
   as one.**
3. **The MoE's aggregate at its shallow point - about 484 tokens per second - is the highest decode
   figure in this document**, and it is a computed product of four requests at 120.9 each, not a
   measured total. What it shows is that four shallow requests on a model with a small active
   parameter count barely interfere (9.7% pool peak, on a second geometry whose per-token cost is a
   third of the dense one).

### Why the second geometry holds deeper contexts

The memory geometry, and only the memory geometry:

- 10 full-attention layers with 2 KV heads against 16 with 4 give the MoE 10,240 against 32,768 bytes
  of state per token, so a pool of a given size holds **3.2x as many tokens**.
- The weight difference acts in the opposite direction on the same ledger: the pool is a remainder
  after the reserve terms, so 1.93 GiB more weights is 1.93 GiB less pool at the same utilisation.
  That is exactly why the recommendation on this geometry is a utilisation of 0.90 or below rather
  than the dense model's remembered 0.93.
- At the configurations actually measured, the two single-request ceilings are much closer than the
  3.2x per-token advantage suggests: the dense geometry at 4-bit KV measured **175,542 tokens**, and
  the MoE's computed ceiling is about **192,800 tokens** (from its recorded start-up limit, at an
  8-bit-class cost per token). About 10% apart, because the MoE's pool is smaller and because its
  measured configuration used a wider KV format. **The 3.2x is a per-token property; the observed
  ceilings are not 3.2x apart.** Had the MoE been run at 4-bit KV, its ceiling would be roughly twice
  the one computed here - an arithmetic statement, not a measurement.

That also settles the "the MoE can hold 200K where the dense model cannot" claim that the material
behind this repository carries. Both ceilings above are below 200K: the dense one at 175,542 tokens is
measured, the MoE's about 192,800 is computed, and the single recorded attempt at a 200K context on
this geometry failed by roughly 0.07 GiB. **No 200,000-token context was served on this card**, and
nothing in this repository claims one was.

### Concurrency and the occupancy ceiling on the second geometry

The three MoE points in the ladder table above are the reason the occupancy rule of the previous
chapter exists: 9.7% with zero preemptions at shallow depth, 80.6% with zero preemptions at 80K, and
97.6% with one preemption at 100K. By the comparison chapter's own rejection rule, the 100K x 5 point
is not a clean measurement. Read together with the dense points, they say something narrower than
"keep the pool under 85%": at every point where the pool was under the line, no eviction occurred;
the one point above it evicted. Three points, one card, one week - a heuristic.

The recorded configuration note for this geometry - a 150,000-token context ceiling with 12 sequence
slots - is a remembered working point, not a measured ceiling. The working point the source material
recommends for multi-request use on this geometry is the 80K x 4 row, chosen because it sits at 80.6%
occupancy with zero preemptions. The 150,000 figure is a compromise against a 200K target, and
publishing it as this geometry's capacity would be wrong in the same direction as publishing the
dense model's 150,000 figure as a capacity (geometry chapter).

### The honest limits of this comparison

1. **Not the identical protocol.** Depths (about 0, 80K, 100K against about 4K and 48K), concurrency
   (4 and 5 against 1, 3 and 5), output lengths (400 against 400 and 500) and request ceilings
   (150,000 against 100,000) all differ between the two series.
2. **Not the same KV format.** The MoE series ran at an 8-bit-class format, the dense series at the
   4-bit per-(token, head) layout: 10,240-class against 22,020 bytes of pool cost per token at their
   working points. That is a factor of about 2 in the pool cost, so a *pool percentage* means
   different things on the two geometries and is not comparable without the per-token cost.
3. **Not the same read-out.** Both series read rates and counters from the serving side, but the
   stress series does not record the rule that turned its output into a rate, and one dense cell
   spans 2.2x across two rounds of an identical configuration (the concurrency chapter). The dense
   cells in the table above carry that spread.
4. **The aggregate column is computed**, as a rate multiplied by a concurrency, and assumes equal
   shares.
5. **No quality measurement.** Reasoning quality on long multi-step work was not benchmarked, and the
   routing behaviour of the second geometry under load was not measured at all. Nothing here says the
   MoE answers better or worse than the dense model.
6. **One start-up failure of this geometry carries a label that does not reconcile.** The source
   material records a failed 200K attempt at 4-bit KV with a speculative head; the pool it needed,
   2.04 GiB for 200,000 tokens, implies about 10,952 bytes per token - an 8-bit-class cost, not a
   4-bit one. This document therefore quotes that failure's limit (192,832 tokens) and its pool
   (about 1.97 GiB) and **not** its dtype label, and it treats the label as an open calibration item.
7. **The launcher's estimator was reading the wrong constant for this geometry.** Its per-token
   constant at the time was the dense model's, so any capacity it displayed for the MoE was computed
   from the wrong geometry. The source material records this as uncalibrated. No estimator-derived
   MoE capacity is published here.
8. **The start-up cost of this geometry is not a serving number.** The artifact is a multimodal
   checkpoint, so its first start profiles image inputs, and the first start on this geometry can
   compile a large graph set whose host-memory peak exceeds physical memory and triggers swapping -
   which reads as a hang rather than as a failure. Cached afterwards, the same start is quick. All of
   that is reported by the source material and was not re-measured here.

What survives this comparison: the per-token geometry does what the formula says, a pool of a given
size holds 3.2x the context on the second geometry, the shallower-context speed advantage is real
(about 2.5x at the measured ceilings), and the deep-context occupancy points behave the way the
per-token cost predicts. What does not survive: any claim that the second geometry is faster at every
depth, any claim about its quality, and any claim that a 200K context was held on this card.

## What each engine actually optimises for

<!-- Sources: cn-ledger and cn-compare = the internal two-engine comparison write-up and its own
later correction sections, which are the source of the memory-bookkeeping material; cn-concurrency
as in the concurrency chapter and cn-moe as in the previous one; the comparison matrix of this
document for the across-engine figures; the start-up logs behind the geometry chapter for the
per-token costs. This chapter is where mechanism is read off measurements, so every sentence that
reasons past a measured number is marked as an inference where it is made, and the measured facts it
reasons from are named.
The corrected version of the two attributions the source material superseded is used here; the
superseded versions appear nowhere in this document. -->

### The three budget models

This is the difference with the largest effect on what a user can actually run, and it is a
bookkeeping difference, not a performance one. Measured on this machine, all three configurations in
the comparison matrix held about 28.8 to 30.6 GB of device memory while serving the same model
family; what they could do with it differs.

- **vLLM** takes its budget as a fraction of the whole device (`gpu-memory-utilization`), charges
  reserve terms against it - peak activations, non-torch overhead, the CUDA-graph capture, and the
  speculative-decoding head when enabled - and calls the remainder the KV pool. The ledger and its
  constants, including the +-12% band on the pool, are in the geometry chapter.
- **NInfer** takes a KV capacity in tokens at start-up and does not derive it from a utilisation
  fraction. Its value in these measurements, 196,608 tokens, is a declaration rather than a
  remainder: a number the operator states and the engine either honours or refuses.
- **LM Studio** has no utilisation ceiling: weights and KV state go into free device memory, and the
  KV side can grow until the device fills.

**Inference, from the measured ledger:** this is why the same card can hold a 200,000-token-class
context on one engine and refuse it on another. The first engine's reserve terms are about 4.2 GiB at
the measured constants (computed: activations 1.92 + non-torch 1.9 + graphs 0.4), and they are charged
before the pool exists, while the other two write a KV budget down and then fill it. What is measured
is the ledger table, the per-token pool costs and the three measured footprints; the allocators
themselves were not instrumented, so the claim that this is the deciding mechanism is inference.

### Batching and prefix handling

The first engine schedules with continuous batching over a paged KV pool: a batch is composed at each
step from whatever requests are in flight, and the pool is allocated in pages, which is what makes it
possible for two requests to share a prefix. Three measured consequences, and one piece of inference:

- Four requests in flight on the second geometry kept a per-request rate of 120.9 tokens per second
  with a 9.7% pool peak - the four requests barely interfered - and the computed aggregate over them
  is about 484 tokens per second, far above any single request's rate. **Inference:** an aggregate
  above any individual request's rate is batch-level parallelism, and the pool peak is what says so.
- Five requests carrying identical content held a 22.3% pool peak where five with distinct content
  held 50.8% (the concurrency chapter). **Inference:** the first figure is prefix merging - one
  cached copy serving five requests - and it is the mechanism the comparison chapter's per-request
  salt exists to disable. That the requests were merged is measured through the pool counters; the
  bookkeeping that performs the merge was not read.
- The engine's speculative-decoding gain collapses with depth and goes negative at the deepest point
  (-33.9%), while a second engine on the same card holds its rate flat with a 1.00 draft acceptance at
  every depth (comparison chapter - that 1.00 came from the engine's own request logs, under a prompt
  whose output is one sentence repeated 20 times, and the engine's own documentation records 61-62% on
  ordinary text, so it belongs to the measurement condition rather than to the engine).
  **Inference:** the first engine's draft work scales with the context it must attend over while the
  tokens it saves per step are a fixed small number, so the two costs cross; the second engine's draft
  scheme evidently does not pay that attention cost on this workload. Neither engine's per-step
  composition was instrumented here.
- The closed-source comparison application has no continuous batching, so its per-request rate is what
  its log reports for one request at a time. The comparison protocol therefore never issued two
  requests to it at once, and no concurrency figure for it appears anywhere in this repository. That
  absence is a property of the method, not a finding about the engine.

### Prefill strategy

Prefill is where the engines separate most cleanly in the measurements that exist:

| Prompt tokens | NInfer | LM Studio |
|---|---|---|
| about 4,500 | 9,102 tok/s | 3,288 tok/s |
| about 17,900 | 8,563 tok/s | 3,333 tok/s |
| about 53,500 | 6,236 tok/s | 2,804 tok/s |

The first engine ingested these prompts 2.2x to 2.8x faster at every depth measured (computed). The
third engine's cells are empty for the reason given in the method chapter: it publishes no
independent prefill counter and the client-side combination was never implemented - so **the engine
whose reported mechanism is strongest at prefill has no prefill figure under this protocol at all.**

What exists instead, from the material behind this repository, is a report rather than a measurement:
that the closed-source engine's prefill was slow enough on long prompts to trip request timeouts
repeatedly, and that switching to the open-source engine removed the problem. Three reasons are
offered there for that engine's prefill behaviour: its batch-size parameters are not exposed in its
interface; some quantisation families it supports have more expensive dequantisation block
structures, and prefill is the compute-bound phase; and an open item in its tracker describes a
prefill regression after a driver generation. **The first is structural and visible in the interface;
the other two are claims from elsewhere and are listed as unverified in the last chapter.**

**Inference:** the prefill gap and the batching gap are the same design difference seen from two
sides - one engine is built to ingest long prompts and run several requests at once, the other to run
one conversation as fast as possible. The measurements support the first half of that (multi-request
aggregate and ingestion rates); the design intention is inference.

### Memory bookkeeping: the corrected attribution

The source material behind this repository first attributed one engine's more constrained memory position to the fact that it captures CUDA graphs where the other does not. **That attribution is superseded and does not appear in this document.** The corrected version, which is what is stated here:

- The **main** cause is the budget model above: one engine charges reserve terms against a
  whole-device ceiling before the pool exists; the other writes a weight-plus-KV configuration
  straight into free memory. The direction of the original claim was right; its magnitude was not.
- The CUDA-graph capture is a **secondary** term: 0.11 to 0.51 GiB depending on the attention
  backend, measured across the dtype series. It is also a cost that should still be paid, because
  disabling it costs 40% to 247% of the decode rate (the concurrency chapter).
- The comparison "one engine adds a graph and the other does not" is not supportable in the first
  place. The closed-source engine also performs kernel fusion and persistent-kernel work, and its log
  carries a reuse counter that runs into the thousands of lines; it simply exposes no switch. **The
  earlier claim that it skips dequantisation for its 4-bit KV format is also wrong:** that format is
  dequantised as well; the difference is that its dequantisation is fused into a persistent attention
  kernel, where the per-(token, head) scale layout measured in the chapters above carries a
  separately visible per-step cost. That fusion claim is a code-level claim carried by the source
  material and was not re-measured here.
- The same source states a scope limit for the observation that follows from it: "4-bit KV is slow"
  belongs to long contexts, where the whole history is unpacked at every step. At a near-zero context
  the same configuration was measured at 120 tokens per second on the second geometry. It is a depth
  effect, not a property of the format.

### KV quantisation, engine by engine

The three engines were not holding the same state per token in the comparison matrix: 22,020 bytes
per token on the first (`int4_per_token_head`, measured), an 8-bit-class format on the second
(`fp8`), and a block-quantised 4-bit format on the third (`q4_0`, which the geometry formula puts at
16,384 bytes per token if it is a pure 4-bit layout - **computed, and the closed-source engine's own
per-token accounting could not be read**, so that is a formula value and not a measured pool cost).

**Inference:** the same nominal decision - "use a lower-precision KV cache to hold more context" -
therefore costs each engine a different share of its own per-token budget, and the depth at which
each engine's pool fills is decided by geometry and bookkeeping before it is decided by anything an
engine does. What is measured is the three formats and two of the three per-token costs; the claim
that this is what sets the practical depth ceiling follows from the pool arithmetic and is an
inference from it.

### What survives once the request surface is identical

The comparison chapter held five things constant - the depth as each engine reported it, the output
length, the sampling temperature, the speculative-decoding on/off state, and one request in flight -
and named the things that do not match (weight format, KV format, speculation scheme, the definition
of the acceptance rate). With that surface fixed, the following statements survive:

1. **Each engine's own decode rate falls with depth, and the ordering between engines is stable
   across the four depths**, with the leading engine first at every point by a margin far outside the
   run-to-run band.
2. **One engine's decode rate is nearly flat in depth** (-12% over the whole range, measured) where
   the other two lose 20% to 32% end to end. **Inference:** flatness of that kind comes from the work
   per step growing more slowly with depth than on the other two, which is consistent with that
   engine's draft acceptance reading 1.00 at every depth (from its own request logs, under this
   protocol's repeated-sentence prompt rather than on general text - the acceptance-rate section of
   the comparison chapter states the limit). Its per-step composition was not instrumented, so this
   is inference.
3. **The fastest cell in the matrix is not the shortest context.** The closed-source engine's best
   point is its third depth, where its acceptance rate reached 0.99 (measured). **Inference:** with
   speculation on, the per-step cost is set by the acceptance rate and the draft cost rather than by
   depth alone - which is also the shape of the first engine's speculative gain, which collapses with
   depth and turns negative at the deepest point (a measured, reproducible -33.9%).
4. **Nothing about concurrency survives**, because the matrix is one request at a time. The
   concurrency and occupancy figures in this document come from a different series with its own
   read-out rules and its own 2.2x spread; they are not part of this surface.
5. **Nothing about quality survives**, because no engine was given a task whose answers could be
   scored. Every figure in this document is a rate.

### Reading rule for this chapter

Every paragraph above that reasons past a measured number says so where it does it. The measured
facts it reasons from are: the three budget models as configuration, the ledger constants and the
three measured footprints, the pool-peak pairs with and without identical content, the
aggregate-against-single ratios, the prefill rates of the two engines that publish one, the CUDA
graph pair, the acceptance rates, the per-token costs of two of the three KV formats, and the
depth curves. Everything else here - allocator behaviour, batch composition, kernel fusion, why one
rate is flat or one acceptance rate is high - is inference from those facts and is offered as such.

## What we did not verify

<!-- Sources: every chapter of this document, plus the source material named in them. This chapter
exists so that a reader does not have to reconstruct, from nine chapters of measurements, which
numbers are this machine's and which are somebody else's. -->

### Figures that came from outside this machine

The material behind this repository mixes measurements from this machine with readings from other
sessions, community reports and vendor documentation. The rule applied in this document - stated in
its method chapter - is that a figure from outside this machine is labelled as such in the sentence
that quotes it, or it is not quoted. These are the ones that were kept, with what each actually is:

| Figure | What it is | Its nature |
|---|---|---|
| 78.5 tok/s, 200K context (201,000 in this document's configuration snapshot), on the closed-source engine with its draft head | an earlier session's reading on this machine | not part of this protocol, and **not reproduced by it**: the protocol pre-registered a judgement window for this figure, and this protocol's 0K point for that configuration falls outside it. The check is stated in full below the table. |
| 58 tok/s for the same model without a draft head | an earlier session's reading | quoted as the no-speculation comparison, on a checkpoint whose weights differ from the one in the matrix; not re-measured |
| about 11.9K tok/s prefill at 8K on the open-source engine | a community measurement on a similar card | no method and no run record; used nowhere as a comparison value |
| the closed-source engine being 30-50% slower than a source build of the same inference library | a community claim | no run record here; presented as a caveat on that engine's numbers, not as a measurement |
| about 26.8 ms per output token at near-zero context | an average over a 16-request histogram from an earlier session | the histogram itself is not retained |
| a context configuration on the closed-source side that an earlier note records as 20,000 tokens | a per-model setting, and one whose own records disagree with that note: the same sessions carry 200,192 to 231,168 tokens, and this document's configuration snapshot quotes 201,000 | the 20,000 reading is not reproducible from those records and is used as a depth nowhere in this document; the same-protocol matrix ran that configuration at 201,000 |
| any statement that a 200,000-token context "fits" | a formula result | the single recorded attempt at that size failed by about 0.07 GiB; the ceilings published in this document are one measured capacity and several computed ones |

**The 78.5 check, stated once.** One row above carries a figure whose status has to be explicit, and
the check that decides it was designed before the measurement rather than after it. The protocol
behind this document pre-registered the judgement for that earlier reading: a 0K-depth, single-stream
decode measurement landing inside **74.6 to 82.4** - the earlier number plus or minus 5% - and
carrying a draft acceptance rate of at least 0.8 would count as that reading standing up. The 0K point
this protocol actually measured on that configuration is **91.0 tokens per second at an acceptance
rate of 0.89**, which is about 10% above that window's upper bound and about 16% above the earlier
record. **The window is not met, and nothing in this document should be read as a reproduction of
78.5.** The statement the measurement does support is the weaker one: under that configuration, 0K
single-stream decode measured 91.0 in this round, roughly 16% above the earlier record, so the earlier
number's magnitude holds.

Depth has to travel with both halves of that sentence. At the protocol's 96K step - a nominal
98,304-token request that arrived as 53,547 prompt tokens - the same configuration measures **72.5**,
below the lower bound of that window. So neither the window judgement nor the comparison above it is
a level for that engine: both are 0K readings, and the deep points of that configuration are not part
of them.

### What was never measured

1. **The quality of any quantised KV cache.** Six candidates started and each answered one request
   with a numeric answer expected. The 4-bit candidate's long-context recall and coherence are
   unmeasured, and the fill corpus used throughout is a repeated text block rather than a document
   with facts to retrieve, so nothing here tests retrieval at any precision.
2. **A 16-bit-class KV reference.** No such candidate runs on this checkpoint, so the KV chapter has
   a measured capacity ladder and no measured quality ladder and no upper anchor.
3. **Output quality of any engine.** No task with scorable answers was run. Every published figure is
   a rate.
4. **Concurrency under this protocol.** The comparison matrix is one request at a time; the
   concurrency and occupancy figures come from a different series whose identical-configuration
   rounds differed by 2.2x.
5. **The combination points of the switch matrix.** Chapter 6 carries one point per switch under the
   comparison protocol, but nothing there puts two switches together: graphs off with speculation on,
   the narrower KV cache with the wider draft window, and the memory-saving dtype with any of the
   others were all left unmeasured. The fp8-KV point is also still a two-variable point, and no
   switch point has an under-load pool occupancy or a preemption counter.
6. **The second geometry under the comparison protocol**, and its ceiling: about 192,800 tokens is
   computed from a start-up limit and a pool, never measured; its 150,000-token working point is a
   remembered configuration.
7. **The per-depth draft acceptance and the preemption counters for the speculative series.** They
   were read during the run and not retained per point, so that engine's deep-context collapse is
   reproducible but not decomposed between draft overhead and pool pressure. The acceptance column of
   the leading engine in the matrix has the same defect from the other side: the harness's field is
   empty for that engine, and the published value came from its own request lines, which is stated
   where the column appears.
8. **Where the recurrent state is charged inside the pool.** The residual is measured and flat across
   dtypes; the *position* of the recurrent state in the ledger is not settled by the retained
   evidence, and the allocation format it uses comes from a code reading rather than a measurement.
9. **The launcher's estimator constants for the second geometry.** They were never calibrated - the
   per-token constant was the dense model's - so no capacity the estimator displayed for that
   geometry is published here.
10. **Reproducibility beyond this machine.** One host, one operating-system pair, one build set in
    September 2026. Identical configurations have differed by about 10% between boots and pooled
    capacity by about 7%; nothing here should be reproduced elsewhere without re-measuring, and the
    two series that produced the concurrency numbers spread by 2.2x between rounds of one
    configuration.
11. **A third checkpoint of the same family.** The launcher's menu on this machine also offers an
    earlier member of the family (a 27B artifact of the previous version line), and no measurement
    in this repository covers it: its geometry, its speed and its capacity are unknown here, and no
    figure in these chapters describes it. Every statement in this document is about the two
    artifacts named in its method chapter.

### Licence terms

The licences of the artifacts and applications involved were not read while this document was
written, and nothing here asserts any right to redistribute them:

- the quantised weight releases served by the two open-source engines, and the upstream model's own
  terms;
- the third-party quantisation and packaging of those weights;
- the closed-source desktop application used as the third comparison point. It appears here only as a
  comparison point, no file from it is redistributed, and only numeric fields extracted from its
  server log are quoted. Producing those fields required enabling a logging switch; the switch was
  turned back off after the measurement window, and the log files produced during that window were
  truncated back to their pre-measurement size. **No log line, request or response body, or
  filesystem path from it appears in this repository or anywhere it points to.**
- the open-source serving engine and its kernel and runtime dependencies, whose licences are stated
  in their own projects rather than repeated here.

The repository's own LICENSE covers the text and the scripts in it, and says nothing about the
model artifacts.

### Upstream issue numbers

The source material behind this repository quotes three upstream issue numbers as support for
mechanisms described here: a capability gate that restricts a 4-bit key/value cache to the
data-centre compute family; a silent repetition collapse reported for one key/value backend combined
with speculative decoding; and a prefill regression after a driver generation in the closed-source
application's tracker. **None of the three was checked against a link, a title or a date while this
document was written, and no issue number therefore appears anywhere in it.** The mechanisms are
described instead, and where they bear on a measurement they are tied to it: the capability gate to
the refusing candidate in the KV chapter, the repetition collapse to the warning that this
combination is untested on this machine, and the prefill regression to the unverified claims in the
engines chapter. An identifier that cannot be verified is not evidence, and a document that publishes
one is asking to be trusted rather than checked.

### Method gaps that limit every number above

1. **The KV dtype series' driver was not retained**, so those seven start-up points cannot be re-run
   as they stand.
2. **The stress series does not record the rule that turned its output into a rate**, which is the
   leading candidate for the 40% disagreement between its short-context figure and the comparison
   series' engine-counter figure on the same regime.
3. **The closed-source engine's figures depend on a log format.** Its numbers exist only because its
   server log prints timing fields in a parseable shape, on a closed-source application nobody here
   controls; a formatting change upstream would silently remove the source of those figures.
4. **The client-side wall-clock column is deliberately unpublished** as a comparison value: it
   includes network time, JSON parsing and time to first token, and at one point it diverges from the
   server figure by 6.5x.
5. **One of the three engines has no prefill figure** under this protocol, and the two that do were
   measured by two different read-out paths.
6. **The nominal depths overshoot by up to 45%**, so every published figure is stated against the
   prompt token count each engine reported rather than against the number requested.
7. **The rejection thresholds were applied as rules and fired asymmetrically**: the 32-token floor
   and the 400 tok/s ceiling never fired in the comparison matrix, while the preemption rule is what
   makes the deepest occupancy point in the concurrency chapter not a clean measurement.
8. **Only one engine is resident at a time** - three configurations each need roughly 28 to 31 GB of
   a 32 GB card - so every comparison in this document is a sequence of measurements rather than a
   simultaneous one, and the boot-to-boot band applies to each configuration separately.

### What this repository does claim

That one machine's numbers, taken under a stated method, support the specific statements labelled as
measured in the nine chapters above - and nothing further. The statements that did not survive
recomputation were corrected or dropped rather than carried, the mechanisms that were inferred are
marked as inference where they appear, and the material that came from elsewhere is either labelled
as such or absent. A reader who wants to disagree with a figure has, for every figure, the method
that produced it and the source it came from.
