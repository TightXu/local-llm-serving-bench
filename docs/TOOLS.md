# Tools

Everything in `scripts/` is a plain script: no package to install, no build step, no service to
run. Three of the five files are used in normal work (the launcher, the pressure script, the
browser page); `start_cg_on.sh` is a one-off reference command and `saved_params.json` is a data
file the launcher owns. This document is written against the code in this repository, not against
a description of it.

Two general notes before the file-by-file detail:

- **These tools are for a vLLM server started on the same machine.** Where a script assumes a
  port, a path layout or a metrics endpoint, the assumption is stated rather than silently
  inherited.
- **Nothing here installs anything.** The environment they expect is described in
  [QUICKSTART.md](QUICKSTART.md).

---

## `select_and_run.py` (517 lines, Python 3 standard library only)

An interactive launcher for `vllm serve` on a single consumer Blackwell card under WSL2. It was
written to make one repeatable thing easy: pick a model, adjust the handful of parameters that
actually matter on a 32 GB card, see an honest estimate of what will fit, and start the server -
with the parameters remembered for next time.

```bash
# from the directory that holds it, inside the Linux environment
python3 scripts/select_and_run.py            # interactive
python3 scripts/select_and_run.py --dry-run  # build the command, print it, start nothing
```

`--dry-run` is the only recognised command-line flag, and anything else you pass is ignored.
Everything else is answered at the prompt, so the script needs a terminal: it reads standard
input, and it is not meant to be driven by a service manager. Its banner reports the intended
target (`vllm 0.28.0`).

### What it does, in order

1. **Lists models.** It scans both roots - the Windows-side root first, then any name that exists
   only on the Linux side. A directory counts as a model if its name does not end in `_disabled`.
   If neither root yields anything it prints `[ERROR] no models found in either directory` and
   exits with code 1 before showing a menu.
2. **Picks one.** Enter selects the default, which is the entry whose name contains `Qwen3.8`
   (marked with an asterisk in the list). The model directory must contain `config.json`; if it
   does not, the script reports an incomplete model directory and exits.
3. **Resolves which copy to serve.** The Linux filesystem copy is preferred, and the Windows-side
   root is a fallback. This is not cosmetic: reading weights across the Windows mount is roughly
   an order of magnitude slower, which is the difference between a 20-second model load and a
   20-minute one.
4. **Loads remembered parameters** for that model name, if any, and overlays them on the built-in
   defaults. Remembered values win. See `saved_params.json` below - this is the single most
   surprising behaviour in the tool.
5. **Asks for the mode:** Enter starts with the current values, `a` enters advanced mode.
6. **In advanced mode** it walks eleven parameters in order. Enter keeps the bracketed value, `/?`
   prints that parameter's help text, and the parameters with a fixed set of values are answered
   by number. The help text carries the measured numbers, which is where the defaults come from.
7. **Shows the VRAM estimate and asks for confirmation**: Enter starts, `a` reconfigures, `q`
   quits. With `--dry-run` it prints the command it would have run and exits instead.
8. **Saves the parameters, then starts the server.** The save happens before the launch, so a
   start that fails still keeps the values, which is the point: probing a context or capacity
   limit means changing one number and re-running.
9. **On failure** it writes the child's full output to `logs/vllm_fail_<timestamp>.log` next to
   the script - the command line, the exit code, then the log - and prints the last line
   containing `ValueError:`, `RuntimeError:` or `OSError:` as a root-cause hint. Ctrl-C instead
   terminates the child and prints `[interrupted]`; that path does not write a log file.

### The environment variable: `VLLM_MODEL_WIN_ROOT`

The Windows-side model root is read from the **`VLLM_MODEL_WIN_ROOT`** environment variable at
start-up. It is the Windows partition's model directory as seen from inside the Linux
environment, and it is empty by default.

- **Set it** to the Windows-side root and the launcher lists models from both roots and can fall
  back to the Windows copy when a model is not on the Linux filesystem.
- **Leave it unset** and the variable is an empty string. `os.path.isdir("")` is false, so the
  Windows root contributes no models, the Linux-side root is searched on its own, and any name
  found only on the Windows side is invisible. Resolution then falls through to a relative path
  that will not hold `config.json`, so the model is reported as incomplete rather than silently
  served from a slow mount. If neither root has a model, the script exits with code 1 and a
  message that shows the empty root as a blank between the two separators - which is the usual
  sign that this variable is missing.
- Setting it does not change what gets loaded from where: the Linux copy is still preferred when
  both exist.

The Linux-side root is the `models` directory in the home directory, and it is not configurable.

### The eleven parameters

Defaults below are the built-in ones, which apply the first time a model is launched. Remembered
values override them per model.

| Parameter | Default | Meaning and the measured reason for the default |
|---|---|---|
| `max_model_len` | 65536 | Per-request context ceiling in tokens. The estimated ceiling for this model on this card is about 97K; set it too high and vLLM reports `KV cache needed > available` at start-up. |
| `max_num_seqs` | 32 | Sequences processed at once. This architecture is hybrid: 48 of 64 layers keep a recurrent state per sequence, so the ceiling is about 82 regardless of free memory. Raising it past that fails at start-up. |
| `kv_cache_dtype` | `fp8` | KV cache storage format, chosen from eight candidates. See the table below. |
| `gpu_memory_utilization` | 0.92 | Fraction of total VRAM vLLM may claim. Leave room for the desktop and the system; 0.92 is the steady-state value used here, 0.94 is a stress-test ceiling, and the larger MoE (mixture-of-experts) weights want 0.90 or less. |
| `temperature` | 1.0 | Sampling temperature, the published recommendation for this family's thinking mode. It is passed to vLLM **only when it differs from this default**. |
| `top_p` | 0.95 | Nucleus sampling cutoff; same pass-it-only-if-changed rule. |
| `top_k` | 20 | Sample from the k most likely tokens; same rule. |
| `port` | 8000 | Service port. The OpenAI-compatible endpoint is `http://localhost:<port>/v1`. |
| `enforce_eager` | `n` | `n` leaves CUDA graphs on (the engine's own default). `y` adds `--enforce-eager` and disables them. Measured: CUDA graphs on gave 13.9 to 48.2 tokens/s single-stream at short context (+247%) and +40% under a long-context load in that round (two other rounds of the same 5 x 48K point read 6.3 and 13.7 - different runs, not comparable), costing 0.11 to 0.51 GiB depending on the KV dtype (start-up log values; an earlier write-up of the same series used 0.15 to 0.51 GiB). Eager is for a card that is out of memory, nothing else. |
| `reasoning_parser` | `qwen3` | `qwen3` moves the thinking trace into a separate `reasoning_content` field; the empty choice leaves it in the output text. |
| `mtp` | `off` | Speculative decoding with 3 draft tokens, added as `--speculative-config`. Only models whose checkpoint ships an MTP head (MTP is multi-token prediction: the model drafts several tokens ahead of the verifier) can use it: the launcher checks the model name against `Qwen3.8` and silently ignores this setting for anything else, printing an informational line in the estimate panel. |

Two things the launcher does not expose. `--tool-call-parser qwen3_coder` and
`--enable-auto-tool-choice` are always passed, not configurable; and the sampling parameters are
only sent when they differ from the defaults above, so the defaults never appear on the command
line - if you need the exact invocation for a record, take it from the line the launcher prints
before starting.

### The KV cache dtype choices, with the measured numbers

Selecting by number in advanced mode; the help text carries the same figures. Capacities are
pooled capacity in tokens for this model at 65536 context on this card.

| Value | Measured pooled capacity | Notes |
|---|---|---|
| `fp8` | ~93K | The launcher's default, and the only option with a long history. |
| `auto` | ~86K | The engine's own default, and the baseline here. Pool cost per token 37,948 B (37.06 KiB). |
| `fp8_per_token_head` | ~102K | Per-token, per-head scales: closer to a 16-bit-class precision at 8-bit capacity. |
| `int8_per_token_head` | ~102K | The int8 member of the same family. |
| `int4_per_token_head` | ~176K | Twice the capacity; the highest precision risk here, and long-context quality was not measured. |
| `turboquant_k8v4` | ~133K | 8-bit keys, 4-bit values, a new backend with no long-context track record; verified to start and answer, nothing more. |
| `nvfp4`, `fp8_e5m2` | fails at start-up | Both are listed in the menu and both are known not to work on this model: the first has no kernel for this compute capability, the second is rejected because the checkpoint is itself fp8-quantised. They are kept visible with their measured outcome rather than removed. |

The counter-intuitive result, measured: `fp8` reduced the per-token pool cost by almost nothing
against the engine's own `auto` default - 37,703 against 37,948 bytes per token, 36.82 against 37.06
KiB, a difference of 0.65%, with the default the higher of the two. There is no 16-bit-class row on
this checkpoint to compare either of them against, so "8-bit KV saves almost nothing here" cannot be
measured on it at all. The pool cost is dominated by the recurrent state of the 48 linear-attention
layers, which shares the same pool and does not read this setting at all.

### The VRAM estimator

Before starting, the launcher prints an estimate and compares it against the selected context.
The estimator is calibrated against measurements from this machine, not read from vLLM, and it
carries a stated error band of plus or minus 12%.

It reads total and free VRAM from `nvidia-smi`, falling back to constants (31.82 GiB total, about
1.6 GiB already in use) if that fails. Then:

```
requested      = gpu_memory_utilization * total VRAM
budget         = min(requested, free VRAM)
KV pool        = budget - weights - activations - non-torch overhead - CUDA graph cost - MTP head
capacity(tok)  = KV pool * 2^30 / bytes-per-token[kv_cache_dtype]
```

The constants it ships with: model weights 21.34 GiB for the dense 27B-class model and 23.27 GiB
for the MoE; peak activations 1.92 GiB; non-torch overhead 1.9 GiB; an extra 0.4 GiB when MTP is
on; CUDA graph cost by dtype, from 0.11 to 0.51 GiB (zero when eager); bytes per token by dtype,
from 22,020 for 4-bit per-token-head to 38,979 for fp8 per-token-head. The weights constant is
chosen by model-name prefix and falls back to the dense value.

Its two pre-flight checks are the useful part:

1. **Free VRAM against the utilisation budget.** If free memory is below `util * total`, vLLM will
   refuse to start (`Free memory ... is less than desired`). A shortfall up to about 1.5 GiB is
   reported as a hint rather than a warning, because background processes usually release that
   much while the weights load; a larger shortfall is reported as a probable failure, with the
   utilisation value that would fit.
2. **KV pool against the requested context.** If the pool ceiling is below `max_model_len`, it
   says so before the load rather than after, and names the two error strings you would otherwise
   meet: `KV cache needed > available`, or vLLM quietly scaling the context down.

The estimator was calibrated on a handful of closed-loop runs and takes no responsibility for
another machine's driver, desktop overhead or background load. Treat it as a way to avoid an
obviously doomed start, not as a prediction.

---

## `kv_bench.py` (144 lines, Python 3 standard library only)

A concurrency and KV-pressure probe for a vLLM server. It fires N requests of a chosen prompt
size at once, and while they run it samples the server's own Prometheus counters, so the numbers
it reports are the server's view rather than a client's guess.

```bash
python3 scripts/kv_bench.py --concurrency 5 --prompt-tokens 48000 --max-tokens 400 --name five48k
python3 scripts/kv_bench.py --base-url http://127.0.0.1:8000/v1 --concurrency 1 --max-tokens 256
```

| Flag | Default | Meaning |
|---|---|---|
| `--concurrency` | 1 | Number of requests, one thread each, started 0.5 s apart. |
| `--prompt-tokens` | 4000 | Size target for the prompt. See the warning below: this is not a token count. |
| `--max-tokens` | 500 | Completion length requested. |
| `--name` | `run` | Label used in the printed summary and in the machine-readable result line. |
| `--base-url` | `http://localhost:8000/v1` | OpenAI-compatible base URL of the server, trailing slash ignored. |

### `--base-url` decides two things

It is both the request endpoint (`<base-url>/chat/completions`) and the source of the metrics
endpoint: the tool removes a trailing `/v1` and appends `/metrics`. Point it at the `/v1` root and
metrics resolve to the server root; point it at the server root itself and the same URL is used
directly. Point it at anything else and the metrics request will 404 or time out.

Two things to know before you run it:

- **It is a vLLM-specific tool, and it does not stop when vLLM is not there.** The counters it reads
  (`vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:generation_tokens_total`,
  `vllm:num_preemptions_total`, `vllm:kv_cache_usage_perc`) exist only on vLLM. Against an engine
  with no `/metrics`, the snapshot returns nothing, the tool prints
  `start: /metrics not reachable - this engine publishes no Prometheus counters` (and the same line
  in place of the end block), and it still sends every request: the per-request wall-clock rates come
  back as usual while the preemption delta and the pool peak read `n/a` in the `[RESULT:...]` line.
  That is a useful run for per-request timing and says nothing about the engine's pool, so do not
  quote its occupancy or preemption numbers at all.
- **The model id is discovered, not passed.** It reads the first id from `<base-url>/models` with a
  5-second timeout. If that fails it falls back to the hard-coded name `Qwen3.8-27B-NVFP4`, which
  will only be accepted by a server that happens to serve that same name.

### `--prompt-tokens` is a size target, not a token count

This is the flag most likely to mislead. The prompt is one fixed filler paragraph repeated
`prompt-tokens / 173` times, which makes the prompt about `prompt-tokens` **characters** long. The
divisor is a property of the original filler text, where roughly one character was one token; the
filler is now English prose, which tokenises at roughly four characters per token. So the prompt
that arrives is roughly **a quarter** of what the flag asks for: `--prompt-tokens 48000` lands
near 12,000 prompt tokens, not 48,000. Treat the flag as a filler-size knob, and read the real
prompt token count out of the server's own response rather than trusting the label. This was left
in place rather than silently recalibrated so that the numbers this tool produced earlier stay
comparable with the ones it produces now.

### The salt, and why it is there

Every request carries a unique salt at the very front of the prompt. Without it, N requests built
from identical text hit the server's prefix or context cache, and all of them then measure the
cache instead of the engine: the cheapest way to get a beautiful and completely wrong concurrency
number. The salt changes content from the first token onward, so each request is genuinely
independent. It also understates how much KV a realistic workload needs, since real requests
usually do share a prefix - the honest reading is "this is what independent histories cost".

### What it prints, and how to read it

The field names and the layout below are exactly what the script prints; the values are one
example run, kept for the shape of the output rather than as a result.

```
[five48k] concurrency=5 prompt~48000tok max_out=400tok
  start: running=0 waiting=0 kv=12.3% preempt=0
  end: running=0 kv=48.0% preempt=0
  preemption delta: 0 (0 = no KV swapping)
  KV pool peak occupancy: 50.8%
  ok 5/5: mean 6.3 tok/s | min 5.9 | max 6.8 | median 6.3
    req0: 6.4 tok/s (400 tok / 62.1s)
  [RESULT:five48k] preempt_delta=0 mean=6.3 peak_kv=50.8%
```

- The **preemption delta** is the headline number. Zero means the KV pool was never oversubscribed
  and the point is a clean measurement. Anything above zero means the scheduler evicted and
  recomputed a request, and the speed figures from that run describe the eviction, not the engine.
- **Pool peak occupancy** comes from a sampler thread that reads the metrics endpoint every 2
  seconds while the requests are in flight, so it is a floor on the true peak, not a ceiling.
- The **per-request speed is client wall-clock**: completion tokens divided by the whole request,
  including prompt ingestion, network time and JSON parsing. It is deliberately not the same
  quantity as the server-reported decode rates in [DESIGN.md](../DESIGN.md), and the two must not
  be mixed. Time to first token is not measured at all - the field exists in the record but is
  always empty.
- The final `[RESULT:...]` line is the same numbers in one line, meant to be copied into a log.
  **The script writes nothing to disk**; stdout is the only output.

A failure inside a request is printed as `reqN: FAIL <error>` with the message truncated, and the
remaining requests still run to completion. A request that takes longer than 600 seconds is cut
off. Concurrency is a single burst, not sustained load, so it says nothing about behaviour after
minutes of pressure.

---

## `chat.html` (76 lines, no dependencies)

A single file browser page for a smoke test: it lists the models from `http://localhost:8000/v1/models`
on load, sends whatever you type to `http://localhost:8000/v1/chat/completions` at temperature
0.6, `top_p` 0.95 and `max_tokens` 500, and prints `choices[0].message.content`, falling back to
the raw JSON when there is no normal reply. Useful for confirming a server answers before blaming
your own client.

Two assumptions are baked in: the server is on **port 8000 on localhost**, and it accepts
cross-origin browser requests. Opened straight from the filesystem, the page's origin is not a
normal site origin, and the fetch may be refused by the browser - if you see a fetch error rather
than an answer, that is the first thing to check, before the server. Moving the page to the same
origin as the API, or allowing the page's origin in the server's CORS settings, is the usual fix;
which default this build ships with was not verified here. There is no build step, no framework
and no state beyond the textarea.

---

## `start_cg_on.sh` (14 lines, bash)

A one-off reference script, not a tool you run as part of a workflow. It exports the two
environment variables the kernel build needs, prepends the CUDA toolkit and the user's local
binary directory to `PATH`, and starts the server with CUDA graphs on - the flag is absent, which
is the whole point, since `--enforce-eager` is what turns them off. It is the minimum command
behind the "CUDA graphs are the biggest lever" result.

```bash
export FLASHINFER_CUDA_ARCH_LIST='12.0'
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=/usr/local/cuda-13.2/bin:$HOME/.local/bin:$PATH
vllm serve ~/models/<model-dir> \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 100000 --max-num-seqs 12 \
  --kv-cache-dtype int4_per_token_head \
  --gpu-memory-utilization 0.94 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --served-model-name <model-dir> \
  --reasoning-parser qwen3 \
  --override-generation-config '{"temperature": 0.8}'
```

Read it as a record of one configuration and not as a recommendation: the context and sequence
values in it are the aggressive ones used for a capacity probe, the host is every interface rather
than loopback, and the model path assumes a copy on the Linux filesystem. The launcher in this
directory produces equivalent commands with the values you choose, and additionally sets the same
two environment variables for the child process - but it does **not** prepend anything to `PATH`.
If you start it from a shell whose `PATH` puts a CUDA toolkit older than 12.9 first, the kernel
build fails with `No supported CUDA architectures found`, which is a `PATH` problem, not a driver
problem.

---

## `saved_params.json` (the launcher's parameter memory)

A JSON object keyed by **model directory name**, written by `select_and_run.py` every time a start
is attempted, and read on the next run for that model. It is a convenience with a sharp edge: it
overrides the built-in defaults, and the built-in defaults are where the measured advice lives.

It currently remembers two models, under the keys `Qwen3.8-27B-NVFP4` and
`Qwen3.6-35B-A3B-NVFP4`, and the values in it are the ones a capacity-probing session settled on,
not the recommended defaults:

| Parameter | `Qwen3.8-27B-NVFP4` | `Qwen3.6-35B-A3B-NVFP4` | The built-in default |
|---|---|---|---|
| `max_model_len` | 100000 | 150000 | 65536 |
| `max_num_seqs` | 12 | 12 | 32 |
| `kv_cache_dtype` | `int4_per_token_head` | `fp8` | `fp8` |
| `gpu_memory_utilization` | 0.93 | 0.94 | 0.92 |
| `temperature` | 0.8 | 0.6 | 1.0 |
| `top_p` / `top_k` | 0.95 / 20 | 0.95 / 20 | 0.95 / 20 |
| `port` | 8000 | 8000 | 8000 |
| `enforce_eager` | `n` | `n` | `n` |
| `reasoning_parser` | `qwen3` | `qwen3` | `qwen3` |
| `mtp` | `off` | `off` | `off` |

Consequences a reader should take seriously:

- On a fresh copy, launching the dense model starts from **100000 context, 12 sequences, 4-bit KV
  and 0.93 utilisation** - not from the 65536 / 32 / fp8 / 0.92 defaults. If you want the
  conservative values, enter advanced mode and set them, or delete the entry.
- **`enforce_eager` is the value to check.** This file previously carried `y` for the dense entry,
  which adds `--enforce-eager` and silently disables CUDA graphs; measured, that costs 13.9
  tokens/s where 48.2 was available. It was corrected to `n`, and the file as published shows
  `n`. If you have a copy with `y`, or you set choice 2 in the CUDA graph prompt once, every later
  launch for that model will keep using it, and the slowness will look like a hardware problem.
- The file is written before the server starts, so a failed start still updates it. That is
  intended for limit probing and annoying otherwise: a mistyped context length becomes the new
  default for that model.
- Models with no entry here simply use the built-in defaults. The launcher's defaults in code and
  the values in this file are two different things, and it is worth reading both before quoting
  either.

---

## `logs/` and `tools/`: not part of the published tree

Neither directory ships with this repository, and `.gitignore` excludes both. What each one holds,
and why it stays out:

- **`logs/`** - the engines' own capture files. The first of them is the start-up sweep behind the
  KV cache dtype table above: one block per candidate, each with a pass or fail line and, where the
  engine returned anything, its answer. It is evidence rather than prose, and that is exactly why it
  is quoted rather than shipped: its lines carry machine-local paths - the Linux home directory, the
  Windows-side model root, the temporary directories of the process that supervised it - and its
  labels are not in English. Those are the strings a public copy must not contain. The sweep's
  numeric content is reproduced, with its source, in
  [DESIGN.md](../DESIGN.md), which is also where its per-attempt outcome is read from.
- **`tools/`** - the pre-publication gate this copy was checked with: local paths and usernames,
  private platform names, credential-shaped strings, CJK characters and emoji, and file size. It is
  not published either, because its pattern table lists the private names it screens for; it stays in
  the working copy and is run before each push.

The per-run logs the launcher writes are excluded for the same reason, and they are useful locally:
the launcher writes one on every failed start and prints its path, and that file's tail usually holds
the root cause, which is why it is written at all. One of them is megabytes of compiler output from a
first-run kernel build, and per-boot state that means nothing on another machine. To publish one, scrub the paths, the user names and any host-specific lines first,
then re-run the same gate against the result.
