# Quick start

This is the linear path: one consumer Blackwell card, Windows 11 on the host, vLLM inside WSL2,
a model on the Linux filesystem, and one benchmark request at the end. Commands are meant to be
copied. Placeholders in angle brackets are the only things you have to fill in.

It is written from the configuration this repository measured - vLLM 0.28.0, FlashInfer
0.6.16.post3, torch 2.13.0+cu130, CUDA toolkit 13.2, Ubuntu 24.04 under WSL2 - and it says where
that record is incomplete instead of pretending otherwise. See the note at the end before you
expect a byte-exact rebuild.

## What you need

| Item | What was used here |
|---|---|
| GPU | RTX 5090, 32 GB, compute capability 12.0 (sm_120, consumer Blackwell). A smaller card runs, but the context and sequence figures in this repository do not transfer. |
| NVIDIA driver | A recent Windows driver for this generation. The machine ran a 610-series driver; the exact build is not recorded here, and the driver must be a Windows-side install - do not install a separate Linux GPU driver inside WSL. |
| Host | Windows 11, 32 GiB of memory, fast disk. A model copy needs 20-25 GB, and the host memory matters because one of the engines pins several GiB of it. |
| WSL2 | Ubuntu 24.04. vLLM does not support a native Windows install, so the Linux environment is the requirement, not a preference. |
| CUDA toolkit | 13.2 inside WSL, at the path below. The kernels are compiled on first run and need a toolkit at 12.9 or newer for this compute capability. |
| Python | A Python 3 environment inside WSL. The measured setup used the system interpreter; a virtual environment is the cleaner choice and was not what was measured (see the end). |

Everything is installed inside the Linux environment unless it is explicitly a Windows-side step.

## 1. Confirm the GPU is visible inside WSL

```bash
wsl -l -v                      # from the Windows side: which distribution is running
nvidia-smi                     # inside WSL: the card and the driver version
nvcc --version                 # inside WSL: which CUDA toolkit is first on PATH
```

If `nvidia-smi` works inside WSL, the Windows driver is wired through correctly and nothing else
is needed on the graphics side.

## 2. Pin WSL networking to NAT

Do this before chasing a connection problem. With WSL's mirrored networking mode, a failed
initialisation can leave the environment with only loopback: the server answers inside WSL and the
Windows side cannot reach it at all. NAT mode has no such failure and costs nothing for serving.

Create or edit `.wslconfig` in the Windows user home directory:

```ini
[wsl2]
networkingMode=NAT
```

```bash
wsl --shutdown        # from the Windows side, then start the distribution again
```

## 3. Install vLLM inside WSL

```bash
# inside WSL
python3 -m venv ~/venvs/vllm && source ~/venvs/vllm/bin/activate   # optional, see the end
pip install --upgrade pip
pip install 'vllm==0.28.0'
```

If you are installing into the system interpreter rather than a virtual environment, a modern
distribution refuses the install with `externally-managed-environment` and you have to decide for
yourself whether to override it:

```bash
pip install --break-system-packages 'vllm==0.28.0'
```

Check the two packages the attention kernels come from, since a mismatch is the most common
early failure:

```bash
python3 -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
```

## 4. Get the weights

The measurements used a pre-quantised NVFP4 release of a 27B-class model. Two traps here:

- **A library filter on the model hub can hide the release you want.** Filtering by a serving
  library returns third-party quantisations and may not list the one that is actually supported.
  Query the repository directly, and treat the engine project's own recipe list as the
  compatibility statement.
- **Download through a mirror when the direct route is slow**, and treat the mirror as a download
  channel only: the licence terms are the original repository's.

```bash
pip install modelscope
python3 -c "from modelscope.cli.cli import run_cmd; run_cmd()" download \
  --model <org>/<model> --local_dir ~/models/<model-dir>
```

The quoted command is deliberate: on the machine measured here the `modelscope` console script was
broken (it exited with status 1 and printed nothing) and the package has no `__main__`, so calling
the entry point function directly was the only route that worked.

## 5. Put the weights on the Linux filesystem

Reading weights across the Windows mount is roughly an order of magnitude slower: minutes of disk
wait on every start, against seconds once they are on the Linux side. Copy them once.

```bash
mkdir -p ~/models
rsync -a --info=progress2 /mnt/<windows-drive>/<model-dir>/ ~/models/<model-dir>/
```

The model directory must contain `config.json`. It is also what the launcher in `scripts/` looks
for when it decides whether a directory is a usable model.

## 6. Start the server and verify it

Three environment variables decide whether the kernels build for this compute capability. The
first tells the kernel build which architecture to target; the second and third give it a toolkit
new enough for it.

```bash
export FLASHINFER_CUDA_ARCH_LIST='12.0'
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=/usr/local/cuda-13.2/bin:$HOME/.local/bin:$PATH
```

```bash
vllm serve ~/models/Qwen3.8-27B-NVFP4 \
  --served-model-name Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 --max-num-seqs 32 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

For a deeper single context at the cost of KV precision, swap in
`--kv-cache-dtype int4_per_token_head --max-model-len 114688 --gpu-memory-utilization 0.94`.
That is the configuration the three-engine comparison used, and it is a capacity trade - the
long-context output quality of the 4-bit option was not measured.

Note that `max_model_len` must cover the deepest request you intend to send plus the output
length. If the pool cannot hold it, vLLM fails fast at start-up with `KV cache needed > available`
rather than degrading quietly, which is the behaviour you want.

**The first start is slow, and it is not hung.** Compiling the kernels can take up to about twenty
minutes with several compiler processes running in parallel; check the process list and GPU memory
before concluding anything is dead. Later starts take tens of seconds, because the kernels are
cached.

Verify both endpoints - the API and the metrics the tools in this repository read:

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/metrics | grep -E 'kv_cache_usage_perc|num_preemptions_total'
```

If you drive WSL from the Windows side, keep the command in a script file inside WSL or pass it as
one escaped string: quoting that survives two shells is the one thing that reliably goes wrong.

## 7. The second engine: NInfer

NInfer is a separate engine built from its own source tree in the same Linux environment, with its
own artifact format and its own endpoint (port 8080). Its start-up parameters mean something
different from vLLM's, and knowing which one allocates memory is the whole game:

| Parameter | Meaning |
|---|---|
| `--kv-capacity` | The size of the KV pool, allocated once at start-up. **This is what occupies device memory.** |
| `--max-context` | The per-request logical context ceiling. It is a limit, not an allocation: raising it alone costs nothing. |
| `--max-concurrency` | Requests in flight at once, which must fit inside the pool above. |
| `--kv-dtype` | KV cache storage format. |
| `--spec` / `--draft-tokens` | Speculative decoding and its draft depth. |
| `--host-kv-mib` | Host memory pinned for KV state. It is pinned, so it does not swap - budget for it and do not run large memory jobs during a measurement. |
| `--no-thinking` | Turns the model's thinking mode off at **start-up**, where the other two engines take it as a request-level directive. That asymmetry is a real limit of the three-engine comparison, not a detail. |
| `--model-id` | Overrides the model id served, which otherwise comes from the artifact's own metadata. |

The shape of the invocation, with the values the comparison used:

```bash
pkill -f ninfer-serve        # one engine instance per artifact: stop the old one first
<engine-launcher> --yes --model 27b \
  --kv-dtype fp8 --max-context 196608 --kv-capacity 196608 \
  --max-concurrency 1 --spec mtp --draft-tokens 3 --no-thinking
```

The exact launcher and the spelling of some flags come from that build's source tree and are not
in this repository, so treat the command above as the parameter set rather than a literal line.

Two failure modes worth recognising:

- **`kv_capacity is outside the usable range for max_context and max_concurrency`.** The pool was
  set larger than the per-request limit allows, or the other way round. Make the two consistent.
- **Speculative decoding that is silently off.** If the response body has no draft counters and
  the per-request log line has no acceptance figures, it was not enabled, whatever the flag said.

## 8. The third engine: LM Studio, on the Windows side

LM Studio is a desktop application: it runs on Windows, loads a GGUF model, and exposes an
OpenAI-compatible endpoint on its own port. The command line interface ships with it.

```text
lms server start --port 1234
lms load qwen/qwen3.8-27b --context-length 201000 --gpu max
```

Equivalent to setting Context Length and GPU Offload to the maximum in the application. Its
per-model settings matter and are not in the request: K/V cache quantisation and whether the draft
head is enabled live in the model configuration, not on the command line, and changing the
quantisation requires restarting the server. Record that configuration alongside any number you
take from this engine.

**Its numbers can only come from its server log.** LM Studio has no timings API, so the only
record of how fast a request ran is the timing lines in the log - which is also the weakest source
of the three engines, and the reason the extraction was scripted instead of read off the screen.
Two rules if you do this:

- Write requests to `http://127.0.0.1:1234/v1/chat/completions` like any other OpenAI-compatible
  endpoint, with the model id `lms ps` reports.
- Extract **numeric fields only** from the log. Do not copy log lines into a report or an issue:
  they can contain local paths, addresses and other text that has nothing to do with performance.
  Restore the logging switch that produces the timing lines to its previous value afterwards.

```text
lms unload --all
lms server stop
```

## 9. One benchmark request, and how to read the answer

With a vLLM server up on port 8000, the pressure script in this repository sends concurrent
requests and reports what the server's own counters did while they ran:

```bash
python3 scripts/kv_bench.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 5 --prompt-tokens 48000 --max-tokens 400 --name five48k
```

What comes back, and what each line means:

```
[five48k] concurrency=5 prompt~48000tok max_out=400tok
  start: running=0 waiting=0 kv=12.3% preempt=0
  end: running=0 kv=48.0% preempt=0
  preemption delta: 0 (0 = no KV swapping)
  KV pool peak occupancy: 50.8%
  ok 5/5: mean 6.3 tok/s | min 5.9 | max 6.8 | median 6.3
  [RESULT:five48k] preempt_delta=0 mean=6.3 peak_kv=50.8%
```

- **`preemption delta` is the number to look at first.** Zero means the pool was never
  oversubscribed. Above zero means the scheduler evicted and recomputed a request, and every speed
  figure from that run describes the eviction rather than the engine.
- **`KV pool peak occupancy`** is sampled from the server every 2 seconds while the requests are in
  flight, so it is a floor on the real peak. When it approaches the ceiling, expect the next
  deeper or more concurrent run to preempt.
- **The per-request speeds are client wall-clock**, completion tokens divided by the whole request,
  prompt ingestion included. They are not the same quantity as the server-reported decode rates in
  [DESIGN.md](../DESIGN.md), and the two must never be quoted side by side.
- **`--prompt-tokens` is a filler-size knob, not a token count.** It sizes the prompt in
  characters, and English prose runs at roughly four characters per token, so asking for 48000
  lands near 12000 prompt tokens. Read the real depth out of the server's own response.

Two things this script will not do: it only works against vLLM, because it reads counters that
only vLLM publishes at `/metrics` - point it at another engine and it fails before sending a
request - and it is a single burst, not a sustained load test. [TOOLS.md](TOOLS.md) has the full
flag list and the estimator's assumptions.

To eyeball the server instead of measuring it, open `scripts/chat.html` in a browser while the
server is on port 8000: it lists the models and sends one request. If the page reports a fetch
error, suspect the browser blocking a cross-origin request from a filesystem page before suspecting
the server.

## 10. Running more than one engine

Do not. Each of the three configurations needs roughly 28-31 GB of the 32 GB card, so they were
started, measured and stopped one after another. Before switching engines, confirm the previous
one is gone - its port stopped answering, and for the WSL-side engines no server process left
running.

## What this quick start does not give you

Being explicit, because "reproducible" is easy to claim and hard to earn:

- **The virtual environment is not recorded as measured.** vLLM 0.28.0 and the FlashInfer and
  torch versions above were installed with `--break-system-packages` into the system interpreter,
  after an earlier source install was removed. The virtual environment in step 3 is the cleaner
  equivalent, and it is a recommendation rather than something that was verified the same way.
- **The exact NVIDIA driver build is not recorded.** Only the series is known.
- **How the CUDA 13.2 toolkit was installed is not recorded**, only that it is at the path used
  above and that the toolkit first on `PATH` was an older one - which is why these variables are
  exported explicitly in every start-up recipe.
- **The hardware list is one machine**, and the boot-to-boot spread measured on it was about 10%,
  with pooled KV figures moving about 7% between runs. Differences smaller than that are noise.
- **The engines drift.** These steps describe one build set in September 2026; a renamed counter,
  a changed default or a replaced kernel breaks the record while leaving every command above
  untouched. If your numbers disagree, that is worth reporting with the versions and flags.
