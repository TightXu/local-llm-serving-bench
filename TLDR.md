# local-llm-serving-bench: the 30-second version

A measurement record for local LLM serving, not a benchmark and not software. It holds what vLLM, NInfer and LM Studio returned for a 27B-class model on one RTX 5090 (32 GB), measured in September 2026, plus the launcher and pressure scripts behind the numbers.

## What it shows

- NInfer was the fastest engine at every depth: 262.5 tokens/s at the shallowest point, 230.9 at the deepest, 3.9x to 5.1x vLLM without speculative decoding.
- CUDA graphs were the largest single switch measured: 13.9 to 48.2 tokens/s single stream at short context (+247%), and +40% under one long-context load.
- Speculative decoding on vLLM turns into a loss at depth: +125% / +96% / +2% / -34% across the four points.
- The one collapse to 0.5 tokens/s was about 97% pool occupancy with preemptions, not the GPU.

## Where to start

- README, section `## Start here`.
- [DESIGN.md](DESIGN.md) for the full report and the switch matrix; [docs/QUICKSTART.md](docs/QUICKSTART.md) to rebuild the setup.

## What it does not claim

- One machine and one build set only, and the engines held different weight formats (NVFP4 against GGUF), so part of the gap is weight format, not engine.
- Output quality was not measured except where stated.
