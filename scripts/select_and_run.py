# -*- coding: utf-8 -*-
r"""select_and_run.py — interactive vLLM model launcher (RTX 5090 / WSL2)
Called by start_vllm.bat (Windows) or start_vllm.sh (WSL) → this script

- Press Enter all the way = one-key start of the default model (a setup verified on this machine)
- In advanced mode, typing /? prints the help text of the current parameters
- **Parameters are saved as soon as a start is attempted (kept on success and on failure)** and become
  that model's defaults next time — when probing KV / context limits, changing one value and rerunning is enough
- On a failed start the full log is written under logs/ next to this script
"""
import os
import sys
import json
import subprocess
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WIN_MODEL_ROOT = os.environ.get("VLLM_MODEL_WIN_ROOT", "")  # Windows-partition model root (WSL view); set the env var
EXT4_MODEL_ROOT = os.path.expanduser("~/models")  # WSL ext4 copy root (preferred)
HOST = "0.0.0.0"
PORT = 8000
DEFAULT_MODEL_HINT = "Qwen3.8"                    # one-key default: model name containing this keyword
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
PARAMS_FILE = os.path.join(SCRIPT_DIR, "saved_params.json")

# Verified recommended defaults (32 GB card + 27B NVFP4)
DEFAULTS = {
    "max_model_len": "65536",
    "max_num_seqs": "32",
    "kv_cache_dtype": "fp8",
    "gpu_memory_utilization": "0.92",
    "temperature": "1.0",        # officially recommended for Qwen3.8 thinking mode
    "top_p": "0.95",
    "top_k": "20",
    "port": str(PORT),
    "enforce_eager": "n",
    "reasoning_parser": "qwen3",
    "mtp": "off",
}

# Advanced-mode parameter definitions: key / label / help text / choices (None = free text)
# The help text is only printed when the user types /?
PARAM_DEFS = [
    {"key": "max_model_len", "label": "context length max-model-len",
     "help": "Maximum token count for one request. On a 32 GB card running 27B NVFP4 the estimated ceiling is about 97K; "
             "set it too high and startup reports 'KV cache needed > available'. Recommended: 65536"},
    {"key": "max_num_seqs", "label": "concurrent sequences max-num-seqs",
     "help": "Upper bound on sequences processed at the same time. This model has a hybrid architecture (linear attention), so every sequence holds a "
             "Mamba state and the ceiling is about 82. Recommended: 32"},
    {"key": "kv_cache_dtype", "label": "KV quantization kv-cache-dtype",
     "help": "Measured (Qwen3.8-27B, 65536 ctx, pool ~3.2 GiB): auto = 86K tok baseline; "
             "fp8 = 93K safe (small gain on hybrid architectures, the 48 GDN layers cannot be quantized); "
             "fp8_per_token_head = 102K per-token per-head scale, better precision at the same capacity; "
             "int8_per_token_head = 102K, the int8 version of the above; "
             "int4_per_token_head = 176K, 2x capacity, highest precision risk; "
             "turboquant_k8v4 = 133K, K 8-bit + V 4-bit, new backend, long-context behaviour not verified; "
             "fp8_e5m2 / nvfp4 are not available on this model (crash at startup)",
     "choices": ["fp8", "auto", "fp8_per_token_head", "int8_per_token_head",
                 "int4_per_token_head", "turboquant_k8v4", "nvfp4", "fp8_e5m2"]},
    {"key": "gpu_memory_utilization", "label": "VRAM utilization gpu-memory-utilization",
     "help": "Fraction of VRAM vLLM may use. Leave headroom for the system and the desktop; 0.92 is recommended"},
    {"key": "temperature", "label": "temperature",
     "help": "Sampling temperature. 1.0 is officially recommended for Qwen3.8 thinking mode; only passed to vLLM when it differs from the default"},
    {"key": "top_p", "label": "top_p",
     "help": "Cumulative probability cutoff for nucleus sampling. Officially recommended: 0.95"},
    {"key": "top_k", "label": "top_k",
     "help": "Sample every step from only the k most likely tokens. Officially recommended: 20"},
    {"key": "port", "label": "port",
     "help": "API service port; the OpenAI-compatible endpoint is http://localhost:<port>/v1"},
    {"key": "enforce_eager", "label": "CUDA graph mode",
     "help": "Choice 1 = CUDA graphs on (default, strongly recommended): measured +247% single-stream speed at short context "
             "(13.9→48.2 tok/s), +40% at long context; costs an extra 0.15~0.51 GiB of VRAM.\n"
             "Choice 2 = CUDA graphs off / --enforce-eager: only for when VRAM is very low; "
             "it is significantly slower. 2026-09-05 stress-test conclusion: choice 1 is mandatory for normal use",
     "choices": ["n", "y"],
     "choice_names": ["CUDA graphs on (fast, default)", "CUDA graphs off / enforce-eager (saves VRAM)"]},
    {"key": "reasoning_parser", "label": "reasoning-parser",
     "help": "How the thinking trace is parsed. qwen3 = split the <think> content into the response's reasoning_content field; "
             "(none) = do not split, keep the raw thinking text in the output",
     "choices": ["qwen3", ""], "choice_names": ["qwen3", "(none)"]},
    {"key": "mtp", "label": "MTP speculative decoding",
     "help": "Multi-token prediction speedup: draft and verify several tokens in one forward pass; measured +33~88% on single-stream decode "
             "(the same mechanism LM Studio uses).\n"
             "[WARN] only checkpoints that carry the MTP head (nextn layer) can use it: Qwen3.8 [OK] / Qwen3.6 [FAIL] (no such head, the option is ignored);\n"
             "quality status as measured: fp8 KV [OK] / turboquant [FAIL] / int4_per_token_head [WARN];\n"
             "[WARN] on sm120 MTP pushes the fused GDN kernel back onto the slow Triton path, so the real gain needs measuring;\n"
             "check: in /metrics a spec_decode_num_accepted / draft_tokens ratio >0.5 is where it actually pays off",
     "choices": ["off", "on"], "choice_names": ["off (default)", "on (num_spec=3)"]},
]

# Model-name prefixes that carry an MTP head (checkpoint with a nextn layer)
MTP_MODELS = ("Qwen3.8",)
MTP_SPEC_CONFIG = {"method": "mtp", "num_speculative_tokens": 3}
MTP_MEM_GIB = 0.4  # extra VRAM held by the MTP head weights (about one layer); used by the estimator

# Environment-variable fixes required for NVFP4 on sm120 (missing either one crashes)
ENV_FIX = {
    "FLASHINFER_CUDA_ARCH_LIST": "12.0",
    "CUDA_HOME": "/usr/local/cuda-13.2",
}


# ======================== parameter persistence ========================
def load_saved(model_name):
    """Read the parameters saved for this model last time and overlay them on the defaults."""
    cfg = dict(DEFAULTS)
    try:
        with open(PARAMS_FILE, "r", encoding="utf-8") as f:
            all_saved = json.load(f)
        saved = all_saved.get(model_name, {})
        for k, v in saved.items():
            cfg[k] = str(v)
    except (OSError, ValueError):
        pass
    return cfg


def save_params(model_name, cfg):
    """Save the current parameters as this model's defaults (called once the server has started)."""
    data = {}
    try:
        with open(PARAMS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        pass
    data[model_name] = {d["key"]: cfg.get(d["key"], DEFAULTS[d["key"]]) for d in PARAM_DEFS}
    try:
        with open(PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("[parameters saved] -> " + PARAMS_FILE)
    except OSError as e:
        print("[WARN] could not save parameters: " + str(e))


def save_log(lines, cmd, exit_code):
    """On a failed start, save the full log under logs/ and return the log path."""
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, "vllm_fail_" + stamp + ".log")
    try:
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write("$ " + " ".join(cmd) + "\n")
            f.write("[exit_code=" + str(exit_code) + "]\n\n")
            f.writelines(lines)
    except OSError as e:
        print("[WARN] could not save the log: " + str(e))
        return path
    return path


# ======================== interactive input ========================
def ask_free(prompt, default, help_text):
    """Free text: Enter = default, /? = show help."""
    while True:
        tip = default if default != "" else "none"
        raw = input(prompt + " [default=" + tip + "]: ").strip()
        if raw == "/?":
            print("    " + help_text)
            continue
        return raw if raw else default


def ask_choice(prompt, default, options, names, help_text):
    """Multiple choice: numbered menu, Enter = default, /? = show help. The default shows its friendly name, not the internal value."""
    def display(val):
        if names and val in options:
            return names[options.index(val)]
        return val if val else "none"
    while True:
        print(prompt + " [default=" + display(default) + "]  (/? = help)")
        for i, val in enumerate(options, 1):
            mark = "  *default" if val == default else ""
            name = names[i - 1] if names else val
            print("  " + str(i) + ". " + (name if name else val) + mark)
        raw = input("choice [default=" + display(default) + "]: ").strip()
        if raw == "/?":
            print("    " + help_text)
            continue
        if not raw:
            return default
        try:
            idx = int(raw)
        except ValueError:
            print("please enter a number")
            continue
        if 1 <= idx <= len(options):
            return options[idx - 1]
        print("out of range, please enter 1-" + str(len(options)))


def configure_advanced(cfg):
    print("\n--- advanced parameters (Enter = value in brackets, /? = help) ---")
    for d in PARAM_DEFS:
        key = d["key"]
        if "choices" in d:
            cfg[key] = ask_choice(d["label"], cfg[key], d["choices"],
                                  d.get("choice_names"), d["help"])
        else:
            cfg[key] = ask_free(d["label"], cfg[key], d["help"])
    return cfg


# ======================== model selection ========================
def list_models(root):
    if not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and not d.endswith("_disabled"):
            out.append(d)
    return out


def resolve_model_dir(name):
    """Prefer the ext4 copy, fall back to the Windows partition. Returns (path, source tag)."""
    p_ext4 = os.path.join(EXT4_MODEL_ROOT, name)
    if os.path.isfile(os.path.join(p_ext4, "config.json")):
        return p_ext4, "ext4"
    p_win = os.path.join(WIN_MODEL_ROOT, name)
    if os.path.isfile(os.path.join(p_win, "config.json")):
        return p_win, "win"
    return None, None


def pick_model():
    names = list_models(WIN_MODEL_ROOT)
    for e in list_models(EXT4_MODEL_ROOT):
        if e not in names:
            names.append(e)
    if not names:
        print("[ERROR] no models found in either directory: " + WIN_MODEL_ROOT + " / " + EXT4_MODEL_ROOT)
        sys.exit(1)

    default_idx = 0
    print("\navailable models (* = default, Enter selects it): ")
    for i, name in enumerate(names, 1):
        mark = " *" if DEFAULT_MODEL_HINT in name else ""
        if mark:
            default_idx = i
        print("  " + str(i) + ". " + name + mark)

    while True:
        raw = input("\nselect a model [default=" + str(default_idx) + "]: ").strip()
        if not raw:
            idx = default_idx
        else:
            try:
                idx = int(raw)
            except ValueError:
                print("please enter a valid number")
                continue
        if 1 <= idx <= len(names):
            return names[idx - 1]
        print("out of range, please enter 1-" + str(len(names)))


# ======================== startup ========================
def build_command(model_path, cfg, model_name=None):
    cmd = [
        "vllm", "serve", model_path,
        "--host", HOST,
        "--port", cfg["port"],
        "--max-model-len", cfg["max_model_len"],
        "--max-num-seqs", cfg["max_num_seqs"],
        "--kv-cache-dtype", cfg["kv_cache_dtype"],
        "--gpu-memory-utilization", cfg["gpu_memory_utilization"],
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]
    # Expose a short API name (e.g. Qwen3.8-27B-NVFP4); otherwise the whole path becomes the model name
    if model_name:
        cmd += ["--served-model-name", model_name]
    if cfg.get("reasoning_parser"):
        cmd += ["--reasoning-parser", cfg["reasoning_parser"]]
    if cfg.get("enforce_eager") == "y":
        cmd += ["--enforce-eager"]
    # MTP speculative decoding: only takes effect on models with an MTP head (Qwen3.8 has the nextn layer, Qwen3.6 does not)
    if cfg.get("mtp") == "on" and model_name and model_name.startswith(MTP_MODELS):
        cmd += ["--speculative-config", json.dumps(MTP_SPEC_CONFIG)]
    # Sampling parameters: passed only when they differ from the official defaults, via --override-generation-config
    gen = {}
    if cfg["temperature"] != DEFAULTS["temperature"]:
        gen["temperature"] = float(cfg["temperature"])
    if cfg["top_p"] != DEFAULTS["top_p"]:
        gen["top_p"] = float(cfg["top_p"])
    if cfg["top_k"] != DEFAULTS["top_k"]:
        gen["top_k"] = int(cfg["top_k"])
    if gen:
        cmd += ["--override-generation-config", json.dumps(gen)]
    return cmd


def run_vllm(cmd, on_started=None):
    env = dict(os.environ)
    env.update(ENV_FIX)
    print("\nlaunch command:\n  " + " ".join(cmd) + "\n")
    print("=" * 60)

    buf = []
    started_saved = False
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env)
    except OSError as e:
        print("[ERROR] could not start vllm: " + str(e))
        save_log(["[OSError] " + str(e)], cmd, -1)
        return

    try:
        for line in proc.stdout:
            buf.append(line)
            print(line, end="")
            # Success marker: save the parameters (only the first time it is seen)
            if not started_saved and ("Application startup complete" in line
                                      or "Uvicorn running on" in line):
                started_saved = True
                if on_started:
                    on_started()
        proc.wait()
    except KeyboardInterrupt:
        print("\ninterrupted by the user, shutting the server down...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        print("[interrupted]")
        return

    code = proc.returncode
    if code != 0:
        log_path = save_log(buf, cmd, code)
        print("\n[vLLM failed to start, code=" + str(code) + "]")
        print("[full log] " + log_path)
        # Print the most likely root-cause line directly
        root_causes = [l.rstrip() for l in buf
                       if ("ValueError:" in l or "RuntimeError:" in l
                           or "OSError:" in l)]
        if root_causes:
            print("\n[root-cause hint] " + root_causes[-1].strip())
    else:
        print("\n[vLLM exited normally, code=0]")


# ======================== VRAM estimator ========================
# Basis: the vllm 0.28 memory-ledger source (gpu_worker.py determine_available_memory) +
# 3 closed-loop runs on this machine with VLLM_LOGGING_LEVEL=DEBUG (error <=0.1 GiB) + 6 kv_test logs.
# Formula: KV pool = util*total VRAM - weights(21.34) - peak activations(1.92, follows the batch budget, weakly correlated with ctx/seq)
#             - non-torch overhead (~1.9, CUDA context/NCCL/workspace) - CUDA graphs (eager=0, otherwise 0.1~0.7 depending on the backend)
# The per-token cost comes from 6 measurements (hybrid architecture: 16 full-attention layers that can be quantized + 48 GDN layers whose state stays bf16/fp32):
#   auto=37948 fp8=37703 fp8_pt=38979 int8_pt=38873 int4_pt=22020 tq=30240 B/token
MEM_WEIGHTS_GIB = 21.34
MEM_ACT_GIB = 1.92
MEM_NONTORCH_GIB = 1.9
# CUDA-graph VRAM varies with the attention backend (measured in kv_test): flashinfer family 0.44-0.51, triton per-token family 0.11-0.18
CG_EST_BY_DTYPE = {
    "auto": 0.47, "fp8": 0.51, "fp8_per_token_head": 0.15,
    "int8_per_token_head": 0.14, "int4_per_token_head": 0.18,
    "turboquant_k8v4": 0.11, "nvfp4": 0.5, "fp8_e5m2": 0.5,
}
BYTES_PER_TOKEN = {
    "auto": 37948, "fp8": 37703, "fp8_per_token_head": 38979,
    "int8_per_token_head": 38873, "int4_per_token_head": 22020,
    "turboquant_k8v4": 30240, "nvfp4": 19000, "fp8_e5m2": 37703,
}
GPU_TOTAL_GIB = 31.82
ERR_BAND = 0.12   # error band from non-torch overhead / CUDA-graph variation, +-12%

# Measured weights per model (GiB). 27B = 21.34 (6 kv_test runs), 35B-A3B = 23.27 (2026-09-05 startup log)
WEIGHTS_BY_MODEL = {
    "Qwen3.8-27B": 21.34,
    "Qwen3.6-27B": 21.34,
    "Qwen3.6-35B": 23.27,   # MoE, larger weights, so util has to come down accordingly (<=0.90 recommended)
}


def get_weights_gib(model_name):
    """Match a weight constant by model-name prefix; unmatched models fall back to the 27B value 21.34."""
    for prefix, w in WEIGHTS_BY_MODEL.items():
        if model_name.startswith(prefix):
            return w
    return 21.34


def get_gpu_mem_info():
    """Read from nvidia-smi (total VRAM GiB, currently free GiB). On failure use (constant, constant-1.6)."""
    fallback_free = GPU_TOTAL_GIB - 1.62   # background usage measured on the kv_test day
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        total_mib, free_mib = out[0].split(",")
        return int(total_mib.strip()) / 1024.0, int(free_mib.strip()) / 1024.0
    except Exception:
        return GPU_TOTAL_GIB, fallback_free


def estimate_vram(cfg, weights_gib=None):
    """Return (KV pool GiB low, high, tokens low, high, total usage GiB, free GiB, reject-risk bool).
    When the model is not on ext4 the weights come over 9p and still occupy VRAM, they just load slower; the ledger is unaffected."""
    dtype = cfg["kv_cache_dtype"]
    util = float(cfg["gpu_memory_utilization"])
    total, free_now = get_gpu_mem_info()
    eager = cfg.get("enforce_eager") == "y"
    cg = 0.0 if eager else CG_EST_BY_DTYPE.get(dtype, 0.4)
    bpt = BYTES_PER_TOKEN.get(dtype, 38000)
    w = weights_gib if weights_gib else MEM_WEIGHTS_GIB
    # The MTP head holds one extra set of weights (~1 layer) plus the drafting activations
    mtp_extra = MTP_MEM_GIB if cfg.get("mtp") == "on" else 0.0

    # What vLLM actually does: requested = util*total; if free < requested it errors out at startup.
    # pool = min(requested, free) - non_kv  (this is the part background usage eats into)
    requested = util * total
    will_reject = free_now < requested
    budget = min(requested, free_now)
    pool_mid = budget - w - MEM_ACT_GIB - MEM_NONTORCH_GIB - cg - mtp_extra
    pool_lo = pool_mid * (1 - ERR_BAND)
    pool_hi = pool_mid * (1 + ERR_BAND)
    if pool_mid <= 0:
        return 0, 0, 0, 0, total, free_now, will_reject
    tok_lo = int(pool_lo * 2**30 / bpt)
    tok_hi = int(pool_hi * 2**30 / bpt)
    used = w + MEM_ACT_GIB + MEM_NONTORCH_GIB + cg + mtp_extra + pool_mid
    return pool_lo, pool_hi, tok_lo, tok_hi, used, free_now, will_reject


def show_estimate_and_confirm(cfg, cmd, dry_run, model_name=""):
    """Show the VRAM estimate; the user confirms the start / a = reconfigure / q = quit."""
    weights_gib = get_weights_gib(model_name)
    while True:
        pool_lo, pool_hi, tok_lo, tok_hi, used, free_now, will_reject = estimate_vram(cfg, weights_gib)
        dtype = cfg["kv_cache_dtype"]
        shortfall = max(0.0, float(cfg["gpu_memory_utilization"]) * GPU_TOTAL_GIB - free_now)
        if tok_lo == 0:
            print("\n[estimate] settings too aggressive: the KV pool is negative and startup will certainly fail. Lower the VRAM utilization or the context length.")
        else:
            need = int(cfg["max_model_len"])
            fits = "[OK] the selected context fits" if tok_lo >= need else \
                   "[FAIL] the pool ceiling is below the selected context! Startup will report 'KV cache needed > available' or vLLM will scale it down automatically"
            mtp_on = cfg.get("mtp") == "on" and model_name.startswith(MTP_MODELS)
            print("\n" + "-" * 60)
            print(f"   VRAM estimate (calibrated on 6 measurements from this machine, error ±{int(ERR_BAND*100)}%)")
            print(f"   total VRAM    : {GPU_TOTAL_GIB:.1f} GiB")
            print(f"   free now      : {free_now:.1f} GiB  (other programs already hold {GPU_TOTAL_GIB - free_now:.1f} GiB)")
            print(f"   weights       : {weights_gib:.2f} GiB"
                  + ("  (35B MoE: util <=0.90 recommended)" if weights_gib > 22 else "")
                  + ("  + MTP head ≈0.4" if mtp_on else ""))
            if will_reject and shortfall <= 1.5:
                # Shortfall <=1 GiB: under VRAM pressure the Windows desktop / browser release memory by themselves,
                # and vLLM usually squeezes it out while loading weights, so startup usually succeeds (past experience: a 0.03 GiB shortfall did fail,
                # but shortfalls of 0.5~1 GiB succeeded several times). Warn, but do not treat it as fatal.
                print(f"   [HINT] free is slightly below the util budget (short by {shortfall:.2f} GiB): "
                      "under VRAM pressure Windows background processes usually give it up, so startup will most likely succeed;")
                print("     if startup reports 'Free memory ... is less than desired', "
                      f"lower util below {free_now / GPU_TOTAL_GIB:.2f} or close whatever is holding VRAM")
            elif will_reject:
                # Large shortfall: background processes cannot free that much; this is genuinely dangerous
                print(f"   [WARN] free is {shortfall:.1f} GiB below the util budget; the shortfall is too large and startup will most likely error out!")
                print(f"    → lower util below {free_now / GPU_TOTAL_GIB:.2f}, or close whatever is holding VRAM")
            print(f"   est. total    : ≈{used:.1f} GiB  (weights {weights_gib:.1f} + activations/system ≈3.8 + CUDA graphs {CG_EST_BY_DTYPE.get(dtype,0.4):.2f} + KV pool)")
            print(f"   KV pool       : {pool_lo:.2f} ~ {pool_hi:.2f} GiB")
            print(f"   KV capacity[{dtype}]: {tok_lo:,} ~ {tok_hi:,} tokens")
            print(f"   context needed: {need:,} tokens/request  {fits}")
            if cfg.get("mtp") == "on" and not model_name.startswith(MTP_MODELS):
                print("   [INFO] MTP is selected but the current model has no MTP head, so the option will be ignored")
            print("-" * 60)
        print("\nEnter = start / a = reconfigure / q = quit: ", end="", flush=True)
        raw = input().strip().lower()
        if raw == "a":
            return "reconfig"
        if raw == "q":
            return "quit"
        if dry_run:
            print("\n[dry-run] would execute:")
            print(" ".join(cmd))
            return "quit"
        return "run"


# ======================== main flow ========================
if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print(" vLLM launcher | RTX 5090 sm120 | vllm 0.28.0")
    print(" Enter all the way = one-key start of the default model; a = advanced parameters (/? = help)")
    print("=" * 60)

    name = pick_model()
    path, src = resolve_model_dir(name)
    if path is None:
        print("[ERROR] incomplete model directory (config.json missing): " + name)
        sys.exit(1)
    print("\nselected: " + name + "  [" + src + "]")

    cfg = load_saved(name)   # parameters saved by the last successful start take priority
    cfg["tool_call_parser"] = "qwen3_coder"

    mode = input("\nEnter = start with defaults / a = advanced parameters: ").strip().lower()
    while True:
        if mode == "a":
            cfg = configure_advanced(cfg)

        cmd = build_command(path, cfg, model_name=name)

        action = show_estimate_and_confirm(cfg, cmd, dry_run, model_name=name)
        if action == "run":
            # Save before the start: even if it then fails the parameters are kept, which makes it easy to keep tuning (limit probing)
            save_params(name, cfg)
            run_vllm(cmd)
            break
        if action == "quit":
            break
        mode = "a"   # after reconfiguring, go through the estimate/confirm step again
