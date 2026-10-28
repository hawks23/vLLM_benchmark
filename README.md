# vLLM concurrency benchmarks on a laptop GPU

I built this benchmark to see how serving several requests together changes throughput and waiting time on a single GPU. I ran a quantized Qwen model locally with vLLM inside WSL, using an NVIDIA GeForce RTX 5070 Ti Laptop GPU and CUDA configured for WSL.

The experiment compares sequential generation, continuously refilled pairs, fixed pairs, and larger groups of concurrent requests. The scripts and saved results are included here.

## What I tested

Each scenario runs in a fresh Python process with the same model and sampling settings. Scenarios run one after another; requests within a scenario can run concurrently.

| Case | Active-sequence cap | How requests are submitted |
| --- | ---: | --- |
| Sequential | 1 | Submit the workload together; vLLM schedules one sequence at a time. |
| Continuous pairs | 2 | Keep requests available so vLLM can refill a slot when a sequence finishes. |
| Static pairs | 2 | Submit two requests and wait for both to finish before sending the next pair. |
| Auto capacity | Up to workload size | Start with a cap equal to the request count; lower it only after a recognized memory failure. |

Static pairs simulate waiting between batches at the client. They do not turn off vLLM's internal scheduler. Auto capacity finds a successful cap within the chosen workload size; it does not search for the fastest setting or establish the GPU's absolute limit.

The six built-in prompts cover transformer explanations, model selection, Python code, an occupancy calculation, concurrency, and experiment design. Larger workloads repeat these prompts in order. This is a text-generation timing experiment, with no answer-quality scoring or HTTP server in the measured path.

## Results

The saved runs are dated 24 October 2026. The main sweep used one round per scenario, 16/20/24 requests, and a maximum of 1,024 output tokens per request.

| Requests per round | Sequential tok/s | Continuous pairs tok/s | Static pairs tok/s | Full-workload cap tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 29.18 | 54.76 | 51.54 | 327.71 |
| 20 | 28.68 | 52.09 | 50.99 | 430.78 |
| 24 | 28.64 | 52.73 | 48.90 | 497.84 |

These are aggregate output tokens per second across all requests. They are not the decode speed of an individual response. All three full-workload caps succeeded on the first attempt, so this sweep did not reach a memory failure.

For 24 requests, the p95 completion time measured from the round's start fell from 383.23 seconds with a cap of one to 22.90 seconds with a cap of 24. Sampled peak GPU memory was 11.81 GiB and 11.48 GiB respectively. Those memory readings cover the whole GPU, including other processes.

There are also two four-request runs, a six-request smoke test with an eight-token output budget, and a six-request comparison with a 512-token budget. The smoke test is useful for checking the procedure, but its short outputs make it a poor basis for performance conclusions.

See [RESULTS.md](RESULTS.md) for every saved run, latency and memory measurements, and links to the original JSON reports. [results_summary.csv](results_summary.csv) contains the same comparison data in a flat table. Personal filesystem prefixes have been removed from the JSON; recorded numeric measurements are unchanged.

## Setup

The reports record this stack:

| Component | Recorded value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Model | `sanskar003/Qwen3.5-9B-AWQ` |
| Quantization backend | `compressed-tensors` |
| Compute dtype | `float16` |
| vLLM | `0.31.0` |
| PyTorch | `2.13.0+cu130` |
| PyTorch CUDA runtime | `13.0` |

The runs used a 2,048-token context limit, 512 batched tokens, GPU memory utilization of 0.85, eager execution, and chunked prefill. Prefix caching, thinking, image/video inputs, and the FlashInfer sampler were disabled. Sampling used temperature 0.8, top-p 0.95, top-k -1, and seed 42.

I used WSL to run the Linux inference stack on Windows and configured CUDA so PyTorch and vLLM could use the GPU from inside WSL. For a new setup, install a WSL-compatible NVIDIA driver on Windows and check GPU access with `nvidia-smi` inside WSL. NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/) explains the driver/toolkit split: do not install a Linux display driver inside WSL. If a CUDA toolkit is needed, use its WSL-compatible installation instructions.

The saved reports do not contain the Python, Transformers, Windows driver, or WSL distribution versions, or the model revision. The CUDA value above is PyTorch's runtime version, not proof of the installed toolkit version. `requirements.txt` pins the recorded vLLM version; it is not a complete historical environment lock. Follow the [vLLM GPU installation instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) if a compatible wheel is unavailable for your environment.

Run these commands inside WSL from the repository directory, using a Python version supported by the selected vLLM release:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
```

The CUDA availability check should print `True`. The first benchmark run downloads the model and tokenizer from Hugging Face. The model identifier is fixed in `try.py`.

## Run the benchmarks

```bash
# Four cases, six requests each, one round, up to 512 output tokens.
python total_benchmark.py --output-dir runs

# Reproduce the settings of the saved 16/20/24-request sweep.
python max_concurrency_tests.py --max-tokens 1024 --rounds 1 \
  --output-dir runs --results-file runs/max_concurrency_tests.json

# A single case: four requests, two active sequences, three rounds.
python try.py --label four_requests_two_active --requests 4 \
  --concurrency 4 --max-num-seqs 2 --rounds 3 --output-dir runs
```

Use `--help` on any script for its options. The wrapper defaults to a 512-token output budget; the saved large sweep used the explicit `--max-tokens 1024` override above. To collect more repetitions, increase `--rounds`.

New reports in `runs/` are ignored by Git. The wrapper's default named output is the root `max_concurrency_tests.json`, so the example overrides it to keep the archived result intact. Existing named outputs are backed up before replacement.

## How timing works

Each process loads the engine, generates a single 32-token warmup response, then measures the workload. Initialization and warmup have separate timings and are excluded from throughput and hardware aggregates.

- **Throughput:** total generated tokens divided by the sum of measured round durations.
- **TTFT:** submission to the first observed token-bearing stream chunk, including engine queueing.
- **Workload TTFT and latency:** start at the round boundary, so they include client-side and static-wave waiting. Use these when comparing the four cases.
- **Decode rate:** calculated from observed stream chunks. It is an approximation, not a GPU kernel measurement.
- **Hardware:** sampled every 0.5 seconds with NVML and psutil. GPU counters cover the whole device; CPU, RAM, and swap cover the WSL system.

Reports store aggregate measurements, not generated answers or per-token logs. Answers and diagnostic details can still appear in terminal output. The publication copy stops collecting GPU UUIDs, omits local output paths from new suite metadata, and keeps subprocess error traces out of saved reports. Review terminal logs separately before sharing them.

## Limits of the comparison

The main sweep has one round per case. The p95 values describe a small, repeated prompt set and are not production latency estimates. Cases always run in the same order, so cache state, laptop thermals, power limits, and background GPU activity can affect the comparison. A single short warmup may leave some initialization work for the first measured round.

Output lengths vary even with the same seed, and some earlier runs hit their token budget. Different workload sizes also repeat the six prompts in different proportions. Compare scenarios within the same workload and budget first. These results support what happened in these local runs; they do not establish performance on other models or hardware.

## Files

- `try.py`: streaming benchmark and hardware sampling.
- `total_benchmark.py`: subprocess runner for the four scenarios and memory-failure fallback.
- `max_concurrency_tests.py`: preset for 16, 20, and 24 requests.
- `benchmark_results/`: individual reports and three suite comparisons.
- `max_concurrency_tests.json`: duplicate aggregate of the saved large sweep, retained for convenience.
- `summarize_results.py`: rebuilds the Markdown and CSV tables using only the standard library.

```bash
python summarize_results.py
```

The publication preparation checked Python syntax, CLI help, report consistency, privacy, and summary generation. It did not rerun GPU inference; the measurements are the archived experiment results.
