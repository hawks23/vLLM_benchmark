"""Local streaming benchmark: python3 try.py --help. Run inside WSL."""
import os
os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
import argparse
import asyncio
import json
import platform
import re
import statistics
import threading
import time
from collections.abc import Mapping
from numbers import Integral
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def save_report(report, output_dir, label):
    """Write complete JSON atomically; each run has a distinct labeled file."""
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r'[^A-Za-z0-9_-]+', '_', label).strip('_')[:80] or 'run'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = outdir / f'{safe_label}_{stamp}.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return path

PROMPTS = [
    'Explain how a transformer language model turns a question into an answer. Cover tokenization, attention, training versus inference, and sampling. Use a concrete example. Write approximately 300 words for a beginner.',
    'A university wants a local assistant for lecture notes on a laptop with 12 GB GPU memory. Compare a small full-precision model with a larger 4-bit model. Discuss quality, speed, context length, privacy and evaluation. Recommend an approach in approximately 300 words.',
    'Write a Python function that groups dictionaries by a requested key without changing the original data. Handle missing keys explicitly. Include a worked example, three useful tests and time and space complexity. Aim for 250 to 350 words including code.',
    'A library receives 120 visitors per hour. Each stays an average of 30 minutes. There are 50 seats. Calculate expected occupancy, explain why averages do not guarantee sufficient seats, and propose a measurement plan before expansion. State assumptions in approximately 300 words.',
    'Explain the difference between concurrency and parallelism using a restaurant analogy, then relate it to an AI inference server. Include queueing, batching, latency and throughput. Write approximately 200 words.',
    'Design a reproducible experiment comparing two local language models for summarizing lecture notes. Describe the dataset, quality rubric, timing measurements, hardware controls and common pitfalls. Write approximately 350 words.',
]


async def dispatch_requests(items, worker, mode, batch_size):
    """Static waves never refill a slot until the entire wave finishes."""
    if mode == 'continuous':
        return await asyncio.gather(*(worker(i, item) for i, item in enumerate(items)))
    rows = []
    for start in range(0, len(items), batch_size):
        rows.extend(await asyncio.gather(*(worker(i, items[i])
                    for i in range(start, min(start + batch_size, len(items))))))
    return rows

def dist_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

def stats(values):
    if not values:
        return None
    values = sorted(values)
    def percentile(p):
        i = (len(values) - 1) * p
        lo = int(i)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (i - lo)
    return dict(mean=statistics.mean(values), p50=percentile(.5), p95=percentile(.95), peak=max(values))

class Hardware:
    """Sample whole-GPU counters and WSL-system CPU/RAM, not process-only."""
    def __init__(self, interval, gpu):
        self.interval, self.samples, self.notes, self.device = interval, [], [], {}
        self.stop_event = threading.Event()
        self.ps = self.nv = self.handle = None
        try:
            import psutil
            self.ps = psutil
        except ImportError:
            self.notes.append('psutil unavailable: CPU and RAM metrics omitted.')
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            for name, fun in [('name', pynvml.nvmlDeviceGetName)]:
                value = fun(self.handle)
                self.device[name] = value.decode() if isinstance(value, bytes) else value
        except Exception as exc:
            self.notes.append(f'NVML initialization: {exc}')
    def sample(self):
        row = dict(time_s=time.perf_counter() - self.start_time,
                   timestamp_utc=datetime.now(timezone.utc).isoformat())
        if self.ps:
            row.update(cpu_percent=self.ps.cpu_percent(),
                       wsl_ram_used_gib=self.ps.virtual_memory().used / 2**30,
                       wsl_swap_used_gib=self.ps.swap_memory().used / 2**30)
        if self.handle is not None:
            n, h = self.nv, self.handle
            queries = {
                'gpu_percent': lambda: n.nvmlDeviceGetUtilizationRates(h).gpu,
                'gpu_vram_used_gib': lambda: n.nvmlDeviceGetMemoryInfo(h).used / 2**30,
                'gpu_vram_total_gib': lambda: n.nvmlDeviceGetMemoryInfo(h).total / 2**30,
                'gpu_power_w': lambda: n.nvmlDeviceGetPowerUsage(h) / 1000,
                'gpu_temperature_c': lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
            }
            for key, fun in queries.items():
                try:
                    row[key] = fun()
                except Exception:
                    row[key] = None
        self.samples.append(row)
    def loop(self):
        if self.ps:
            self.ps.cpu_percent()
        while not self.stop_event.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.notes.append(f'Sampling error: {exc}')
                return
    def start(self):
        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
    def stop(self):
        self.stop_event.set()
        self.thread.join()
        if self.nv:
            self.nv.nvmlShutdown()
    def summary(self):
        keys = ['cpu_percent', 'wsl_ram_used_gib', 'wsl_swap_used_gib', 'gpu_percent',
                'gpu_vram_used_gib', 'gpu_power_w', 'gpu_temperature_c']
        return {k: stats([s[k] for s in self.samples if s.get(k) is not None]) for k in keys}

async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='baseline')
    parser.add_argument('--requests', type=int, default=4,
                        help='Requests per round; repeat the six benchmark prompts in order if needed')
    parser.add_argument('--submission-mode', choices=['continuous', 'static-waves'], default='continuous')
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'benchmark_results'),
                        help='Directory for JSON reports (default: benchmark_results beside this script)')
    for name, default in [('rounds', 3), ('concurrency', 4), ('max-num-seqs', 1),
                          ('max-tokens', 512), ('max-model-len', 2048), ('batch-tokens', 512), ('seed', 42), ('monitor-gpu', 0)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in [('gpu-memory-utilization', .85), ('temperature', .8), ('top-p', .95), ('sample-interval', .5)]:
        parser.add_argument('--' + name, type=float, default=default)
    for name, default in [('eager', True), ('prefix-cache', False), ('thinking', False)]:
        parser.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=default)
    args = parser.parse_args()
    if args.requests <= 0:
        parser.error('requests must be positive')
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.requests)]
    if min(args.rounds, args.concurrency, args.max_tokens, args.sample_interval) <= 0:
        parser.error('rounds, concurrency, max-tokens and sample-interval must be positive')
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_settings = dict(model='sanskar003/Qwen3.5-9B-AWQ', quantization='compressed-tensors',
        dtype='float16', gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.batch_tokens, enforce_eager=args.eager,
        limit_mm_per_prompt={'image': 0, 'video': 0}, seed=args.seed,
        enable_prefix_caching=args.prefix_cache, enable_chunked_prefill=True,
        generation_config='vllm', disable_log_stats=True)
    sampling_settings = dict(temperature=args.temperature, top_p=args.top_p,
                             top_k=-1, max_tokens=args.max_tokens, seed=args.seed)
    print('Loading model. Initialization and warmup are excluded from measured rounds.', flush=True)
    start = time.perf_counter()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_settings))
    startup_s = time.perf_counter() - start
    monitor = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(engine_settings['model'])
        def encode(text):
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                tokenize=True, return_dict=False, add_generation_prompt=True,
                enable_thinking=args.thinking)
            # Transformers versions can return a mapping/BatchEncoding.
            # vLLM's TokensPrompt requires a flat list of integer token IDs.
            if isinstance(ids, Mapping):
                ids = ids['input_ids']
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            if not isinstance(ids, (list, tuple)) or not ids or not all(
                isinstance(token, Integral) for token in ids
            ):
                raise TypeError('Chat template must produce a nonempty flat list of integer token IDs')
            ids = [int(token) for token in ids]
            if len(ids) + args.max_tokens > args.max_model_len:
                raise ValueError('Prompt plus output budget exceeds max-model-len')
            return ids
        encoded = [encode(p) for p in prompts]
        async def generate(ids, request_id, max_tokens=None):
            settings = dict(sampling_settings)
            if max_tokens is not None:
                settings['max_tokens'] = max_tokens
            params = SamplingParams(**settings, output_kind=RequestOutputKind.DELTA)
            started = time.perf_counter()
            first = last = None
            count, text, finish = 0, [], None
            async for output in engine.generate(prompt={'prompt_token_ids': ids},
                    sampling_params=params, request_id=request_id):
                now = time.perf_counter()
                for part in output.outputs:
                    text.append(part.text)
                    finish = part.finish_reason or finish
                    if part.token_ids:
                        first = now if first is None else first
                        last = now
                        count += len(part.token_ids)
            elapsed = time.perf_counter() - started
            tpot = (last-first)/(count-1) if count > 1 and last > first else None
            return dict(request_id=request_id, input_tokens=len(ids), output_tokens=count,
                ttft_s=first-started if first is not None else None, latency_s=elapsed,
                tpot_s=tpot, decode_tokens_s=1/tpot if tpot else None,
                output_tokens_per_latency_s=count/elapsed, finish_reason=finish,
                text=''.join(text))
        start = time.perf_counter()
        await generate(encode('Explain briefly why the sky appears blue.'), 'warmup', 32)
        warmup_s = time.perf_counter() - start
        print(f'Initialization {startup_s:.2f}s; warmup {warmup_s:.2f}s. Measuring now.', flush=True)
        monitor = Hardware(args.sample_interval, args.monitor_gpu)
        monitor.start()
        records, rounds = [], []
        for ri in range(args.rounds):
            semaphore = asyncio.Semaphore(args.concurrency)
            async def limited(index, ids):
                async with semaphore:
                    client_wait = time.perf_counter() - start
                    row = await generate(ids, f'r{ri+1}-p{index+1}')
                    row.update(round=ri+1, prompt=prompts[index], client_wait_s=client_wait,
                        workload_ttft_s=client_wait+row['ttft_s'] if row['ttft_s'] is not None else None,
                        workload_latency_s=client_wait+row['latency_s'])
                    return row
            start = time.perf_counter()
            rows = await dispatch_requests(encoded, limited, args.submission_mode, args.max_num_seqs)
            elapsed = time.perf_counter() - start
            tokens = sum(r['output_tokens'] for r in rows)
            rounds.append(dict(round=ri+1, wall_s=elapsed, output_tokens=tokens,
                               start_telemetry_s=start-monitor.start_time,
                               end_telemetry_s=start-monitor.start_time+elapsed,
                               output_tokens_s=tokens/elapsed, requests_s=len(rows)/elapsed))
            records.extend(rows)
        monitor.stop()
        hardware = monitor.summary()
        elapsed = sum(r['wall_s'] for r in rounds)
        tokens = sum(r['output_tokens'] for r in records)
        summary = dict(measured_wall_s=elapsed, output_tokens=tokens,
            input_tokens=sum(r['input_tokens'] for r in records),
            aggregate_output_tokens_s=tokens/elapsed, requests_s=len(records)/elapsed,
            length_limited_outputs=sum(r['finish_reason'] == 'length' for r in records))
        for key in ['ttft_s', 'latency_s', 'tpot_s', 'decode_tokens_s',
                    'client_wait_s', 'workload_ttft_s', 'workload_latency_s']:
            summary[key] = stats([r[key] for r in records if r[key] is not None])
        report = dict(schema_version=1, status='completed',
            timestamp_utc=datetime.now(timezone.utc).isoformat(), benchmark=vars(args),
            engine_settings=engine_settings, sampling_settings=sampling_settings,
            sampling_params_resolved=repr(SamplingParams(**sampling_settings, output_kind=RequestOutputKind.DELTA)),
            environment={k: os.environ.get(k) for k in ['VLLM_USE_FLASHINFER_SAMPLER', 'CUDA_HOME', 'CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE']},
            versions={k: dist_version(k) for k in ['vllm', 'torch', 'transformers', 'flashinfer-python', 'psutil', 'nvidia-ml-py']},
            platform=platform.platform(), torch_cuda=torch.version.cuda,
            hardware_device=monitor.device, startup_s=startup_s, warmup_s=warmup_s,
            summary=summary, rounds=rounds, requests=records, hardware_summary=hardware,
            hardware_samples=monitor.samples, notes=monitor.notes)
        report['measurement_metadata'] = {
            'telemetry_scope': 'Measured rounds only; excludes model initialization and warmup.',
            'sample_interval_s': args.sample_interval,
            'sample_count': len(monitor.samples),
            'missing_counter': 'null or absent means unavailable; never interpret as zero.',
            'hardware_scope': 'GPU: whole physical device. CPU/RAM/swap: WSL system, not Windows overall.',
            'ttft_s': 'Request submission to first observed token-bearing chunk; includes engine queueing, excludes tokenization and client semaphore wait.',
            'latency_s': 'Request submission to end of stream.',
            'tpot_s': '(Last observed token time - first observed token time) / (output tokens - 1). Chunk arrival approximation.',
            'decode_tokens_s': 'Reciprocal of tpot_s; null if insufficient observations.',
            'aggregate_output_tokens_s': 'Total output tokens / sum of measured round wall times.',
            'telemetry_time_s': 'Seconds since hardware monitoring started; round offsets use the same clock.',
            'units': {'time': 'seconds', 'memory': 'GiB (2^30 bytes)', 'power': 'watts',
                      'temperature': 'degrees Celsius', 'utilization': 'percent'},
            'limitations': ['Sampled peaks may miss brief spikes.',
                           'One 32-token warmup; subsequent lazy initialization may affect the first round.',
                           'P95 is descriptive for this small sample; repeated prompts are not independent workloads.'],
        }
        # Persist only comparison essentials. Token arrivals are timed, not logged;
        # individual answers and hardware samples are excluded from the JSON.
        compact_report = {
            'schema_version': 2,
            'timestamp_utc': report['timestamp_utc'],
            'label': args.label,
            'settings': {
                'engine': engine_settings,
                'sampling': sampling_settings,
                'rounds': args.rounds,
                'prompts_per_round': len(prompts),
                'submission_mode': args.submission_mode,
                'prompt_policy': 'Repeat the same six prompts in order; prefix caching disabled unless explicitly enabled.',
                'concurrency': args.concurrency,
                'thinking': args.thinking,
                'flashinfer_sampler': False,
                'hardware_sample_interval_s': args.sample_interval,
            },
            'system': {
                'gpu': monitor.device.get('name'),
                'vllm': report['versions']['vllm'],
                'torch': report['versions']['torch'],
                'cuda': report['torch_cuda'],
            },
            'performance': {
                'initialization_s': startup_s,
                'warmup_s': warmup_s,
                'request_count': len(records),
                **summary,
            },
            'hardware': {
                key: {'mean': data['mean'], 'peak': data['peak']} if data else None
                for key, data in hardware.items()
            },
            'measurement_notes': [
                'Performance and hardware aggregates exclude initialization and warmup.',
                'TTFT includes engine queueing; excludes tokenization and client semaphore wait.',
                'workload_ttft_s and workload_latency_s start at the round boundary, including static-wave/client waits; use these to compare submission modes.',
                'static-waves simulates no batch refill at the client; it does not disable the internal vLLM scheduler.',
                'Decode rate uses observed stream timing; GPU counters cover the whole device, CPU/RAM cover WSL.',
                'Hardware peaks are sampled; null means unavailable.',
                *monitor.notes,
            ],
        }
        path = save_report(compact_report, args.output_dir, args.label)
        def fmt(value):
            return 'N/A' if value is None else f'{value:.3f}'
        print('\n' + '='*80 + '\nANSWERS (first measured round)')
        for row in records[:len(prompts)]:
            print(f"\n{row['request_id']} | {row['output_tokens']} tokens | stop={row['finish_reason']}\n{row['text']}")
        print('\n' + '='*80 + '\nBENCHMARK: ' + args.label)
        print(f'Initialization: {startup_s:.2f}s | warmup: {warmup_s:.2f}s (excluded)')
        print(f'Measured: {elapsed:.2f}s | requests: {len(records)} | output tokens: {tokens}')
        print(f'Aggregate output: {tokens/elapsed:.2f} tok/s | requests: {len(records)/elapsed:.3f}/s')
        print(f"Length-limited outputs: {summary['length_limited_outputs']}/{len(records)}")
        print(f"{'Metric':28} {'Mean':>12} {'P50':>12} {'P95':>12}")
        for key in ['ttft_s', 'latency_s', 'tpot_s', 'decode_tokens_s']:
            data = summary[key] or {}
            print(f"{key:28} {fmt(data.get('mean')):>12} {fmt(data.get('p50')):>12} {fmt(data.get('p95')):>12}")
        print(f"\n{'Request':12} {'Input':>6} {'Output':>6} {'TTFT s':>10} {'Latency s':>12} {'Decode tok/s':>14}")
        for r in records:
            print(f"{r['request_id']:12} {r['input_tokens']:6} {r['output_tokens']:6} {fmt(r['ttft_s']):>10} {fmt(r['latency_s']):>12} {fmt(r['decode_tokens_s']):>14}")
        print('\nPER ROUND')
        for r in rounds:
            print(f"Round {r['round']}: {r['wall_s']:.2f}s | {r['output_tokens_s']:.2f} output tok/s")
        print('\nHARDWARE (sampled during measured rounds)')
        for key, data in hardware.items():
            data = data or {}
            print(f"{key:28} mean={fmt(data.get('mean')):>9} peak={fmt(data.get('peak')):>9}")
        print('\nSETTINGS / VERSIONS')
        for group in ['benchmark', 'engine_settings', 'sampling_settings', 'environment', 'versions', 'hardware_device']:
            print(group + ':')
            for key, value in report[group].items():
                print(f'  {key:30} {value}')
        print('Resolved sampling parameters:', report['sampling_params_resolved'])
        print(f"PyTorch CUDA: {report['torch_cuda']} | {report['platform']}")
        print('\nTTFT includes engine queueing; excludes tokenization and client semaphore wait.')
        print('TPOT/decode rate use observed token-bearing chunks, not internal kernel timestamps.')
        print('GPU counters include all device users; CPU/RAM describe WSL, not Windows overall.')
        print('N/A means unsupported. Sampled peaks can miss spikes. Small-sample P95 is descriptive.')
        print('One short warmup is used; first measured round can still include lazy initialization.')
        for note in monitor.notes:
            print(note)
        print('Aggregate metrics and settings JSON:', path)
    finally:
        if monitor is not None and not monitor.stop_event.is_set():
            monitor.stop()
        engine.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
