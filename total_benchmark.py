"""Run four vLLM scenarios per workload size sequentially inside WSL (default: six requests).
Case 3 uses static client waves, not a replacement vLLM scheduler.
Case 4 finds the highest successful scheduling cap up to the workload size.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MEMORY_ERRORS = ('cuda out of memory', 'torch.outofmemoryerror',
                 'no available memory for the cache blocks',
                 'less than desired gpu memory utilization',
                 'larger than the maximum number of tokens that can be stored in kv cache',
                 'not enough memory')


def save(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def execute(args, folder, label, cap, mode, request_count=6):
    destination = folder / label / f'cap_{cap}'
    command = [sys.executable, '-u', str(HERE / 'try.py'), '--label', label,
               '--requests', str(request_count), '--rounds', str(args.rounds), '--concurrency', str(request_count),
               '--max-num-seqs', str(cap), '--submission-mode', mode,
               '--max-tokens', str(args.max_tokens), '--max-model-len', str(args.max_model_len),
               '--batch-tokens', str(args.batch_tokens),
               '--gpu-memory-utilization', str(args.gpu_memory_utilization),
               '--output-dir', str(destination)]
    print(f'\n{label}: {request_count} requests/round, sequence cap={cap}, submission={mode}', flush=True)
    tail = deque(maxlen=35)
    memory_failure = False
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace')
    try:
        for line in process.stdout:
            tail.append(line.rstrip())
            memory_failure |= any(marker in line.lower() for marker in MEMORY_ERRORS)
            if any(marker in line for marker in ['Loading model.', 'Measuring now.', 'Aggregate metrics and settings JSON:']):
                print(line.strip(), flush=True)
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()
    attempt = dict(max_num_seqs=cap, exit_code=code, memory_failure=memory_failure)
    if code:
        print(f'Attempt failed (exit {code}, memory-related={memory_failure}).', flush=True)
        attempt['error_tail'] = list(tail)
        return None, attempt
    reports = sorted(destination.glob('*.json'))
    if len(reports) != 1:
        attempt['error_tail'] = ['Expected one aggregate report after successful child exit.']
        return None, attempt
    report = json.loads(reports[0].read_text(encoding='utf-8'))
    expected = request_count * args.rounds
    if report['performance']['request_count'] != expected:
        attempt['error_tail'] = [f'Expected {expected} completed requests.']
        return None, attempt
    attempt['report_file'] = str(reports[0])
    return report, attempt


def display(suite):
    print('\n' + '=' * 106)
    print('REQUEST COMPARISON (TTFT/latency include all waiting from each round start)')
    print(f"{'Scenario':27} {'Cap':>4} {'tok/s':>9} {'TTFT p50':>10} {'TTFT p95':>10} {'Latency p95':>12} {'VRAM GiB':>10}")
    for case in suite['cases']:
        report = case.get('report')
        if not report:
            print(f"{case['label']:27} {case['status'].upper()} (details in comparison report)")
            continue
        p = report['performance']
        vram = report['hardware'].get('gpu_vram_used_gib')
        def f(v):
            return 'N/A' if v is None else f'{v:.2f}'
        ttft = p.get('workload_ttft_s') or {}
        lat = p.get('workload_latency_s') or {}
        print(f"{case['label']:27} {case['selected_cap']:4} {f(p['aggregate_output_tokens_s']):>9} "
              f"{f(ttft.get('p50')):>10} {f(ttft.get('p95')):>10} {f(lat.get('p95')):>12} "
              f"{f(vram['peak'] if vram else None):>10}")
    print('Case 3: static pairs, no refill until both finish; internal vLLM scheduling remains enabled.')
    print('Case 4: highest successful cap <= workload size, not an absolute GPU limit or optimal throughput.')
    print('Successful runs save aggregate metrics only; errors retain a short diagnostic tail.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=1, help='Repetitions of each workload/scenario')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--max-model-len', type=int, default=2048)
    parser.add_argument('--batch-tokens', type=int, default=512)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.85)
    parser.add_argument('--output-dir', default=str(HERE / 'benchmark_results'))
    parser.add_argument('--request-counts', type=int, nargs='+', default=[6],
                        help='Workload sizes; all four scenarios run for each size')
    parser.add_argument('--results-file', help='Optional named aggregate report; previous content is backed up')
    args = parser.parse_args(argv)
    if any(n <= 0 for n in args.request_counts) or len(set(args.request_counts)) != len(args.request_counts):
        parser.error('request-counts must contain distinct positive integers')
    if min(args.rounds, args.max_tokens, args.max_model_len, args.batch_tokens) <= 0:
        parser.error('Counts and token budgets must be positive')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('GPU memory utilization must be between 0 and 1')
    if not (HERE / 'try.py').is_file():
        parser.error('The updated try.py must be beside total_benchmark.py')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = Path(args.output_dir).expanduser().resolve() / ('suite_' + stamp)
    folder.mkdir(parents=True)
    path = folder / 'comparison.json'
    named_path = Path(args.results_file).expanduser().resolve() if args.results_file else None
    if named_path:
        named_path.parent.mkdir(parents=True, exist_ok=True)
        if named_path.exists():
            backup = named_path.with_name(named_path.stem + '_previous_' + stamp + named_path.suffix)
            named_path.replace(backup)
            print(f'Previous results preserved at: {backup}', flush=True)
    def checkpoint():
        save(path, suite)
        if named_path:
            save(named_path, suite)
    suite = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                 status='running', settings=vars(args), cases=[],
                 notes=['All workload prompts are logically available at each round start in every case.',
                        'Workloads repeat the same six prompts in order; prefix caching is disabled. Different workload sizes can have slightly different prompt mixes.',
                        'Use workload timing to include client-side waits in static batching.',
                        'Same model, prompts, sampling and memory budget for every scenario; new engine process per attempt.',
                        'Maximum-cap search retries only recognized memory failures. It does not optimize throughput.',
                        'Initialization and warmup are excluded from throughput; caches and thermals can vary between cases.'])
    checkpoint()
    scenarios = []
    for count in args.request_counts:
        prefix = '' if args.request_counts == [6] else f'n{count}_'
        scenarios.extend([
            (prefix+'1_sequential', [1], 'continuous', count),
            (prefix+'2_continuous_pairs', [2], 'continuous', count),
            (prefix+'3_static_pairs', [2], 'static-waves', count),
            (prefix+'4_auto_capacity', range(count, 0, -1), 'continuous', count),
        ])
    try:
        for label, caps, mode, count in scenarios:
            case = dict(label=label, requests_per_round=count, submission_mode=mode, attempts=[], status='running')
            suite['cases'].append(case)
            checkpoint()
            for cap in caps:
                report, attempt = execute(args, folder, label, cap, mode, count)
                case['attempts'].append(attempt)
                if report is not None:
                    case.update(status='completed', selected_cap=cap, report=report,
                                full_workload_cap_reached=(cap == count))
                    checkpoint()
                    break
                case['status'] = 'failed'
                checkpoint()
                if not attempt['memory_failure']:
                    break
            checkpoint()
        suite['status'] = 'completed' if all(c['status'] == 'completed' for c in suite['cases']) else 'completed_with_failures'
    except BaseException as exc:
        suite['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        suite['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        checkpoint()
        display(suite)
        print(f'\nComparison JSON: {path}', flush=True)
        if named_path:
            print(f'Named aggregate results: {named_path}', flush=True)


if __name__ == '__main__':
    main()
