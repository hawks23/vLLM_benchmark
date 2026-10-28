"""Run 16-, 20-, and 24-request workloads through all four benchmark cases.

Usage (WSL): python3 max_concurrency_tests.py
Optional: --rounds 3 or --max-tokens 1024
Only aggregate settings/metrics are saved; no per-token logs.
"""
import sys
from pathlib import Path
from total_benchmark import main


if __name__ == '__main__':
    main([
        '--request-counts', '16', '20', '24',
        '--results-file', str(Path(__file__).resolve().parent / 'max_concurrency_tests.json'),
        *sys.argv[1:],
    ])
