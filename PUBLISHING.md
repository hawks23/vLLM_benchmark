# Publishing this folder

Create an empty repository on GitHub, then run the commands below from this folder. Replace `YOUR-USERNAME` with your GitHub account. The suggested repository name is `vllm-benchmark` and the description is:

> Local vLLM concurrency benchmarks on an RTX 5070 Ti Laptop GPU, with WSL/CUDA setup and saved results.

Before committing, set a public display name and your GitHub-provided noreply email for this repository. Git stores author names and email addresses in commit history, independently of file contents. Copy the exact noreply address from your GitHub email settings.

```bash
git init -b main
git config user.name "YOUR-PUBLIC-HANDLE"
git config user.email "YOUR-GITHUB-NOREPLY-ADDRESS"
git add README.md RESULTS.md PUBLISHING.md requirements.txt .gitignore .gitattributes
git add try.py total_benchmark.py max_concurrency_tests.py summarize_results.py
git add benchmark_results max_concurrency_tests.json results_summary.csv
git diff --cached --stat
git diff --cached
git commit -m "Add vLLM concurrency benchmarks and recorded results"
git remote add origin https://github.com/YOUR-USERNAME/vllm-benchmark.git
git push -u origin main
```

The original project remains separate. This folder includes 26 JSON files: 22 standalone reports, three suite comparisons, and a duplicate named aggregate of the large sweep. Only stored filesystem path prefixes were changed in those reports; numeric values were preserved.

New benchmark runs in `runs/`, virtual environments, credentials, model downloads, and logs are ignored. Review any additional files before staging them. A new run using the scripts' default output directory writes into the tracked `benchmark_results/` tree, so use the README commands to keep fresh runs separate.

No license has been selected. Add one if you want to grant reuse rights; third-party model weights and dependencies retain their own terms. Model weights are not included here.
