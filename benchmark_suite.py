#!/usr/bin/env python3
"""Run FMBench with multiple configurations. Logs saved to suite_logs/."""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Edit this list to customize benchmark runs
CONFIGS = [
    {"log_level": "INFO", "model": "tinyllama", "scenario": "idle", "scenario.idle_duration": "10"},
    {"log_level": "INFO", "model": "tinyllama", "scenario": "arc_easy", "+scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "tinyllama", "scenario": "arc_challenge", "+scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "tinyllama", "scenario": "classification", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "tinyllama", "scenario": "helm", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "tinyllama", "scenario": "ner", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "tinyllama", "scenario": "sentiment", "scenario.num_samples": "20"},

    {"log_level": "INFO", "model": "qwen2.5", "scenario": "idle", "scenario.idle_duration": "10"},
    {"log_level": "INFO", "model": "qwen2.5", "scenario": "arc_easy", "+scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "qwen2.5", "scenario": "arc_challenge", "+scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "qwen2.5", "scenario": "classification", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "qwen2.5", "scenario": "helm", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "qwen2.5", "scenario": "ner", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "qwen2.5", "scenario": "sentiment", "scenario.num_samples": "20"},

    {"log_level": "INFO", "model": "falcon-7b", "scenario": "idle", "scenario.idle_duration": "10"},
    {"log_level": "INFO", "model": "falcon-7b", "scenario": "arc_easy", "+scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "falcon-7b", "scenario": "arc_challenge", "+scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "falcon-7b", "scenario": "classification", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "falcon-7b", "scenario": "helm", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "falcon-7b", "scenario": "ner", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "falcon-7b", "scenario": "sentiment", "scenario.num_samples": "20"},

    {"log_level": "INFO", "model": "llama3-1b", "scenario": "idle", "scenario.idle_duration": "10"},
    {"log_level": "INFO", "model": "llama3-1b", "scenario": "arc_easy", "+scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "llama3-1b", "scenario": "arc_challenge", "+scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "llama3-1b", "scenario": "classification", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "llama3-1b", "scenario": "helm", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "llama3-1b", "scenario": "ner", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "llama3-1b", "scenario": "sentiment", "scenario.num_samples": "20"},

    {"log_level": "INFO", "model": "llama2-7b", "scenario": "idle", "scenario.idle_duration": "10"},
    {"log_level": "INFO", "model": "llama2-7b", "scenario": "arc_easy", "+scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "llama2-7b", "scenario": "arc_challenge", "+scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "llama2-7b", "scenario": "classification", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "llama2-7b", "scenario": "helm", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "llama2-7b", "scenario": "ner", "scenario.num_samples": "20"},
    {"log_level": "INFO", "model": "llama2-7b", "scenario": "sentiment", "scenario.num_samples": "20"},
]

LOG_DIR = Path("suite_logs")


def run_benchmark(config, log_file):
    """Run one benchmark, streaming output to console and log file."""
    args = [sys.executable, "run.py"] + [f"{k}={v}" for k, v in config.items()]
    
    start = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in proc.stdout:
        print(line, end="")
        log_file.write(line)
    
    proc.wait()
    return proc.returncode == 0, time.time() - start


def fmt_time(secs):
    """Format seconds as human-readable duration."""
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        return f"{int(secs // 60)}m {secs % 60:.0f}s"
    return f"{int(secs // 3600)}h {int((secs % 3600) // 60)}m"


def log(msg, file):
    """Print to console and write to log file."""
    print(msg)
    file.write(msg + "\n")
    file.flush()


def main():
    if not CONFIGS:
        sys.exit("No configurations defined. Edit CONFIGS in benchmark_suite.py")
    
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"suite_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    with open(log_path, "w", encoding="utf-8") as f:
        log(f"FMBench Suite - {len(CONFIGS)} configs - Log: {log_path}", f)
        log("=" * 60, f)
        
        results = []
        total_start = time.time()
        
        for i, cfg in enumerate(CONFIGS, 1):
            cfg_str = " ".join(f"{k}={v}" for k, v in cfg.items())
            log(f"\n[{i}/{len(CONFIGS)}] {cfg_str}", f)
            log("-" * 60, f)
            
            ok, dur = run_benchmark(cfg, f)
            results.append((cfg_str, ok, dur))
            log(f"{'PASSED' if ok else 'FAILED'} in {fmt_time(dur)}\n", f)
        
        # Summary
        log("=" * 60, f)
        log("SUMMARY", f)
        for cfg_str, ok, dur in results:
            log(f"  [{'PASS' if ok else 'FAIL'}] {fmt_time(dur):>10}  {cfg_str}", f)
        
        passed = sum(ok for _, ok, _ in results)
        log(f"\n{passed}/{len(results)} passed in {fmt_time(time.time() - total_start)}", f)
        log(f"Log: {log_path}", f)
    
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()