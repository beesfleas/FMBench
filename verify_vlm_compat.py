"""
VLM Compatibility Verification Script.
Tests multiple VLM models across multiple VLM scenarios.
"""
import subprocess
import sys
import json
from pathlib import Path

# Models to test (config name -> model_id for display)
MODELS = [
    # Larger models with 4-bit quantization
    "qwen2.5-vl",
    "minicpm-v",
    # "molmo",  # Requires trust_remote_code
    # "llama-vision",  # Requires Meta access
]

# Scenarios to test
SCENARIOS = [
    "hagrid",
    "gtsrb", 
    "countbenchqa",
    "docvqa",
    "vqa",
]

NUM_SAMPLES = 2  # Quick test with 2 samples

def run_test(model: str, scenario: str) -> dict:
    """Run a single model-scenario combination and return results."""
    cmd = [
        sys.executable, "run.py",
        f"scenario={scenario}",
        f"model={model}",
        f"scenario.num_samples={NUM_SAMPLES}"
    ]
    
    print(f"\n{'='*60}")
    print(f"Testing: {model} + {scenario}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per test
            cwd=Path(__file__).parent
        )
        
        # Check for success
        success = result.returncode == 0
        error_msg = None
        accuracy = None
        
        if not success:
            # Look for common error patterns
            stderr = result.stderr or result.stdout
            if "Image features and image tokens do not match" in stderr:
                error_msg = "Image token mismatch"
            elif "CUDA out of memory" in stderr:
                error_msg = "OOM"
            elif "Could not process VLM inputs" in stderr:
                error_msg = "Input format unsupported"
            elif "AutoModelForImageTextToText" in stderr and "not support" in stderr:
                error_msg = "Model architecture not supported"
            else:
                # Get last 200 chars of error
                error_msg = stderr[-200:] if stderr else "Unknown error"
        else:
            # Try to extract accuracy from output
            if "Accuracy:" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "Accuracy:" in line:
                        try:
                            # Parse "Accuracy: 80.00%" format
                            acc_str = line.split("Accuracy:")[1].strip().rstrip('%')
                            accuracy = float(acc_str) / 100
                        except:
                            pass
        
        return {
            "model": model,
            "scenario": scenario,
            "success": success,
            "accuracy": accuracy,
            "error": error_msg,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "model": model,
            "scenario": scenario,
            "success": False,
            "accuracy": None,
            "error": "Timeout (>10min)",
        }
    except Exception as e:
        return {
            "model": model,
            "scenario": scenario,
            "success": False,
            "accuracy": None,
            "error": str(e),
        }


def main():
    print("VLM Compatibility Verification")
    print(f"Models: {MODELS}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Samples per test: {NUM_SAMPLES}")
    print()
    
    results = []
    
    for model in MODELS:
        for scenario in SCENARIOS:
            result = run_test(model, scenario)
            results.append(result)
            
            status = "✓" if result["success"] else "✗"
            acc = f"{result['accuracy']:.0%}" if result['accuracy'] is not None else "N/A"
            err = f" ({result['error']})" if result['error'] else ""
            print(f"  {status} {model} + {scenario}: {acc}{err}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Build matrix
    print(f"\n{'Model':<20}", end="")
    for s in SCENARIOS:
        print(f"{s:<15}", end="")
    print()
    print("-" * (20 + 15 * len(SCENARIOS)))
    
    for model in MODELS:
        print(f"{model:<20}", end="")
        for scenario in SCENARIOS:
            r = next((r for r in results if r["model"] == model and r["scenario"] == scenario), None)
            if r:
                if r["success"]:
                    acc = f"{r['accuracy']:.0%}" if r['accuracy'] is not None else "OK"
                    print(f"{acc:<15}", end="")
                else:
                    print(f"FAIL<15", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()
    
    # Count successes
    success_count = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\nTotal: {success_count}/{total} tests passed")
    
    # Show failures
    failures = [r for r in results if not r["success"]]
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f['model']} + {f['scenario']}: {f['error']}")
    
    return 0 if success_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
