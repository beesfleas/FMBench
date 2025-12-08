"""
VLM Output Quality Check - Shows actual model outputs for verification.
"""
import sys
from pathlib import Path
from datasets import load_dataset
from components.models.model_factory import get_model_loader

# Models to test - using ones that work without special requirements
MODELS = [
    ("smolvlm", "HuggingfaceTB/SmolVLM-256M-Instruct", {}),
    ("llava", "llava-hf/llava-1.5-7b-hf", {}),
]

def test_model(config_name, model_id, extra_config):
    """Test a single model and show its output."""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_id}")
    print(f"{'='*70}")
    
    # Load model
    config = {
        "model_id": model_id,
        "model_category": "VLM",
        "max_tokens": 64,
        **extra_config
    }
    
    try:
        loader = get_model_loader(config)
        loader.load_model(config)
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return False
    
    all_sensible = True
    
    # Test 1: HaGRID gesture recognition
    print("\n--- Test 1: HaGRID (Gesture Recognition) ---")
    try:
        ds = load_dataset('cj-mills/hagrid-sample-30k-384p', split='train')
        image = ds[0]['image'].convert('RGB')
        label = ds.features['label'].names[ds[0]['label']]
        label = label.replace('train_val_', '')  # Clean label
        
        # Use prompt WITH <image> placeholder as required by VLM loader
        prompt = "User: <image>\nWhat hand gesture is shown? Answer in one word.\nAssistant:"
        result = loader.predict(prompt, image=image)
        output = result['output'] if isinstance(result, dict) else result
        
        print(f"  Expected: {label}")
        print(f"  Output:   {output[:100]}")
        
        # Check if sensible (not empty, not just repeating prompt, contains words)
        output_clean = output.strip().lower()
        is_sensible = (
            len(output_clean) > 0 and  # Not empty
            len(output_clean) < 200 and  # Not too long (rambling)
            "what hand" not in output_clean  # Not repeating prompt
        )
        print(f"  Sensible: {'✓' if is_sensible else '✗'}")
        all_sensible &= is_sensible
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        all_sensible = False
    
    # Test 2: GTSRB traffic sign
    print("\n--- Test 2: GTSRB (Traffic Sign Recognition) ---")
    try:
        ds = load_dataset('tanganke/GTSRB', split='test')
        image = ds[0]['image'].convert('RGB')
        label = ds[0]['label']
        
        prompt = "User: <image>\nWhat type of traffic sign is this? Answer briefly.\nAssistant:"
        result = loader.predict(prompt, image=image)
        output = result['output'] if isinstance(result, dict) else result
        
        print(f"  Label ID: {label}")
        print(f"  Output:   {output[:100]}")
        
        output_clean = output.strip().lower()
        is_sensible = (
            len(output_clean) > 0 and
            len(output_clean) < 200 and
            "what type" not in output_clean
        )
        print(f"  Sensible: {'✓' if is_sensible else '✗'}")
        all_sensible &= is_sensible
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        all_sensible = False
    
    # Test 3: Simple question
    print("\n--- Test 3: Simple Visual Question ---")
    try:
        # Use HaGRID image for a simple question
        prompt = "User: <image>\nDescribe what you see in this image in one sentence.\nAssistant:"
        result = loader.predict(prompt, image=image)
        output = result['output'] if isinstance(result, dict) else result
        
        print(f"  Output:   {output[:150]}")
        
        output_clean = output.strip().lower()
        is_sensible = (
            len(output_clean) > 5 and
            len(output_clean) < 300 and
            "describe" not in output_clean
        )
        print(f"  Sensible: {'✓' if is_sensible else '✗'}")
        all_sensible &= is_sensible
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        all_sensible = False
    
    # Unload model
    try:
        loader.unload_model()
    except:
        pass
    
    print(f"\n  Overall: {'✓ PASS' if all_sensible else '✗ NEEDS REVIEW'}")
    return all_sensible


def main():
    print("VLM Output Quality Check")
    print("=" * 70)
    
    results = {}
    for config_name, model_id, extra in MODELS:
        try:
            results[config_name] = test_model(config_name, model_id, extra)
        except Exception as e:
            print(f"\n❌ {config_name} completely failed: {e}")
            results[config_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL/NEEDS REVIEW"
        print(f"  {name}: {status}")
    
    passed_count = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed_count}/{len(results)} models produce sensible output")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
