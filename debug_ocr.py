"""Debug script to check LLaVA output vs expected answers for DocVQA"""
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import traceback

try:
    # Load model
    print("Loading LLaVA model...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load dataset
    print("Loading DocVQA dataset...")
    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
    sample = next(iter(dataset))

    print(f"\n{'='*60}")
    print("SAMPLE INFO:")
    print(f"Question: {sample['question']}")
    print(f"Expected Answers: {sample['answers']}")
    print(f"{'='*60}")

    # Prepare input with LLaVA format
    image = sample['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')

    prompt = f"USER: <image>\nQuestion: {sample['question']}\nAnswer:\nASSISTANT:"
    print(f"\nPrompt:\n{prompt}")

    # Process and generate
    print("\nProcessing inputs...")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Move to GPU if available
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print("Generating response...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    # Decode - get only the new tokens
    input_len = inputs['input_ids'].shape[1]
    full_response = processor.decode(output_ids[0], skip_special_tokens=True)
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    print(f"\n{'='*60}")
    print("MODEL OUTPUT:")
    print(f"Full decoded: '{full_response}'")
    print(f"New tokens only: '{response}'")
    print(f"{'='*60}")

    # Calculate ANLS
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    output_norm = response.lower().strip()
    print(f"\nNormalized output: '{output_norm}'")

    max_anls = 0.0
    for ans in sample['answers']:
        ans_norm = ans.lower().strip()
        if not output_norm or not ans_norm:
            score = 0.0
            nl = float('inf')
            dist = 0
        else:
            dist = levenshtein_distance(output_norm, ans_norm)
            max_len = max(len(output_norm), len(ans_norm))
            nl = dist / max_len
            score = 1.0 - nl if nl < 0.5 else 0.0
        print(f"  vs '{ans_norm}': dist={dist}, NL={nl:.4f}, ANLS={score:.4f}")
        max_anls = max(max_anls, score)

    print(f"\nFinal ANLS: {max_anls:.4f}")

except Exception as e:
    print(f"\nERROR: {e}")
    traceback.print_exc()
