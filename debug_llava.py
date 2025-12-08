"""Debug LLaVA generation issue."""
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
import torch

print("Loading processor...")
processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    'llava-hf/llava-1.5-7b-hf',
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading HaGRID image...")
ds = load_dataset('cj-mills/hagrid-sample-30k-384p', split='train')
image = ds[0]['image'].convert('RGB')
print(f"Image size: {image.size}")

# Use proper LLaVA format
prompt = "USER: <image>\nWhat hand gesture is shown?\nASSISTANT:"
print(f"Prompt: {prompt}")

print("\nProcessing inputs...")
inputs = processor(text=prompt, images=image, return_tensors="pt")
print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"pixel_values shape: {inputs['pixel_values'].shape}")

# Move to model device
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("\nGenerating...")
try:
    output_ids = model.generate(**inputs, max_new_tokens=32)
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    print(f"Response: {response}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
