import os
os.environ["HF_TRANSFER"]="1"

# from transformers.models.gemma3 import Gemma3ForConditionalGeneration
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import time

device = "mps"
torch.set_default_device(device)

GEMMA_PATH = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    # torch_dtype=torch.bfloat16, # for a T4.
    # device_map=device,
    )
print(model)

prompt = "What is the capital of France?"
messages = [
    {"role": "user", "content": prompt}
]

formatted_prompt = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False, add_generation_prompt=True)

print(f"Formatted prompt: {formatted_prompt}\n\n---\n")

inputs = tokenizer(formatted_prompt, return_tensors="pt")

print(model.device)

print(f"Inputs: {inputs}\n\n---\n")

# Measure tokens per second
start_time = time.time()
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
end_time = time.time()

# Calculate metrics
generation_time = end_time - start_time
input_length = inputs['input_ids'].shape[1]
output_length = outputs.shape[1]
generated_tokens = output_length - input_length
tokens_per_second = generated_tokens / generation_time

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"\n--- Performance Metrics ---")
print(f"Generated tokens: {generated_tokens}")
print(f"Generation time: {generation_time:.2f} seconds")
print(f"Tokens per second: {tokens_per_second:.2f}")