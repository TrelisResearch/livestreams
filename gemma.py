import os
os.environ["HF_TRANSFER"]="1"

# from transformers.models.gemma3 import Gemma3ForConditionalGeneration
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
GEMMA_PATH = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    # torch_dtype=torch.bfloat16, # for a T4.
    # device_map="mps",
    )
print(model)

prompt = "What is the capital of France?"
messages = [
    {"role": "user", "content": prompt}
]

formatted_prompt = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False, add_generation_prompt=True)

print(f"Formatted prompt: {formatted_prompt}\n\n---\n")

inputs = tokenizer(formatted_prompt, return_tensors="pt")

print(f"Inputs: {inputs}\n\n---\n")

outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))