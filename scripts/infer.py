#!/usr/bin/env python
"""
Load base gpt-oss-12b + trained LoRA adapter and generate a Tamil film plot
based on actor and genres.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "gpt-oss-12b"              # base checkpoint
ADAPTER_DIR = "checkpoints/gpt\_oss\_12b\_tamil\_lora/lora"  # trained adapter

SYSTEM = "You are a helpful Tamil film story writer. Write engaging, spoiler-free plots in 120–200 words."

PROMPT_TEMPLATE = (
"<|system|>\n{system}\n<|user|>\nActor: {actor}\nGenres: {genres}\nTask: Write a concise plot for a Tamil film starring the actor and fitting the genres.\n<|assistant|>\n"
)

def build_prompt(actor, genres):
    return PROMPT_TEMPLATE.format(system=SYSTEM, actor=actor, genres=", ".join(genres))

def main():
# Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model and LoRA adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    # Example input
    actor = "Vijay"
    genres = ["action", "drama"]

    # Build prompt
    prompt = build_prompt(actor, genres)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and print result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== Generated Plot ===\n")
    print(generated_text)

if __name__ == "__main__":
    main()
