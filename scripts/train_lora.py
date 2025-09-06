import argparse
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
import os


def load_model_and_tokenizer(model_name, token=None):
    """
    Load model and tokenizer. Detect if model is already quantized with MXFP4,
    otherwise optionally apply BitsAndBytes 4bit quantization for QLoRA.
    """
    print(f"Loading base model: {model_name}")

    # First, try to load without forcing quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
        use_auth_token=token if token else None,
    )

    # Check if model is already quantized (MXFP4 etc.)
    quantization_config = getattr(model, "quantization_config", None)
    if quantization_config is not None and quantization_config.__class__.__name__ == "Mxfp4Config":
        print("⚡ Model is already quantized with MXFP4. Skipping BitsAndBytesConfig.")
    else:
        # If model is not pre-quantized, reload with BitsAndBytes (QLoRA)
        print("⚡ Model is not pre-quantized. Reloading with BitsAndBytes 4-bit quantization.")
        del model  # free memory
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
            quantization_config=bnb_config,
            use_auth_token=token if token else None,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_auth_token=token if token else None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model ID (e.g. 'openai-community/gpt-oss-20b') or local path to weights")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token if model is private")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training JSONL (sft_train.jsonl)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/gpt_oss_lora")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    args = parser.parse_args()

    # Load base model
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.hf_token)

    # Setup LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # TODO: add dataset loading + Trainer loop here
    print("✅ Model and tokenizer loaded successfully. Ready to fine-tune with LoRA!")


if __name__ == "__main__":
    main()
