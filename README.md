# 🎬 Tamil Film LoRA Fine‑Tuning (gpt‑oss‑20b)

This project demonstrates how to fine‑tune the **gpt‑oss‑20b** model with **LoRA (Low‑Rank Adaptation)** on a Tamil film dataset. The fine‑tuned model generates engaging and spoiler‑free film plots based on **actor** and **genres**.

---

## 📂 Project Structure

```bash
.
├── data/
│   ├── sample_tamil_movies.jsonl   # raw dataset (10 Tamil films)
│   ├── sft_train.jsonl             # processed instruction dataset
├── scripts/
│   ├── prepare_dataset.py          # converts raw data → SFT format
│   ├── train_lora.py               # fine‑tuning script
│   ├── infer.py                    # inference script
└── checkpoints/
    └── gpt_oss_20b_tamil_lora/     # trained LoRA adapters
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

> 💡 A GPU with 24GB+ VRAM is recommended. For smaller GPUs, adjust batch size, LoRA rank, or sequence length.

---

## 🚀 Workflow

### 1. Prepare Dataset

Convert raw dataset to instruction‑tuning format:

```bash
python scripts/prepare_dataset.py
```

### 2. Train LoRA Adapter

Login to Huggingface using your token.

```bash
hf auth login
```


Fine‑tune the model with LoRA:

```bash
python scripts/train_lora.py \
  --model_name openai/gpt-oss-20b \
  --data_path data/sft_train.jsonl \
  --output_dir checkpoints/gpt_oss_20b_tamil_lora \
  --epochs 3 --per_device_batch_size 1 --gradient_accumulation_steps 8
```

### 3. Run Inference

Generate a plot using the trained adapter:

```bash
python scripts/infer.py
```

Example output:

```
=== Generated Plot ===
A principled bus conductor is pushed into vigilantism when a land mafia threatens his township...
```

---

## 📘 Dataset Format

Each entry in the raw dataset includes an actor, genres, and plot:

```json
{
  "actor": "Vijay",
  "genres": ["action", "drama"],
  "plot": "A principled bus conductor is pushed into vigilantism..."
}
```

After running `prepare_dataset.py`, entries are transformed into **instruction‑style JSONL** for SFT training.

---

## 🛠️ Notes

* Adjust LoRA parameters (`--lora_r`, `--lora_alpha`) based on VRAM.
* Use `bfloat16` or `float16` precision for efficiency.
* Extend dataset for stronger generalization.

---

## 🙌 Acknowledgements

* **Hugging Face Transformers** for model loading & training utilities.
* **PEFT (Parameter‑Efficient Fine‑Tuning)** for LoRA.
* Inspired by Tamil cinema storytelling traditions.
