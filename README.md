# Fine-Tuning Gemma 3 for Receipt OCR & Structured Information Extraction

This repository fine-tunes **Google Gemma 3 4B** (`google/gemma-3-4b-pt`), a vision-language model, to read receipt images and extract structured JSON data (items, prices, subtotals, and totals) using the **CORD-v2** dataset (`naver-clova-ix/cord-v2`).

Fine-tuning is performed using parameter-efficient fine-tuning (PEFT LoRA) on the attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`), utilizing SDPA memory-efficient attention and an 8-bit AdamW optimizer to run within standard GPU memory (T4 16GB / cloud GPUs).

---

## Dataset: CORD-v2

We use [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2), standard Indonesian receipt dataset with:
- **Train split**: 800 receipts
- **Validation split**: 100 receipts
- **Test split**: 100 receipts

Each receipt's ground truth contains a structured `gt_parse` schema with line items (`menu`: name, count, price), `sub_total`, and `total`. Singletons are normalized into lists during collation so the model learns a consistent JSON schema.

---

## Setup & Installation

```bash
git clone https://github.com/Siddz-17/gemma3-object-detection.git
cd gemma3-object-detection
pip install -r requirements.txt
```

Log in to Hugging Face (Gemma 3 is a gated model):
```bash
hf auth login
```

---

## Usage

1. **Inspect Dataset Schema (`create_dataset.py`)**:
   Sanity-checks the CORD-v2 download and prints the normalized JSON schema of sample 0:
   ```bash
   python create_dataset.py
   ```

2. **Configuration (`config.py`)**:
   Central configuration for dataset ID, model ID, batch size, learning rate, and generation token limits.

3. **Training (`train.py`)**:
   Fine-tunes Gemma 3 with SDPA attention, gradient accumulation, gradient checkpointing, and 8-bit AdamW:
   ```bash
   python train.py
   ```

4. **Prediction & Extraction (`predict.py`)**:
   Runs inference on the test split, parsing generated JSON text and saving paired `.png` images and `.json` sidecars in `outputs/`:
   ```bash
   python predict.py
   ```

---

## Key Features & Optimizations

- **Prompt Format**: Uses PaliGemma/Gemma 3 format: `<boi> extract \n\n{target_json} <eos>`.
- **PEFT LoRA**: Targets attention projections with rank $r=16$, $\alpha=32$, dropping trainable parameters from 4B to ~15M.
- **Memory Efficient**:
  - `attn_implementation="sdpa"` eliminates $O(N^2)$ attention matrices.
  - 8-bit AdamW reduces optimizer state memory by 75%.
  - Gradient accumulation (`gradient_accumulation_steps=4`) maintains effective batch size while keeping peak VRAM under 10 GB.
- **Robust Model Saving**: Automatically merges LoRA weights back into the base model before pushing to Hugging Face Hub, allowing standard zero-dependency downstream inference.
