import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
import time
import wandb
from functools import partial

import sys
# Bypass outdated pre-installed torchao (< 0.16.0) on cloud environments causing PEFT dispatch crash
sys.modules["torchao"] = None

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import LoraConfig, get_peft_model

from config import Configuration
from utils import (
    parse_cord_ground_truth,
    parse_model_output,
    train_collate_function,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_dataloader(processor, cfg):
    logger.info("Fetching the train dataset")
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=cfg.dtype
    )

    logger.info("Building train data loader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )
    return train_dataloader


def get_eval_dataloader(processor, cfg):
    logger.info("Fetching the validation dataset")
    val_dataset = load_dataset(cfg.dataset_id, split="validation")
    val_collate_fn = partial(
        train_collate_function, processor=processor, dtype=cfg.dtype
    )

    logger.info("Building validation data loader")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=val_collate_fn,
        shuffle=False,
    )
    return val_dataloader, val_dataset


def evaluate_loss(model, eval_dataloader, input_device, dtype):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    use_cuda = "cuda" in str(input_device)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_cuda):
            for batch in eval_dataloader:
                outputs = model(**batch.to(input_device))
                total_loss += outputs.loss.item()
                num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def log_validation_predictions(model, processor, cfg, val_dataset, input_device, epoch, global_step, num_samples=4):
    """Generate predictions on validation sample receipts and log an interactive visual table to W&B."""
    model.eval()
    table = wandb.Table(columns=["epoch", "receipt_image", "ground_truth_json", "extracted_json", "is_valid_json"])
    valid_json_count = 0
    num_to_log = min(num_samples, len(val_dataset))

    logger.info(f"Generating sample validation predictions for W&B table ({num_to_log} receipts)...")
    for i in range(num_to_log):
        sample = val_dataset[i]
        image = sample["image"]
        gt_parse = parse_cord_ground_truth(sample["ground_truth"])
        gt_json_str = json.dumps(gt_parse, indent=2, ensure_ascii=False)

        prompt = f"{processor.tokenizer.boi_token} extract \n\n"
        inputs = processor(images=[[image]], text=[prompt], return_tensors="pt").to(input_device)
        inputs["pixel_values"] = inputs["pixel_values"].to(cfg.dtype)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=cfg.dtype, enabled="cuda" in str(input_device)):
                output_ids = model.generate(**inputs, max_new_tokens=min(cfg.max_new_tokens, 256))

        pred_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        parsed = parse_model_output(pred_text)
        is_valid = isinstance(parsed, dict)
        if is_valid:
            valid_json_count += 1
            pred_display = json.dumps(parsed, indent=2, ensure_ascii=False)
        else:
            pred_display = pred_text

        # Create a thumbnail for responsive W&B gallery display
        display_img = image.copy()
        display_img.thumbnail((400, 600))
        table.add_data(epoch, wandb.Image(display_img), gt_json_str, pred_display, is_valid)

    validity_pct = (valid_json_count / num_to_log) * 100.0
    wandb.log({
        "val/prediction_samples": table,
        "val/json_validity_pct": validity_pct,
        "epoch": epoch,
    }, step=global_step)
    logger.info(f"[EVAL] JSON Validity Rate on sample receipts: {validity_pct:.1f}% ({valid_json_count}/{num_to_log})")
    model.train()


def train_model(model, optimizer, processor, cfg, train_dataloader, eval_dataloader=None, val_dataset=None, params_to_train=None, input_device="cuda:0"):
    logger.info("Start training")
    global_step = 0
    grad_accum_steps = getattr(cfg, "gradient_accumulation_steps", 1)
    use_cuda = "cuda" in str(input_device)

    # Enable GradScaler for fp16 stability (disabled automatically for bf16)
    use_scaler = use_cuda and (cfg.dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if use_scaler:
        logger.info("Enabled GradScaler for FP16 training stability on Tensor Cores")

    optimizer.zero_grad()
    smoothed_loss = None
    last_step_time = time.time()

    for epoch in range(cfg.epochs):
        for idx, batch in enumerate(train_dataloader):
            step_start = time.time()

            with torch.amp.autocast(device_type="cuda", dtype=cfg.dtype, enabled=use_cuda):
                outputs = model(**batch.to(input_device))
                raw_loss = outputs.loss.item()
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()

            # Update moving average loss
            smoothed_loss = raw_loss if smoothed_loss is None else (0.9 * smoothed_loss + 0.1 * raw_loss)

            grad_norm = None
            if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(train_dataloader):
                if params_to_train is not None:
                    scaler.unscale_(optimizer)
                    if hasattr(cfg, "max_grad_norm") and cfg.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(params_to_train, cfg.max_grad_norm).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            step_duration = max(time.time() - step_start, 1e-5)
            samples_per_sec = cfg.batch_size / step_duration

            # Log step metrics every 25 batches
            if idx % 25 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                log_data = {
                    "train/loss": raw_loss,
                    "train/loss_smoothed": smoothed_loss,
                    "train/learning_rate": current_lr,
                    "train/samples_per_sec": samples_per_sec,
                    "train/step_time_ms": step_duration * 1000.0,
                    "epoch": epoch + (idx / len(train_dataloader)),
                }
                if grad_norm is not None:
                    log_data["train/grad_norm"] = grad_norm

                wandb.log(log_data, step=global_step)
                logger.info(f"Epoch: {epoch} Iter: {idx}/{len(train_dataloader)} Loss: {raw_loss:.4f} (smoothed: {smoothed_loss:.4f}) Speed: {samples_per_sec:.2f} samples/s")

        # Validation check at the end of each epoch
        if eval_dataloader is not None:
            val_loss = evaluate_loss(model, eval_dataloader, input_device, cfg.dtype)
            logger.info(f"[EVAL] Epoch: {epoch} Val Loss: {val_loss:.4f}")
            wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

        # Log visual prediction table with sample receipts
        if val_dataset is not None:
            log_validation_predictions(model, processor, cfg, val_dataset, input_device, epoch, global_step, num_samples=4)

        # Local checkpoint save
        ckpt_dir = os.path.join("checkpoints", f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        logger.info(f"Saved intermediate checkpoint to {ckpt_dir}")

    return model


if __name__ == "__main__":
    cfg = Configuration()
    logger.info(f"Configuration: Device={cfg.device}, Dtype={cfg.dtype}, BatchSize={cfg.batch_size}, GradAccum={cfg.gradient_accumulation_steps}")

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    train_dataloader = get_dataloader(processor, cfg)
    eval_dataloader, val_dataset = get_eval_dataloader(processor, cfg)

    # Determine optimal device mapping (multi-GPU auto sharding or single device)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        device_map = "auto"
        input_device = "cuda:0"
        logger.info(f"Detected {num_gpus} GPUs. Using device_map='auto' to shard model across all available GPUs.")
    elif num_gpus == 1:
        device_map = "cuda:0"
        input_device = "cuda:0"
    else:
        device_map = None
        input_device = "cpu"

    logger.info("Loading model with SDPA attention...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    if device_map is None and cfg.device == "cpu":
        model.to("cpu")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    logger.info("Applying LoRA to attention projection layers...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.train()

    # Trainable parameters
    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))

    # Use 8-bit AdamW optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params_to_train, lr=cfg.learning_rate)
        logger.info("Using bitsandbytes 8-bit AdamW optimizer")
    except Exception as e:
        logger.warning(f"Could not use 8-bit AdamW ({e}), falling back to standard AdamW")
        optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)

    torch.cuda.empty_cache()

    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name if hasattr(cfg, "run_name") else None,
        config=vars(cfg),
    )

    train_model(
        model=model,
        optimizer=optimizer,
        processor=processor,
        cfg=cfg,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        val_dataset=val_dataset,
        params_to_train=params_to_train,
        input_device=input_device,
    )

    # Push the checkpoint to hub (merge LoRA weights back so predict.py works standalone)
    logger.info("Merging LoRA weights and pushing to Hub...")
    try:
        merged_model = model.merge_and_unload()
        merged_model.push_to_hub(cfg.checkpoint_id)
    except Exception as e:
        logger.warning(f"Could not merge LoRA weights, pushing PEFT model: {e}")
        model.push_to_hub(cfg.checkpoint_id)

    processor.push_to_hub(cfg.checkpoint_id)

    wandb.finish()
    logger.info("Train finished")
