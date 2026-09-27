import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import wandb
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import LoraConfig, get_peft_model

from config import Configuration
from utils import train_collate_function

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_dataloader(processor):
    logger.info("Fetching the dataset")
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=cfg.dtype
    )

    logger.info("Building data loader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )
    return train_dataloader


def train_model(model, optimizer, cfg, train_dataloader):
    logger.info("Start training")
    global_step = 0
    grad_accum_steps = getattr(cfg, "gradient_accumulation_steps", 1)
    optimizer.zero_grad()

    for epoch in range(cfg.epochs):
        for idx, batch in enumerate(train_dataloader):
            outputs = model(**batch.to(model.device))
            raw_loss = outputs.loss.item()
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if idx % 100 == 0:
                logger.info(f"Epoch: {epoch} Iter: {idx} Loss: {raw_loss:.4f}")
                wandb.log({"train/loss": raw_loss, "epoch": epoch}, step=global_step)

    return model


if __name__ == "__main__":
    cfg = Configuration()
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    train_dataloader = get_dataloader(processor)

    logger.info("Loading model with SDPA attention...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
        attn_implementation="sdpa",
    )

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
    model.to(cfg.device)

    # Credits to Sayak Paul for this beautiful expression
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

    train_model(model, optimizer, cfg, train_dataloader)

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
