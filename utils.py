import json
import os
import re
from typing import Any, Dict, Union
from PIL import Image


def normalize_cord_parse(data: Any) -> Any:
    """Normalize CORD-v2 gt_parse so singleton items are consistently formatted as lists."""
    if not isinstance(data, dict):
        return data

    normalized = dict(data)
    # CORD-v2 "menu" is a list of line items; normalize single dict to list
    if "menu" in normalized:
        if isinstance(normalized["menu"], dict):
            normalized["menu"] = [normalized["menu"]]
        elif isinstance(normalized["menu"], list):
            normalized_menu = []
            for item in normalized["menu"]:
                if isinstance(item, dict):
                    item_copy = dict(item)
                    if "sub_menu" in item_copy and isinstance(item_copy["sub_menu"], dict):
                        item_copy["sub_menu"] = [item_copy["sub_menu"]]
                    normalized_menu.append(item_copy)
                else:
                    normalized_menu.append(item)
            normalized["menu"] = normalized_menu

    return normalized


def parse_cord_ground_truth(ground_truth_str: str) -> Dict[str, Any]:
    """Parse CORD-v2 ground_truth JSON string, extract gt_parse, and normalize singletons."""
    try:
        data = json.loads(ground_truth_str)
        if isinstance(data, dict):
            gt_parse = data.get("gt_parse", data)
        else:
            gt_parse = data
        return normalize_cord_parse(gt_parse)
    except Exception:
        return {"raw": ground_truth_str}


def parse_model_output(output_text: str) -> Union[Dict[str, Any], str]:
    """Parse generated model text into a JSON dict, falling back to raw text if parsing fails."""
    text = output_text.strip()
    # Handle markdown ```json ... ``` codeblocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1]

    try:
        return json.loads(text)
    except Exception:
        return output_text.strip()


def save_receipt_prediction(image: Image.Image, prediction_data: Any, image_path: str, json_path: str):
    """Save the receipt image and a sidecar JSON file with the extracted fields."""
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path)
    with open(json_path, "w", encoding="utf-8") as f:
        if isinstance(prediction_data, (dict, list)):
            json.dump(prediction_data, f, indent=2, ensure_ascii=False)
        else:
            f.write(str(prediction_data))


def train_collate_function(batch_of_samples, processor, dtype):
    images = []
    prompts = []
    for sample in batch_of_samples:
        gt_parse = parse_cord_ground_truth(sample["ground_truth"])
        target_text = json.dumps(gt_parse, ensure_ascii=False)
        images.append([sample["image"]])
        prompts.append(
            f"{processor.tokenizer.boi_token} extract \n\n{target_text} {processor.tokenizer.eos_token}"
        )

    batch = processor(images=images, text=prompts, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()

    # List from https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    batch["pixel_values"] = batch["pixel_values"].to(dtype)
    return batch


def test_collate_function(batch_of_samples, processor, dtype):
    images = []
    prompts = []
    for sample in batch_of_samples:
        images.append([sample["image"]])
        prompts.append(f"{processor.tokenizer.boi_token} extract \n\n")

    batch = processor(images=images, text=prompts, return_tensors="pt", padding=True)
    batch["pixel_values"] = batch["pixel_values"].to(dtype)
    return batch, images
