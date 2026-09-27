import os
from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import Configuration
from utils import parse_model_output, save_receipt_prediction, test_collate_function

os.makedirs("outputs", exist_ok=True)


def get_dataloader(processor):
    test_dataset = load_dataset(cfg.dataset_id, split="test")
    test_collate_fn = partial(
        test_collate_function, processor=processor, dtype=cfg.dtype
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=test_collate_fn
    )
    return test_dataloader


if __name__ == "__main__":
    cfg = Configuration()
    processor = AutoProcessor.from_pretrained(cfg.checkpoint_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.checkpoint_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
    )
    model.eval()
    model.to(cfg.device)

    test_dataloader = get_dataloader(processor=processor)
    sample, sample_images = next(iter(test_dataloader))
    sample = sample.to(cfg.device)

    generation = model.generate(**sample, max_new_tokens=cfg.max_new_tokens)
    decoded = processor.batch_decode(generation, skip_special_tokens=True)

    file_count = 0
    for output_text, sample_image in zip(decoded, sample_images):
        image = sample_image[0]
        extracted_data = parse_model_output(output_text)
        save_receipt_prediction(
            image,
            extracted_data,
            image_path=f"outputs/output_{file_count}.png",
            json_path=f"outputs/output_{file_count}.json",
        )
        file_count += 1
