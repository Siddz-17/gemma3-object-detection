import json
from datasets import load_dataset
from config import Configuration
from utils import parse_cord_ground_truth


if __name__ == "__main__":
    cfg = Configuration()
    print(f"[INFO] Loading dataset '{cfg.dataset_id}' from hub...")
    dataset = load_dataset(cfg.dataset_id, split="train")

    sample = dataset[0]
    print(f"[INFO] Sample keys: {list(sample.keys())}")
    print(f"[INFO] Image size: {sample['image'].size}")

    gt_parse = parse_cord_ground_truth(sample["ground_truth"])
    print("[INFO] Parsed and normalized gt_parse sample:")
    print(json.dumps(gt_parse, indent=2, ensure_ascii=False))
