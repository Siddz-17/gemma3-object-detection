from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    dataset_id: str = "naver-clova-ix/cord-v2"

    model_id: str = "google/gemma-3-4b-pt"
    checkpoint_id: str = "gemma-3-4b-pt-receipt-extraction"
    project_name: str = "gemma-3-receipt-extraction"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    batch_size: int = 4
    learning_rate: float = 2e-05
    epochs = 2
    max_new_tokens: int = 512

