from dataclasses import dataclass

import torch


def get_optimal_dtype() -> torch.dtype:
    """Auto-detect optimal dtype: bf16 on Ampere+ (capability >= 8), fp16 on Turing (T4), fp32 on CPU."""
    if torch.cuda.is_available():
        major_cc = torch.cuda.get_device_capability()[0]
        # Ampere (RTX 30+, A100, L4) supports native bf16 Tensor Cores
        if major_cc >= 8:
            return torch.bfloat16
        # Turing (T4, RTX 20+) supports fast fp16 Tensor Cores
        return torch.float16
    return torch.float32


@dataclass
class Configuration:
    dataset_id: str = "naver-clova-ix/cord-v2"

    model_id: str = "google/gemma-3-4b-pt"
    checkpoint_id: str = "gemma-3-4b-pt-receipt-extraction"
    project_name: str = "gemma-3-receipt-extraction"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = get_optimal_dtype()

    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-05
    max_grad_norm: float = 1.0

    epochs: int = 2
    max_new_tokens: int = 512
