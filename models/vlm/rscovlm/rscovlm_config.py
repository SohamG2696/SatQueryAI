"""
RSCoVLM-7B-2512-SatQuery Configuration and Metadata.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RSCoVLMConfig:
    """Configuration settings for fine-tuned RSCoVLM model."""

    model_name: str = "RSCoVLM-7B-2512-SatQuery"
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_name: str = "RSCoVLM-7B-2512-SatQuery-LoRA"

    # Model Paths
    base_model_path: str = ""
    adapter_path: str = ""

    # LoRA Hyperparameters
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Generation Parameters
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.90
    do_sample: bool = False

    # Inference Execution Modes
    device: str = "cuda"
    use_half_precision: bool = True
    remote_enabled: bool = False
    remote_url: Optional[str] = None
