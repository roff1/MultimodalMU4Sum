"""Loading of any Hugging Face vision-language model from a model preset (YAML only).

The model class is chosen by `model.auto_class` (e.g. AutoModelForImageTextToText), so adding a new
VLM whose processor ships a chat template needs no code change. Hardware is handled here: 4-bit
quantization is applied only when CUDA and bitsandbytes are present, otherwise it falls back cleanly.
"""
import importlib.util
import warnings
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from mu4sum.utils.environment import get_device


def _torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def load_processor(model_cfg: DictConfig):
    """Load the AutoProcessor for the preset, passing `processor.kwargs` when configured."""
    from transformers import AutoProcessor

    kwargs = OmegaConf.to_container(model_cfg.processor.kwargs) if model_cfg.get("processor") else {}
    return AutoProcessor.from_pretrained(model_cfg.name, **kwargs)


def load_model(model_cfg: DictConfig, adapter_dir: Optional[Path] = None, trainable_adapter: bool = False):
    """Load the base VLM (4-bit when possible) and optionally a saved LoRA adapter on top.

    `trainable_adapter` keeps the loaded adapter weights trainable for further training."""
    import torch
    import transformers

    device = get_device()
    kwargs = {"torch_dtype": _torch_dtype(model_cfg.torch_dtype)}

    quantize = bool(model_cfg.load_in_4bit)
    if quantize and (device != "cuda" or importlib.util.find_spec("bitsandbytes") is None):
        warnings.warn("load_in_4bit requested but CUDA/bitsandbytes unavailable: loading without quantization.")
        quantize = False
    if quantize:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch_dtype(model_cfg.compute_dtype),
        )
    if device == "cuda":
        kwargs["device_map"] = model_cfg.device_map
    elif device == "cpu":
        kwargs["torch_dtype"] = torch.float32  # half precision on CPU is slow and poorly supported

    model = getattr(transformers, model_cfg.auto_class).from_pretrained(model_cfg.name, **kwargs)
    if device != "cuda":
        model.to(device)

    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=trainable_adapter)
    return model


def attach_lora(model, lora_cfg: DictConfig):
    """Wrap the model with a new LoRA adapter, preparing it for k-bit training if loaded in 4-bit."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)
