"""Chat-template helpers shared by the teacher, the training collator and the evaluator."""
from typing import List, Optional, Sequence


def build_messages(prompt: str, n_images: int, system: Optional[str] = None) -> list:
    """Build a chat: optional system turn, then a user turn with image slots followed by the prompt."""
    messages = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": prompt}]
    messages.append({"role": "user", "content": content})
    return messages


def render_prompt(processor, prompt: str, n_images: int, system: Optional[str] = None) -> str:
    """Text with the generation prompt appended (the assistant answer goes right after it)."""
    return processor.apply_chat_template(
        build_messages(prompt, n_images, system), tokenize=False, add_generation_prompt=True
    )


def processor_inputs(processor, texts: List[str], images_per_sample: Sequence[list], nested: bool, **kwargs):
    """Tokenize texts together with their (variable number of) images."""
    flat = [img for imgs in images_per_sample for img in imgs]
    images = list(images_per_sample) if nested else (flat or None)
    return processor(text=texts, images=images, return_tensors="pt", padding=True, **kwargs)
