"""Batch collation for the VLM: variable number of images per sample, answer-only loss."""
from mu4sum.models.chat import processor_inputs, render_prompt


class SFTCollator:
    """Tokenize prompt+target samples into a batch whose labels mask padding and the prompt."""

    def __init__(self, processor, nested_images: bool = False):
        self.processor = processor
        self.nested = nested_images
        self.eos = processor.tokenizer.eos_token or ""
        processor.tokenizer.padding_side = "right"  # prompt length below relies on right padding

    def __call__(self, samples):
        """Return a model batch with `labels` set to -100 outside the target (and EOS) tokens."""
        images =[s["images"] for s in samples]
        prompts = [render_prompt(self.processor, s["prompt"], len(s["images"])) for s in samples]
        full = [p + s["target"] + self.eos for p, s in zip(prompts, samples)]

        prompt_len = processor_inputs(self.processor, prompts, images, self.nested)["attention_mask"].sum(dim=1)
        batch = processor_inputs(self.processor, full, images, self.nested)

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, n in enumerate(prompt_len.tolist()):
            labels[i, :n] = -100
        batch["labels"] = labels
        return batch


class TokenWiseCollator:
    """SFT batches plus `token_kind`: +1 on answer tokens outside the forget aspects (retain blocks, separators,
    EOS), -1 on tokens of the forget-aspect blocks, 0 everywhere else (prompt, images, padding)."""

    def __init__(self, base: SFTCollator, forget_aspects):
        tokenizer = base.processor.tokenizer
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Token-wise methods need a fast tokenizer (character offsets of the answer tokens)")
        self.base, self.tokenizer, self.forget = base, tokenizer, set(forget_aspects)

    def kinds(self, sample) -> list:
        """Token kinds (+1 / -1) of the answer tokens (target + EOS) of one sample."""
        text = sample["target"] + self.base.eos
        offsets = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        spans = [(a, b) for aspect, a, b in sample["spans"] if aspect in self.forget]
        return [-1 if end > start and any(a <= end - 1 < b for a, b in spans) else 1 for start, end in offsets]

    def __call__(self, samples):
        """Return the SFT batch with a `token_kind` tensor aligned to `input_ids`/`labels`."""
        import torch

        batch = self.base(samples)
        kind = torch.zeros_like(batch["labels"])
        for i, sample in enumerate(samples):
            answer = batch["labels"][i] != -100
            kinds = self.kinds(sample)
            if int(answer.sum()) != len(kinds):
                raise ValueError(f"Record {sample['id']}: the answer has {int(answer.sum())} tokens in the batch "
                                 f"but {len(kinds)} when tokenized alone; token-wise labels would be misaligned")
            kind[i, answer] = torch.tensor(kinds, dtype=kind.dtype)
        batch["token_kind"] = kind
        return batch


class PairedCollator:
    """Splits a mixed forget/retain batch (samples tagged with `_side`) into two model batches."""

    def __init__(self, base_collate):
        self.base = base_collate

    def __call__(self, samples):
        """Return {"forget": batch, "retain": batch}; a side with no samples is None."""
        out = {}
        for side in ("forget", "retain"):
            group = [s for s in samples if s["_side"] == side]
            out[side] = self.base(group) if group else None
        return out
