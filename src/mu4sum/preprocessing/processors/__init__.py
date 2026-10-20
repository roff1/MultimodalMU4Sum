"""Raw dataset -> list of records. Every dataset has its own processor, named after it.

A processor is a function `(dataset_cfg, raw_root) -> list[dict]` registered with
`@PROCESSORS.register("<dataset name>")` in its own module `processors/<dataset name>.py` (modules of this package
are imported automatically). It reads whatever format the dataset ships in, applies `dataset.filtering`
(see `common.rejection_reason`) and returns one record per sample:

    id: str                  # unique within the dataset
    image_types: list[str]   # one entry per image ("" when the dataset has no notion of image type)
    image_paths: list[str]   # relative to raw_root
    text: str                # the textual input of the sample

The pipeline then adds the `split` of each record (`build_records`). When the dataset has no augmentation
(`augmentation.enabled: false`) the processor also returns `aspect_summaries` (aspect name -> summary, for every
aspect of `aspects.names`), since no later stage will produce them.
"""
import importlib
import pkgutil
from pathlib import Path
from typing import List

from omegaconf import DictConfig

from mu4sum.preprocessing.processors.common import PROCESSORS, assign_splits, check_records

for _module in pkgutil.iter_modules(__path__):
    if _module.name != "common":
        importlib.import_module(f"{__name__}.{_module.name}")


def build_records(cfg: DictConfig, raw_root: Path) -> List[dict]:
    """Run the dataset's registered processor, check its records, then assign each record to a split."""
    name = cfg.dataset.processor
    records = PROCESSORS.get(name)(cfg.dataset, raw_root)
    check_records(records, name, None if cfg.augmentation.enabled else list(cfg.aspects.names))
    return assign_splits(records, cfg.dataset.splits, cfg.seed)
