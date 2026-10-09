import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mu4sum.config import load_config


@pytest.fixture
def raw_dataset(tmp_path):
    """Fake IU-X-ray-like raw folder: 20 studies, some with frontal+lateral views, one image per CSV row."""
    raw = tmp_path / "source" / "320"
    raw.mkdir(parents=True)
    rows = []
    for i in range(20):
        views = ["Frontal", "Lateral"] if i % 2 == 0 else ["Frontal"]
        for v in reversed(views):  # reversed on purpose: loader must reorder frontal first
            name = f"{i}_{v}.png"
            Image.fromarray(np.full((8, 8), i * 10 % 255, dtype=np.uint8)).save(raw / name)
            rows.append({"uid": i, "filename": name, "projection": v,
                         "findings": f"Findings of study {i}. Heart normal. Lungs clear.",
                         "impression": f"No acute disease {i}."})
    pd.DataFrame(rows).to_csv(tmp_path / "source" / "cleaned_dataset.csv", index=False)
    return tmp_path / "source"


@pytest.fixture
def make_cfg(tmp_path, raw_dataset):
    def _make(operation="preprocess", **kwargs):
        overrides = [
            "dataset.source.type=local",
            f"+dataset.source.path={raw_dataset.as_posix()}",
            f"runtime.data_root={(tmp_path / 'data').as_posix()}",
            f"runtime.output_root={(tmp_path / 'outputs').as_posix()}",
            *kwargs.pop("overrides", []),
        ]
        kwargs.setdefault("dataset", "iu_xray")
        if operation != "preprocess":
            kwargs.setdefault("model", "qwen2_vl_7b")
        elif "model" not in kwargs:
            kwargs["model"] = "qwen2_vl_7b"
        return load_config(operation, overrides=overrides, **kwargs)

    return _make
