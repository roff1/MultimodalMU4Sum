"""Environment-agnostic raw-data download. Add a source type by registering one function."""
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from mu4sum.config.paths import raw_dir
from mu4sum.utils.io import read_json, write_json_atomic
from mu4sum.utils.registry import Registry
from mu4sum.utils.reproducibility import fingerprint

DOWNLOADERS = Registry("downloader")
SOURCE_FILE = "_source.json"


@DOWNLOADERS.register("kaggle")
def _download_kaggle(source: DictConfig, raw_dir: Path) -> Path:
    import os

    # Keep the dataset next to our data root instead of ~/.cache (matters on Colab/Drive).
    os.environ.setdefault("KAGGLEHUB_CACHE", str(raw_dir / "_kagglehub"))
    import kagglehub
    return Path(kagglehub.dataset_download(source.handle))


@DOWNLOADERS.register("huggingface")
def _download_hf(source: DictConfig, raw_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=source.handle, repo_type="dataset", local_dir=raw_dir / "hf"))


@DOWNLOADERS.register("url")
def _download_url(source: DictConfig, raw_dir: Path) -> Path:
    target = raw_dir / "url"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / Path(source.url.split("?")[0]).name
    if not archive.exists():
        urllib.request.urlretrieve(source.url, archive)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(target)
    return target


@DOWNLOADERS.register("local")
def _use_local(source: DictConfig, raw_dir: Path) -> Path:
    path = Path(source.path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"dataset.source.path does not exist: {path}")
    return path


def _to_stored(root: Path, data_root: Path) -> str:
    """Path relative to data_root when inside it (portable across machines), else absolute."""
    try:
        return str(root.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(root.resolve())


def raw_root(raw_dir: Path, data_root: Path) -> Optional[Path]:
    """Folder the downloaded files live in, as recorded by `download_dataset` (None if not downloaded)."""
    marker = raw_dir / SOURCE_FILE
    if not marker.exists():
        return None
    root = Path(read_json(marker)["root"])
    return root if root.is_absolute() else Path(data_root) / root


def image_root(cfg: DictConfig) -> Path:
    """Folder the records' relative image paths refer to: the raw root of this machine's download."""
    root = raw_root(raw_dir(cfg), Path(cfg.runtime.data_root))
    if root is None:
        raise FileNotFoundError("Raw data not downloaded yet: run the 'download' stage first.")
    return root


def download_dataset(cfg: DictConfig, raw_dir: Path, force: bool = False) -> Path:
    """Fetch raw data unless a matching download is already recorded. Returns the raw root."""
    data_root = Path(cfg.runtime.data_root)
    source = cfg.dataset.source
    fp = fingerprint(OmegaConf.to_container(source))
    marker = raw_dir / SOURCE_FILE

    if marker.exists() and not force: # Skip download if files exist and configuration matches (unless forced).
        info = read_json(marker)
        existing = raw_root(raw_dir, data_root)
        if info.get("fingerprint") != fp:
            raise RuntimeError(f"Raw data in {raw_dir} was downloaded from a different source; "
                               f"re-run with --force to download again.")
        if existing is not None and existing.exists():
            print(f"[Download] Raw data already present at {existing}, skipping.")
            return existing

    if force and raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Download] Fetching '{source.type}' source...")
    root = DOWNLOADERS.get(source.type)(source, raw_dir)
    write_json_atomic(marker, {"fingerprint": fp, "root": _to_stored(root, data_root),
                               "source": OmegaConf.to_container(source)})
    print(f"[Download] Raw data at {root}")
    return root
