"""Public configuration API: preset loading and immutable run snapshots."""
from mu4sum.config.loader import OPERATIONS, available_presets, config_fingerprint, load_config
from mu4sum.config.snapshot import ImmutableConfigError, save_run_config

__all__ = ["OPERATIONS", "available_presets", "config_fingerprint", "load_config",
           "ImmutableConfigError", "save_run_config"]
