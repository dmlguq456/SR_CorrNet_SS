"""Config name resolution -- maps alias strings to YAML file paths.

Supported variant: "SS" (Speech Separation).

Note:
    This module assumes editable install (``uv sync --extra <cpu|cu126>`` or ``pip install -e .``).
    The returned path string points to the real filesystem location of the YAML file.
    For non-editable (wheel) installs, the Traversable.__str__() may return
    a path inside a zip archive that cannot be opened with plain ``open()``.
    If wheel support is needed in the future, callers should use
    ``importlib.resources.as_file()`` context manager around the full
    config-loading lifecycle instead.
"""
import importlib.resources

_VARIANT_MAP = {
    "SS": "sr_corrnet.models.SR_CorrNet_SS",
}


def resolve_config(variant: str, config_name: str) -> str:
    """Resolve a config alias to an absolute YAML file path.

    Args:
        variant: Must be "SS".
        config_name: YAML filename (e.g. "1ch_WSJ_fix_2spk.yaml").

    Returns:
        Absolute path to the YAML config file (valid for editable installs).

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If variant is not recognized.
    """
    package = _VARIANT_MAP[variant]
    ref = importlib.resources.files(package).joinpath("configs", config_name)
    if not ref.is_file():
        raise FileNotFoundError(f"Config not found: {variant}/{config_name}")
    return str(ref)
