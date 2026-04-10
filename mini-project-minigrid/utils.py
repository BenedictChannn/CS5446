import numpy as np
import minigrid
import gymnasium as gym
from matplotlib import pyplot as plt
from typing import Literal

FigureSize = Literal["small", "medium", "large"]

SIZE_MAP = {
    "small": (2, 2),
    "medium": (4, 4),
    "large": (8, 8),
}

def visualize_frame(frame: np.ndarray, label: str, size: FigureSize = "medium"):
    figsize = SIZE_MAP[size]
    plt.figure(figsize=figsize)
    plt.imshow(frame)
    plt.axis("off")
    plt.title(label)
    plt.show()

def make_env(env_id: str):
    return gym.make(env_id, render_mode="rgb_array")

def visualize_env(env: gym.Env, label: str):
    obs, _ = env.reset()
    frame = env.render()
    env.close()
    visualize_frame(frame, label)



# ---


import base64
import io
import inspect
import pickle
from typing import Any, Literal, Optional

# Optional imports only used when generating PyTorch snippets (runtime still needs torch)
try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None     # type: ignore

Compression = Literal["zlib", "gzip", "bz2", "lzma", "none"]


def _compress_to_b64(data: bytes, compression: Compression, level: int) -> tuple[str, str, str]:
    """
    Compress bytes and return:
      - base64 string of compressed bytes
      - decomp_loader_code: Python code for the generated snippet to decompress
      - comp_name: normalized compression name
    """
    comp = (compression or "zlib").lower()
    if comp not in {"zlib", "gzip", "bz2", "lzma", "none"}:
        comp = "zlib"

    if comp == "zlib":
        import zlib
        raw = zlib.compress(data, level=level)
        decomp_code = "import zlib as _z; _decomp = _z.decompress"
    elif comp == "gzip":
        import gzip
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=level) as f:
            f.write(data)
        raw = buf.getvalue()
        decomp_code = "import gzip as _gz, io as _io; _decomp = lambda b: _gz.GzipFile(fileobj=_io.BytesIO(b), mode='rb').read()"
    elif comp == "bz2":
        import bz2
        lvl = min(max(level, 1), 9)
        raw = bz2.compress(data, compresslevel=lvl)
        decomp_code = "import bz2 as _bz2; _decomp = _bz2.decompress"
    elif comp == "lzma":
        import lzma
        raw = lzma.compress(data, preset=min(max(level, 0), 9))
        decomp_code = "import lzma as _lz; _decomp = _lz.decompress"
    else:  # none
        raw = data
        decomp_code = "_decomp = (lambda b: b)"

    b64 = base64.b64encode(raw).decode("ascii")
    return b64, decomp_code, comp


def _has_noarg_constructor(cls: type) -> bool:
    try:
        sig = inspect.signature(cls)
        params = list(sig.parameters.values())[1:]  # skip self
        return all(
            p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.default is not p.empty
            for p in params
        )
    except Exception:
        return False


def generate_torch_loader_snippet(
    model: "nn.Module",
    prefer: Literal["pickle", "state_dict"] = "state_dict",
    compression: Compression = "zlib",
    level: int = 9,
) -> str:
    """
    Create a copy-pasteable get_model() code string that reconstructs the given PyTorch model.

    Args:
        model: An instantiated torch.nn.Module (instance, not a class).
        prefer: Preferred serialization method ("pickle" or "state_dict").
        compression: Compression algorithm.
        level: Compression level.
    
    Returns:
        Python source string defining get_model(device="cpu", dtype=None).
    
    Raises:
        RuntimeError: If PyTorch is not available in the current environment.
        TypeError: If 'model' is not an instantiated nn.Module.

    Security:
        - Pickle-based payloads can execute code on load if misused;
          the generated loader attempts safe_globals first but may fallback to trusted loading.
    """
    if prefer == "pickle":
        return generate_torch_loader_snippet_with_pickle(model, compression, level)
    else:
        return generate_torch_loader_snippet_with_state_dict(model, compression, level)


def generate_torch_loader_snippet_with_pickle(
    model: "nn.Module",
    compression: Compression = "zlib",
    level: int = 9,
) -> str:
    """
    Create a copy-pasteable get_model() code string that reconstructs the given PyTorch model
    using only full-model pickle serialization.

    Args:
        model: An instantiated torch.nn.Module (instance, not a class).
        compression: Compression algorithm.
        level: Compression level.
    
    Returns:
        Python source string defining get_model(device="cpu", dtype=None).
    
    Raises:
        RuntimeError: If PyTorch is not available in the current environment.
        TypeError: If 'model' is not an instantiated nn.Module.

    Security:
        - Pickle-based payloads can execute code on load if misused;
          the generated loader attempts safe_globals first but may fallback to trusted loading.
    """
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is not available in this environment.")
    if not isinstance(model, nn.Module):
        raise TypeError("Expected an instantiated torch.nn.Module (instance), not a class.")

    # Full model pickle
    full_bytes = _dump_full_pickle_bytes(model)
    if full_bytes is not None:
        b64, decomp_code, comp_name = _compress_to_b64(full_bytes, compression, level)
        module_name = model.__class__.__module__
        class_name = model.__class__.__name__
        return _render_full_pickle_loader(b64, decomp_code, comp_name, module_name, class_name)

    raise RuntimeError("Failed to serialize the model using full-model pickle.")


def generate_torch_loader_snippet_with_state_dict(
    model: "nn.Module",
    compression: Compression = "zlib",
    level: int = 9,
) -> str:
    """
    Create a copy-pasteable get_model() code string that reconstructs the given PyTorch model
    using only state_dict serialization.

    Args:
        model: An instantiated torch.nn.Module (instance, not a class).
        compression: Compression algorithm.
        level: Compression level.
    
    Returns:
        Python source string defining get_model(device="cpu", dtype=None).
    
    Raises:
        RuntimeError: If PyTorch is not available in the current environment.
        TypeError: If 'model' is not an instantiated nn.Module.

    Security:
        - Pickle-based payloads can execute code on load if misused;
          the generated loader attempts safe_globals first but may fallback to trusted loading.
    """
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is not available in this environment.")
    if not isinstance(model, nn.Module):
        raise TypeError("Expected an instantiated torch.nn.Module (instance), not a class.")

    # state_dict
    sd_bytes = _dump_state_dict_bytes(model)
    b64, decomp_code, comp_name = _compress_to_b64(sd_bytes, compression, level)
    zero_arg_ok = _has_noarg_constructor(model.__class__)
    module_name = model.__class__.__module__
    class_name = model.__class__.__name__
    return _render_state_dict_loader(b64, decomp_code, comp_name, module_name, class_name, zero_arg_ok)


# ----- PyTorch generator internals -----


def _dump_full_pickle_bytes(model: "nn.Module") -> Optional[bytes]:
    try:
        buf = io.BytesIO()
        torch.save(model, buf)
        return buf.getvalue()
    except Exception:
        return None


def _dump_state_dict_bytes(model: "nn.Module") -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def _render_full_pickle_loader(
    b64: str,
    decomp_code: str,
    comp_name: str,
    module_name: str,
    class_name: str,
) -> str:
    return f'''\
def get_model(device: str = "cpu", dtype: str | None = None):
    """
    Return the original PyTorch model loaded from an embedded, base64-encoded {'compressed ' if comp_name!='none' else ''}pickle.

    Notes:
      - The original model class should be importable (module: "{module_name}", class: "{class_name}").
      - PyTorch >= 2.6: torch.load defaults to weights_only=True.
        This loader will:
          1) Try to import the class and allowlist it via torch.serialization.safe_globals.
          2) Fall back to weights_only=False (ONLY if you trust this source).

    Args:
        device: Where to map the model (e.g., "cpu", "cuda:0").
        dtype: Optional dtype (string like "float32" or torch.dtype).
    """
    import base64, io, importlib, torch
    {decomp_code}
    _blob_b64 = "{b64}"
    _raw = _decomp(base64.b64decode(_blob_b64))

    # Try to import the class for safe allowlisting
    try:
        mod = importlib.import_module("{module_name}")
        cls = getattr(mod, "{class_name}", None)
    except Exception:
        cls = None

    # Attempt safe load first
    try:
        if cls is not None:
            with torch.serialization.safe_globals([cls]):
                m = torch.load(io.BytesIO(_raw), map_location=device)
        else:
            # Class not importable; last resort: trusted load
            m = torch.load(io.BytesIO(_raw), map_location=device, weights_only=False)
    except Exception:
        # Final fallback: trusted load
        m = torch.load(io.BytesIO(_raw), map_location=device, weights_only=False)

    if dtype is not None:
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        m = m.to(dtype=dt)
    m.eval()
    return m
'''


def _render_state_dict_loader(
    b64: str,
    decomp_code: str,
    comp_name: str,
    module_name: str,
    class_name: str,
    zero_arg_ok: bool,
) -> str:
    ctor = f"{class_name}()" if zero_arg_ok else f"{class_name}(# TODO: fill constructor args)"
    return f'''\
def get_model(device: str = "cpu", dtype: str | None = None):
    """
    Return a PyTorch model by instantiating the class and loading an embedded state_dict
    from a base64-encoded {'compressed ' if comp_name!='none' else ''}blob.

    Requirements:
      - The model class must be importable (module: "{module_name}", class: "{class_name}").
      - If the constructor needs arguments, fill them in where indicated.

    Args:
        device: Where to map the tensors (e.g., "cpu", "cuda:0").
        dtype: Optional dtype (string or torch.dtype).
    """
    import base64, io, importlib, torch
    {decomp_code}
    mod = importlib.import_module("{module_name}")
    cls = getattr(mod, "{class_name}")
    model = {ctor}
    sd = torch.load(io.BytesIO(_decomp(base64.b64decode("{b64}"))), map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("Warning: load_state_dict mismatches. Missing:", missing, "Unexpected:", unexpected)
    if dtype is not None:
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        model = model.to(dtype=dt)
    model.to(device)
    model.eval()
    return model
'''