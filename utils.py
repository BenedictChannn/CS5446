
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box, Discrete
from IPython.display import clear_output


class DictToListWrapper(gym.Wrapper):
    """
    The wrapper to convert Dict observation space to List observation space.
    """

    def __init__(self, env: gym.Env[dict[str, object], dict[str, object]]) -> None:
        super().__init__(env)
        # assert isinstance(env.observation_space,
        #                   gym.spaces.Dict), "The observation space must be of type gym.spaces.Dict"
        assert isinstance(env.action_space, gym.spaces.Dict), (
            "The action space must be of type gym.spaces.Dict"
        )

        # transform the observation space to a Box space
        self.env_features: list[str] = list(env.observation_space.spaces.keys())
        self.observation_space = Box(
            low=-float("inf"), high=float("inf"), shape=(len(self.env_features),), dtype=np.float32
        )
        # transform the action space to a Discrete space,
        # as each action dim is a Discrete space, the size of the Discrete space is the sum of the sizes of each action dim
        # e.g., if action space is {'a': Discrete(2), 'b': Discrete(2)}, then the new action space is Discrete(4),
        # means, 0 -> {'a': 0}, 1 -> {'a': 1}, 2 -> {'b': 0}, 3 -> {'b': 1}
        action_mapping: dict[int, dict[str, np.int64]] = {}
        action_id = 0
        for key, value in env.action_space.spaces.items():
            for i in range(value.n):
                action_mapping[action_id] = {key: np.int64(i)}
                action_id += 1

        self.action_space = Discrete(action_id)
        self.action_mapping: dict[int, dict[str, np.int64]] = action_mapping

    def reset(self, **kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        state, info = super().reset(**kwargs)
        state = self.convert_state_dict2list(state)
        return state, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        env_action = self.convert_action_id2dict(action)
        state, reward, done, truncated, info = super().step(env_action)
        state = self.convert_state_dict2list(state)
        return state, reward, done, truncated, info

    def convert_state_dict2list(self, state_dict: dict[str, object]) -> np.ndarray:
        out = []
        for key in self.env_features:
            v = state_dict.get(key, 0)
            if isinstance(v, (bool, np.bool_)):
                out.append(int(v))
            elif isinstance(v, (int, np.integer)):
                out.append(int(v))
            else:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(0.0)
        return np.array(out, dtype=np.float32)

    def convert_action_id2dict(self, action: int) -> dict[str, np.int64]:
        return self.action_mapping[action]

    def get_state_description(self) -> None:
        print("State description:")
        for f in range(len(self.env_features)):
            print(f"state dim {f}: {self.env_features[f]}")

    def get_action_description(self) -> None:
        print("Action description:")
        for k, v in self.action_mapping.items():
            print(f"Action {k}: {v}")


# ---


def exponential_smoothing(data: list[float] | np.ndarray, alpha: float = 0.1) -> list[float]:
    """Compute exponential smoothing."""
    smoothed = [data[0]]  # Initialize with the first data point
    for i in range(1, len(data)):
        st = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(st)
    return smoothed


def live_plot(
    data_dict: dict[str, list[float]],
    save_pdf: bool = False,
    output_file: str = "training_curves.pdf",
) -> None:
    """Plot the live graph with multiple subplots."""

    plt.style.use("ggplot")
    n_plots = len(data_dict)
    fig, axes = plt.subplots(nrows=n_plots, figsize=(7, 4 * n_plots), squeeze=False)
    plt.subplots_adjust(hspace=0.5)
    plt.ion()
    clear_output(wait=True)

    for ax, (label, data) in zip(axes.flatten(), data_dict.items(), strict=True):
        ax.clear()
        ax.plot(data, label=label, color="yellow", linestyle="--")
        # Compute and plot moving average for total reward
        if len(data) > 0:
            ma = exponential_smoothing(data)
            ma_idx_start = len(data) - len(ma)
            ax.plot(
                range(ma_idx_start, len(data)),
                ma,
                label="Smoothed Value",
                linestyle="-",
                color="purple",
                linewidth=2,
            )
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc="upper left")

    if not save_pdf:
        plt.show()
    else:
        plt.savefig(output_file)


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