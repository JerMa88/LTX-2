import json
import struct

import safetensors
import torch

from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.sd_ops import SDOps

SAFE_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
}


class SafetensorsStateDictLoader(StateDictLoader):
    """
    Loads weights from safetensors files without metadata support.
    Use this for loading raw weight files. For model files that include
    configuration metadata, use SafetensorsModelStateDictLoader instead.
    """

    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def load(self, path: str | list[str], sd_ops: SDOps, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        without memory-mapping full files to avoid Windows virtual address commit exhaustion.
        """
        sd = {}
        size = 0
        dtype = set()
        device = device or torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]
        for shard_path in model_paths:
            with open(shard_path, "rb") as f:
                (header_len,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(header_len).decode("utf-8"))
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    pt_dtype = SAFE_DTYPES.get(info["dtype"])
                    if pt_dtype is None:
                        raise ValueError(f"Unsupported safetensors dtype {info['dtype']} for tensor {name}")
                    shape = info["shape"]
                    start, end = info["data_offsets"]
                    nbytes = end - start
                    f.seek(8 + header_len + start)
                    tensor = torch.empty(shape, dtype=pt_dtype, device="cpu")
                    read_bytes = f.readinto(tensor.reshape(-1).view(torch.uint8).numpy())
                    if read_bytes != nbytes:
                        raise IOError(f"Short read for tensor {name}: expected {nbytes}, got {read_bytes}")
                    value = tensor.to(device=device) if str(device) != "cpu" else tensor
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value

        return StateDict(sd=sd, device=device, size=size, dtype=dtype)

def read_safetensors_header(path: str) -> tuple[dict, list[str]]:
    """Read safetensors JSON header directly without mapping file into memory."""
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        (header_len,) = struct.unpack("<Q", length_bytes)
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)
        metadata = header.get("__metadata__", {})
        parsed_meta = {}
        for k, v in metadata.items():
            try:
                parsed_meta[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                parsed_meta[k] = v
        keys = [k for k in header if k != "__metadata__"]
        return parsed_meta, keys


class SafetensorsModelStateDictLoader(StateDictLoader):
    """
    Loads weights and configuration metadata from safetensors model files.
    Unlike SafetensorsStateDictLoader, this loader can read model configuration
    from the safetensors file metadata via the metadata() method.
    """

    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None):
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader()

    def metadata(self, path: str) -> dict:
        """Read the full safetensors ``__metadata__`` dict directly from header."""
        return read_safetensors_header(path)[0]

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        return self.weight_loader.load(path, sd_ops, device)
