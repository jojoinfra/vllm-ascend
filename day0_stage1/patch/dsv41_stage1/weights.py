"""Load the frozen V4.1 HF checkpoint directly from its original shards."""

import argparse
import gzip
import hashlib
import json
import math
import re
import struct
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


# The release index fixes *all* accepted names and source shards, including those
# excluded from this text-only run. Its provenance is the frozen HF model snapshot.
INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
CHECKPOINT_WIDTH = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
EXPERT = re.compile(r"(?:layers|mtp)\.\d+\.ffn\.experts\.(\d+)\.")
SHARD = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")


@dataclass(frozen=True, slots=True)
class TensorRef:
    file: Path
    offset: int
    shape: tuple
    dtype: str
    nbytes: int


def read_checkpoint(checkpoint_dir):
    root = Path(checkpoint_dir)
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != INDEX_SHA256:
        raise ValueError("Checkpoint index does not match the frozen HF revision")
    index = json.loads(index_bytes)
    weight_map = index["weight_map"]
    tensors = {}
    header_hashes = {}
    for basename in sorted(set(weight_map.values())):
        if not SHARD.fullmatch(basename):
            raise ValueError(f"Invalid shard basename {basename!r}")
        shard = root / basename
        with shard.open("rb") as stream:
            size, = struct.unpack("<Q", stream.read(8))
            if not 0 < size <= 100_000_000:
                raise ValueError(f"Invalid safetensors header size in {basename}")
            header_bytes = stream.read(size)
        header_hashes[basename] = hashlib.sha256(header_bytes).hexdigest()
        header = json.loads(header_bytes)
        header.pop("__metadata__", None)
        payload = 8 + size
        cursor = 0
        for name, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            if weight_map.get(name) != basename or name in tensors:
                raise ValueError(f"Misindexed or duplicate key {name}")
            start, end = entry["data_offsets"]
            shape = tuple(entry["shape"])
            dtype = entry["dtype"]
            if start != cursor or min(shape) < 1 or end - start != math.prod(shape) * CHECKPOINT_WIDTH[dtype]:
                raise ValueError(f"Invalid byte range for {name}")
            tensors[name] = TensorRef(shard, payload + start, shape, dtype, end - start)
            cursor = end
        if shard.stat().st_size != payload + cursor:
            raise ValueError(f"Truncated shard or trailing payload in {basename}")
    if weight_map.keys() != tensors.keys():
        raise ValueError("Checkpoint is missing indexed tensors")
    if sum(t.nbytes for t in tensors.values()) != index["metadata"]["total_size"]:
        raise ValueError("Indexed payload size mismatch")
    return tensors, header_hashes


def classify(name):
    if name.startswith("mtp."):
        return "dspark"
    if (name.startswith(("vision.", "aligner.", "image_"))
            or name.endswith(".ffn.gate.bias_vl")):
        return "vision"
    if ".engram.embed." in name:
        return "engram_host"
    if EXPERT.match(name):
        return "expert"
    return "replicated"


def expert_owner(name, num_experts, ep_size):
    if num_experts % ep_size:
        raise ValueError("Expert count must divide evenly across EP ranks")
    expert = int(EXPERT.match(name)[1])
    if expert >= num_experts:
        raise ValueError(f"Invalid expert index in {name}")
    return expert // (num_experts // ep_size)


def _parameter_contract(name, ref, param):
    import torch

    expected = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
                "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8}[ref.dtype]
    valid = {expected}
    if ref.dtype in ("F8_E4M3", "F8_E8M0", "I8"):
        valid.add(torch.uint8)
    if ref.dtype == "I8":
        valid.add(torch.float4_e2m1fn_x2)
    if ref.dtype == "BF16" and (
        name == "head.weight" or re.fullmatch(r"layers\.\d+\.attn\.compressor\.(wgate|wkv)\.weight", name)
    ):
        valid.add(torch.float32)
    if param.dtype not in valid or tuple(param.shape) != ref.shape:
        raise ValueError(f"{name}: checkpoint {ref.dtype} {ref.shape}, target {param.dtype} {tuple(param.shape)}")


def _read_rows(ref, stream, start, end):
    import torch

    stride = ref.nbytes // ref.shape[0]
    stream.seek(ref.offset + start * stride)
    buf = bytearray((end - start) * stride)
    if stream.readinto(buf) != len(buf):
        raise EOFError(f"Truncated tensor in {ref.file.name}")
    # Keep exotic formats as raw bytes from their original file until conversion/transfer.
    dtype = {"BF16": torch.bfloat16, "F32": torch.float32}.get(ref.dtype, torch.uint8)
    return torch.frombuffer(buf, dtype=dtype).view(end - start, *ref.shape[1:])


def _copy_tensor(ref, param, files, wo_scale=None):
    import torch

    row_bytes = ref.nbytes // ref.shape[0]
    rows = max(1, (8 << 20) // row_bytes)
    if wo_scale is not None:
        rows = max(32, (rows // 32) * 32)
    for start in range(0, ref.shape[0], rows):
        end = min(start + rows, ref.shape[0])
        data = _read_rows(ref, files[ref.file], start, end)
        target = param[start:end]
        if wo_scale is not None:
            # Frozen convert.py stores the block-diagonal output projection in BF16.
            scale = _read_rows(wo_scale, files[wo_scale.file], start // 32, end // 32)
            expanded = scale.view(torch.float8_e8m0fnu).float()[:, None, :, None]
            value = data.view(torch.float8_e4m3fn).float().reshape((end - start) // 32, 32, -1, 32)
            target.copy_((value * expanded).reshape_as(target).bfloat16())
        elif ref.dtype in ("F8_E4M3", "F8_E8M0", "I8"):
            target.view(torch.uint8).copy_(data)
        else:
            target.copy_(data)


def load_checkpoint(model, checkpoint_dir, *, ep_rank, device, ep_size=8, manifest=None):
    """Populate one rank's text backbone. Returns its loaded named-parameter set."""
    import torch
    from .engram import HostEngramEmbedding

    if ep_size != 8 or not 0 <= ep_rank < ep_size:
        raise ValueError("Stage 1 requires eight EP ranks")
    device = torch.device(device)
    before = None
    if manifest is not None and device.type == "npu":
        before = torch.npu.memory_allocated(device)
    tensors, header_hashes = read_checkpoint(checkpoint_dir)
    params = dict(model.named_parameters())
    local, host, converted, actions = {}, {}, {}, {}
    for name, ref in tensors.items():
        family = classify(name)
        action = family
        if family == "expert" and expert_owner(name, 384, ep_size) != ep_rank:
            action = "other_ep_rank"
        elif family == "engram_host":
            prefix, _, leaf = name.rpartition(".")
            host.setdefault(prefix, {})[leaf] = ref
        elif family in ("expert", "replicated"):
            prefix = name.removesuffix(".scale").removesuffix(".weight")
            wo_param = params.get(prefix + ".weight") if prefix.endswith(".attn.wo_a") else None
            if wo_param is not None and wo_param.dtype == torch.bfloat16:
                converted[prefix] = tensors[prefix + ".scale"]
                action = "wo_a_to_bfloat16"
                if name.endswith(".scale"):
                    actions[name] = action
                    continue
            local[name] = ref
        actions[name] = action

    if local.keys() != params.keys():
        missing = sorted(params.keys() - local.keys())[:6]
        unexpected = sorted(local.keys() - params.keys())[:6]
        raise ValueError(f"Checkpoint/target mismatch: missing={missing}, unexpected={unexpected}")
    embeddings = {name: module for name, module in model.named_modules() if isinstance(module, HostEngramEmbedding)}
    if host.keys() != embeddings.keys():
        raise ValueError(f"Engram module paths differ: {sorted(embeddings)}, expected {sorted(host)}")
    for name, pair in host.items():
        embeddings[name].validate_layout(pair["weight"], pair["scale"])
    for name, ref in local.items():
        param = params[name]
        if param.device.type != device.type or (device.index is not None and device.index != param.device.index):
            raise ValueError(f"{name}: expected allocated target on {device}, got {param.device}")
        prefix = name.removesuffix(".weight")
        if prefix in converted:
            scale = converted[prefix]
            if (ref.dtype != "F8_E4M3" or scale.dtype != "F8_E8M0" or len(ref.shape) != 2
                    or ref.shape != tuple(param.shape) or ref.shape != tuple(32 * x for x in scale.shape)):
                raise ValueError(f"Invalid wo_a block geometry at {name}")
        else:
            _parameter_contract(name, ref, param)
    # All rank-local shape/dtype/key checks precede the first data copy.
    with torch.no_grad(), ExitStack() as stack:
        files = {file: stack.enter_context(file.open("rb", buffering=0))
                 for file in {r.file for r in (*local.values(), *converted.values())}}
        for name, ref in local.items():
            _copy_tensor(ref, params[name], files, converted.get(name.removesuffix(".weight")))
        for name, pair in host.items():
            embeddings[name].bind(pair["weight"], pair["scale"], device)
    if manifest is not None:
        path = Path(manifest)
        key_path = path.with_name(path.name + ".keys.jsonl.gz")
        with gzip.open(key_path, "xt", encoding="utf-8") as log:
            for name in sorted(actions):
                print(json.dumps([name, actions[name]], separators=(",", ":")), file=log)
        memory = {}
        if before is not None:
            torch.npu.synchronize(device)
            free, total = torch.npu.mem_get_info(device)
            memory = {"before_copy_allocated": before, "after_bind_allocated": torch.npu.memory_allocated(device),
                      "process_peak_allocated": torch.npu.max_memory_allocated(device),
                      "device_free_after_bind": free, "device_total": total}
        with path.open("x") as log:
            json.dump({"device_type": device.type, "device_memory_bytes": memory, "ep_rank": ep_rank, "ep_size": ep_size, "index_sha256": INDEX_SHA256,
                       "shard_header_sha256": header_hashes, "key_actions": dict(Counter(actions.values())),
                       "all_keys": key_path.name, "named_device_parameters": len(params),
                       "parameter_bytes": sum(p.numel() * p.element_size() for p in params.values()),
                       "engram_host_file_bytes": sum(ref.nbytes for pair in host.values() for ref in pair.values())}, log)
    return set(local)


def inventory(checkpoint_dir, output):
    """Emit a header-only inventory, owner-level byte budget and complete key disposition."""
    tensors, headers = read_checkpoint(checkpoint_dir)
    ep_size = 8
    tally, counts = Counter(), Counter()
    resident = [Counter() for _ in range(ep_size)]
    key_path = output.with_name(output.name + ".keys.jsonl.gz")
    with gzip.open(key_path, "xt", encoding="utf-8") as keys:
        for name, ref in sorted(tensors.items()):
            family = classify(name)
            counts[family] += 1
            tally[family] += ref.nbytes
            owner = expert_owner(name, 384, ep_size) if family == "expert" else None
            print(json.dumps([name, ref.file.name, ref.offset, ref.dtype, ref.shape, ref.nbytes,
                              family, owner], separators=(",", ":")), file=keys)
            if family == "replicated":
                for row in resident:
                    row[family] += ref.nbytes
            elif family == "expert":
                resident[owner][family] += ref.nbytes
    with output.open("x") as log:
        json.dump({"scope": "source shard headers only", "index_sha256": INDEX_SHA256,
                   "shard_header_sha256": headers, "key_counts": counts, "source_bytes": tally,
                   "rank_native_checkpoint_bytes": resident, "per_key_file": key_path.name,
                   "per_key_columns": ["key", "shard", "payload_file_offset", "dtype", "shape", "bytes",
                                       "purpose", "exclusive_ep_owner"],
                   "reasons": {"dspark": "speculative decoding disabled",
                               "vision": "image tower/aligner/delimiters or vision-only routing bias",
                               "expert": "routed experts owned by disjoint EP ranges",
                               "engram_host": "read-only host file mapping of table and block scales",
                               "replicated": "TP1 text parameters replicated across DP ranks"}}, log, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=inventory.__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inventory(args.checkpoint, args.output)
