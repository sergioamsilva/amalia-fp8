"""Split a single model.safetensors into shards, without reprocessing the model.

llm-compressor writes one file. The Hub convention is to shard above 5 GB, which lets
interrupted downloads resume and helps anyone on a weak connection.

This reads tensor by tensor with `safe_open` (lazy), groups them into shards of the
requested size, and writes model.safetensors.index.json. Tensor data is untouched: the
same bytes go out. Header metadata is preserved.

Verified after use by comparing a random sample of tensors byte for byte against a backup
of the original file, and by re-measuring perplexity (identical to four decimal places).

Usage:
    SOURCE=/path/to/model LIMIT_GB=4 python3 scripts/reshard.py
"""
import json, os, sys

from safetensors import safe_open
from safetensors.torch import save_file

SOURCE = os.environ.get("SOURCE", "/root/.cache/huggingface/AMALIA-9B-0626-DPO-FP8")
LIMIT = int(os.environ.get("LIMIT_GB", "4")) * 1000 ** 3

path_in = os.path.join(SOURCE, "model.safetensors")
if not os.path.exists(path_in):
    print(f"nothing to do: {path_in} does not exist (already sharded?)")
    sys.exit(0)

DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
               "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
               "F32": 4, "I32": 4, "U32": 4, "I64": 8, "U64": 8, "F64": 8}

with safe_open(path_in, framework="pt") as f:
    metadata_hdr = f.metadata()
    names = list(f.keys())
    sizes = {}
    for n in names:
        t = f.get_slice(n)
        n_elem = 1
        for d in t.get_shape():
            n_elem *= d
        dt = t.get_dtype()
        if dt not in DTYPE_BYTES:
            # guessing here would silently produce wrongly sized shards
            raise RuntimeError(f"unknown dtype {dt!r} on tensor {n!r}; add it to the table")
        sizes[n] = n_elem * DTYPE_BYTES[dt]

total = sum(sizes.values())
print(f"source: {path_in}")
print(f"  {len(names)} tensors, {total/1e9:.2f} GB")
print(f"  limit per shard: {LIMIT/1e9:.0f} GB")

groups = [[]]
acc = 0
for n in names:
    if acc + sizes[n] > LIMIT and groups[-1]:
        groups.append([])
        acc = 0
    groups[-1].append(n)
    acc += sizes[n]

n_shards = len(groups)
print(f"  -> {n_shards} shards")

mapping = {}
with safe_open(path_in, framework="pt") as f:
    for i, group in enumerate(groups, 1):
        shard_name = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        tensors = {n: f.get_tensor(n) for n in group}
        dest = os.path.join(SOURCE, shard_name)
        save_file(tensors, dest, metadata=metadata_hdr)
        for n in group:
            mapping[n] = shard_name
        print(f"  {shard_name}  {len(group):>4} tensors  {os.path.getsize(dest)/1e9:.2f} GB")
        del tensors

index = {"metadata": {"total_size": total}, "weight_map": mapping}
with open(os.path.join(SOURCE, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)
print("  index written")

os.remove(path_in)
print(f"  {path_in} removed")
print("done")
