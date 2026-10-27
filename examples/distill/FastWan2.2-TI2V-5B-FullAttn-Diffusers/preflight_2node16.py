#!/usr/bin/env python3
"""torchrun 2x8 communication, HSDP gradient, and local-data preflight."""
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
assert world == 16 and int(os.environ["LOCAL_WORLD_SIZE"]) == 8
assert torch.cuda.device_count() == 8, "Visibility mask must expose exactly eight devices"
props = torch.cuda.get_device_properties(local_rank)
expected = os.environ.get("EXPECTED_PCI_BUS_IDS", "")
if expected:
    assert props.pci_bus_id in [int(x, 16) for x in expected.split(",")], str(props)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print(json.dumps({"stage":"device", "rank":rank, "local_rank":local_rank,
                  "mask":os.environ["CUDA_VISIBLE_DEVICES"], "pci_bus_id":props.pci_bus_id}), flush=True)
# Reuse the same initialization and Gloo object groups as the real trainer.
from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
maybe_init_distributed_environment_and_model_parallel(1, 1, timeout=timedelta(seconds=180))
from fastvideo.dataset.parquet_dataset_map_style import get_parquet_files_and_length
paths, lengths = get_parquet_files_and_length(os.environ["DATA_DIR"], require_existing_parquet_cache=True)
assert len(paths) == 100 and sum(lengths) == 800
for path in paths:
    assert Path(path).is_file(), f"Rank {rank} missing local data: {path}"
index_hash = hashlib.sha256(json.dumps([paths, lengths]).encode()).hexdigest()
objects = [None] * world
dist.all_gather_object(objects, index_hash)
assert len(set(objects)) == 1

for dtype in (torch.float32, torch.bfloat16):
    for size in (1, 4096, 1048576):
        value = torch.full((size,), rank + 1, device=device, dtype=dtype)
        dist.all_reduce(value)
        assert torch.all(value == 136).item(), (rank, dtype, size)
print(f"rank={rank} all_reduce PASS fp32/bf16 small/4MiB", flush=True)
mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("replicate", "shard"))
model = torch.nn.Linear(32, 16, bias=False, device=device)
with torch.no_grad():
    model.weight.fill_(0.25)
fully_shard(model, mesh=mesh)
optim = torch.optim.SGD(model.parameters(), lr=0.1)
expected_weight = 0.25
mean_square = sum(((r + 1) / world) ** 2 for r in range(world)) / world
for step in range(3):
    optim.zero_grad(set_to_none=True)
    inputs = torch.full((2, 32), (rank + 1) / world, device=device)
    loss = model(inputs).square().mean()
    assert torch.isfinite(loss).item()
    loss.backward()
    optim.step()
    expected_weight -= 0.1 * (2 * 32 * expected_weight * mean_square / 16)
    torch.cuda.synchronize()
    local_weight = model.weight.to_local().detach()
    err = (local_weight - expected_weight).abs().max()
    dist.all_reduce(err, op=dist.ReduceOp.MAX)
    assert err.item() < 1e-5, (rank, step, err.item(), expected_weight)
dist.barrier()
print(json.dumps({"status":"PREFLIGHT_PASS", "rank":rank, "world_size":world,
                  "hsdp":[2,8], "gradient_steps":3, "data_files":len(paths), "data_rows":sum(lengths)}), flush=True)
from fastvideo.distributed.parallel_state import cleanup_dist_env_and_memory
cleanup_dist_env_and_memory()
