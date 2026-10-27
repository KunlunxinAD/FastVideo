# SPDX-License-Identifier: Apache-2.0
"""Optional local-only DMD smoke-test assertions; no framework code changes."""
import json
import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from fastvideo.train.callbacks.callback import Callback


def local(t):
    return t.to_local() if hasattr(t, "to_local") else t


class DMDHealthCheck(Callback):
    def _parameters(self, method, role):
        return [(name, local(p).detach()) for name, p in method._role_models[role].transformer.named_parameters()
                if p.requires_grad]

    def _snapshot(self, method):
        self.previous = {role: {name: p.clone() for name, p in self._parameters(method, role)}
                         for role in ("student", "critic")}

    def on_train_start(self, method, iteration=0):
        self.rank = dist.get_rank()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.rows = []
        self.update_counts = {"student": 0, "critic": 0}
        assert not any(p.requires_grad for p in method._role_models["teacher"].transformer.parameters())
        self._snapshot(method)
        for role in self.previous:
            assert self.previous[role], f"No trainable {role} parameters"
        self.log_dir = Path(self.training_config.checkpoint.output_dir) / "health"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()

    def on_after_optimizer_step(self, method, iteration=0):
        scheduled = method._should_update_student(iteration)
        self.deltas = {}
        self.grad_norms = {}
        clip = getattr(getattr(self, "_callback_dict", None), "_callbacks", {}).get("grad_clip")
        if clip is not None:
            self.grad_norms = dict(clip.last_grad_norms)
            assert all(math.isfinite(v) for v in self.grad_norms.values()), self.grad_norms
        for role in ("student", "critic"):
            delta = torch.zeros((), device=self.device)
            for name, p in self._parameters(method, role):
                delta += (p.float() - self.previous[role][name].float()).abs().sum()
            dist.all_reduce(delta)
            value = float(delta.item())
            assert math.isfinite(value), (role, iteration, value)
            expected_update = role == "critic" or scheduled
            assert (value > 0) if expected_update else (value == 0), (role, iteration, value, expected_update)
            self.update_counts[role] += int(expected_update)
            self.deltas[role] = value
        self._snapshot(method)

    def on_training_step_end(self, method, loss_dict, iteration=0):
        metrics = {k: float(v) for k, v in loss_dict.items()}
        assert all(math.isfinite(v) for v in metrics.values()), metrics
        assert bool(metrics["update_student"]) == method._should_update_student(iteration)
        row = {"rank": self.rank, "step": iteration, "metrics": metrics,
               "global_parameter_delta_l1": self.deltas, "local_grad_norms": self.grad_norms,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
               "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
        self.rows.append(row)
        with (self.log_dir / f"rank{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if self.rank == 0:
            print("DMD_HEALTH " + json.dumps(row, sort_keys=True), flush=True)

    def on_train_end(self, method, iteration=0):
        assert len(self.rows) == iteration, (len(self.rows), iteration)
        assert self.update_counts["student"] > 0
        assert self.update_counts["critic"] == iteration
        dist.barrier()
        summary = {"status": "DMD_HEALTH_PASS", "rank": self.rank, "steps": iteration,
                   "updates": self.update_counts, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                   "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
        (self.log_dir / f"rank{self.rank}.summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
