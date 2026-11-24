"""Training state for resuming RBench in addition to the OFT inference artifacts."""
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def signature(cfg):
    keys = ("dataset_name", "batch_size", "grad_accumulation_steps", "use_lora", "lora_rank",
            "lora_dropout", "use_film", "use_diffusion", "use_l1_regression", "use_proprio",
            "num_images_in_input", "learning_rate", "lr_warmup_steps", "num_steps_before_decay",
            "image_aug", "shuffle_buffer_size")
    result = {key: getattr(cfg, key) for key in keys}
    result["world_size"] = dist.get_world_size()
    for key in ("manifest.json", "norm_stats.json"):
        result[key] = hashlib.sha256((cfg.data_root_dir / key).read_bytes()).hexdigest()
    result["run_id"] = cfg.run_id_override
    result["base_index"] = hashlib.sha256((Path(cfg.vla_path) / "model.safetensors.index.json").read_bytes()).hexdigest()
    return result


def parameters(modules):
    return {f"{prefix}.{n}": p for prefix, module in modules if module is not None
            for n, p in module.named_parameters() if p.requires_grad}


def save_state(path, step, params, optimizer, scheduler, identity):
    ns = np.random.get_state()
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state().cpu(), random.getstate(),
           (ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]))
    states = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(rng, states, dst=0)
    if dist.get_rank() == 0:
        sched = scheduler.state_dict()
        sched["milestones"] = dict(sched["milestones"])
        payload = dict(completed=step, weights={n:p.detach().cpu() for n,p in params.items()},
                       optimizer=optimizer.state_dict(), scheduler=sched, identity=identity, rng=states)
        if sched["last_epoch"] != step:
            raise RuntimeError("Scheduler and completed update count disagree")
        torch.save(payload, path / "training_state.pt")


def restore(path, params, optimizer, scheduler, identity, run_dir):
    if not (path / "COMPLETE.json").exists():
        raise ValueError(f"Checkpoint is incomplete or predates resumable training: {path}")
    st = torch.load(str(path / "training_state.pt"),map_location="cpu",mmap=True,weights_only=True)
    if st["identity"] != identity or list(st["weights"]) != list(params):
        raise ValueError("Checkpoint dataset, configuration or trainable parameter order differs")
    opt_state = st["optimizer"]
    with torch.no_grad():
        for name,p in params.items():
            value=st["weights"][name]
            if p.shape != value.shape or p.dtype != value.dtype:
                raise ValueError(f"Incompatible checkpoint parameter {name}")
            p.copy_(value)
            if not torch.equal(p, value.to(p.device)):
                raise ValueError(f"Weight mismatch on restore: {name}")
    optimizer.load_state_dict(opt_state)
    sched=dict(st["scheduler"])
    sched["milestones"]=Counter(sched["milestones"])
    scheduler.load_state_dict(sched)
    nstates=0
    for saved_group,group in zip(opt_state["param_groups"],optimizer.param_groups,strict=True):
        for key in saved_group:
            if key!="params" and saved_group[key]!=group[key]:
                raise ValueError(f"Optimizer group mismatch: {key}")
        for pid,param in zip(saved_group["params"],group["params"],strict=True):
            for key,val in opt_state["state"].get(pid,{}).items():
                actual=optimizer.state[param][key]
                if actual.dtype!=val.dtype or not torch.equal(actual,val.to(actual.device)):
                    raise ValueError(f"Optimizer mismatch: {pid}/{key}")
                nstates+=1
    step=st["completed"]
    lr=[group["lr"] for group in optimizer.param_groups]
    if not nstates or scheduler.last_epoch!=step or lr!=scheduler.get_last_lr():
        raise ValueError("Restored optimizer, scheduler or learning rate is inconsistent")
    cpu,cuda,py,ns=st["rng"][dist.get_rank()]
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(cuda)
    random.setstate(py)
    np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),*ns[2:]))
    receipt=dict(step=step,lr=lr,weight_tensors=len(params),verified_optimizer_tensors=nstates,
                 rank=dist.get_rank(),adam_max_step=max(int(v["step"].item()) for v in optimizer.state.values() if "step" in v))
    (run_dir / f"restored-{step}-rank-{dist.get_rank()}.json").write_text(json.dumps(receipt))
    dist.barrier()
    return step


def curve_record(run_dir, step, metrics, lr, device):
    keys=sorted(metrics)
    vals=torch.tensor([metrics[k] for k in keys],device=device,dtype=torch.float64)
    dist.all_reduce(vals)
    vals/=dist.get_world_size()
    row=dict(step=step,time=time.time(),lr=lr,**dict(zip(keys,vals.tolist())))
    if not np.isfinite(row["loss_value"]):
        raise FloatingPointError(f"Nonfinite distributed training loss at update {step}")
    if dist.get_rank()==0:
        with (run_dir / "training_curve.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
            f.flush()
            os.fsync(f.fileno())
    return {k:row[k] for k in keys}


def reopen_curve(run_dir, step):
    path=run_dir/"training_curve.jsonl"
    if dist.get_rank()==0 and path.exists():
        kept=[]
        for line in path.read_bytes().splitlines():
            try:
                row=json.loads(line)
            except ValueError:
                break
            if row["step"]<=step:kept.append(line)
        tmp=path.with_suffix(".partial")
        tmp.write_bytes(b"".join(line+b"\n" for line in kept))
        os.link(path,run_dir/f"curve-before-resume-{time.time_ns()}.jsonl")
        tmp.replace(path)
