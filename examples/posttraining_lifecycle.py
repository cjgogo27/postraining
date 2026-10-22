"""CPU lifecycle and distributed-gradient teaching fixture, NOT an LLM pipeline.

Four contexts, three categorical actions, 12 parameters. SFT -> DPO -> RL is
one illustrative choice. No text generation, tool service or production queue.
--ddp additionally starts two real local Gloo processes on fixed PPO tensors.
"""
import argparse
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import socket

import torch
from torch import nn
from torch.nn import functional as F


DTYPE = torch.float64
TARGET = torch.tensor([0, 1, 2, 0])
PLAN = dict(contexts=4, actions=3, group=4, epochs=2, lr=.02, kl=.01, dtype="float64")


def identity():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def weight_id(tensor):
    return hashlib.sha256(json.dumps(tensor.detach().tolist()).encode()).hexdigest()


def exact_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact_equal(a, b)
    else:
        assert left == right, (left, right)


class TablePolicy(nn.Module):
    def __init__(self, initial=None):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(4, 3, dtype=DTYPE) if initial is None else initial.clone())

    def forward(self):
        return self.logits


class Learner:
    def __init__(self, weights, stage):
        self.model = TablePolicy(weights.detach())
        self.stage = stage
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=PLAN["lr"], weight_decay=.01)
        self.reference = weights.detach().clone()
        self.generator = torch.Generator().manual_seed(42)
        self.updates = 0
        self.rounds = 0
        self.sampled_actions = 0
        self.committed = []
        self.trace = []

    def update(self, loss):
        assert torch.isfinite(loss)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()
        self.updates += 1

    def collect(self):
        with torch.no_grad():
            snapshot = self.model().detach().clone()
            # Local snapshot acknowledgement: not an RPC/weight-resharding test.
            worker_weights = snapshot.clone()
            assert weight_id(worker_weights) == weight_id(snapshot)
            logp = worker_weights.log_softmax(-1)
            actions = torch.multinomial(logp.exp(), PLAN["group"], replacement=True, generator=self.generator)
            rewards = (actions == TARGET[:, None]).to(DTYPE)
            a = (rewards - rewards.mean(-1, keepdim=True)) / rewards.std(-1, keepdim=True, correction=0).clamp_min(1e-8)
        return dict(batch_id=f"rl-{self.rounds}", actor_version=self.updates,
                    weight_id=weight_id(snapshot), reward_version="categorical-verifier-v1", status="ok",
                    actions=actions, old_logp=logp.gather(-1, actions), advantages=a, valid_count=actions.numel())

    def admit(self, batch):
        if batch["batch_id"] in self.committed:
            raise ValueError("duplicate committed batch")
        if not 0 <= self.updates - batch["actor_version"] <= 2:
            raise ValueError("stale or future actor")
        if batch["reward_version"] != "categorical-verifier-v1" or batch["status"] != "ok":
            raise ValueError("score identity/status")
        if batch["valid_count"] <= 0:
            raise ValueError("empty batch")

    def train_batch(self, batch):
        self.admit(batch)
        old_before = batch["old_logp"].clone()
        for _ in range(PLAN["epochs"]):
            current = self.model().log_softmax(-1)
            ratio = (current.gather(-1, batch["actions"]) - batch["old_logp"]).exp()
            a = batch["advantages"]
            surrogate = torch.minimum(ratio * a, ratio.clamp(.8, 1.2) * a).mean()
            ref_logp = self.reference.log_softmax(-1)
            kl = (current.exp() * (current - ref_logp)).sum(-1).mean()
            self.update(-surrogate + PLAN["kl"] * kl)
        assert torch.equal(old_before, batch["old_logp"])
        self.committed.append(batch["batch_id"])
        self.rounds += 1
        self.sampled_actions += batch["valid_count"]
        self.trace.append(dict(batch_id=batch["batch_id"], sampled_actor=batch["actor_version"],
                               published_actor=self.updates, actions=batch["actions"].tolist(),
                               sampled_actions=batch["valid_count"], old_logp_unchanged=True))

    def state(self):
        return copy.deepcopy(dict(schema=1, code_sha256=identity(), torch=str(torch.__version__), world_size=1,
            plan=PLAN, stage=self.stage, model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            reference=self.reference, generator=self.generator.get_state(), updates=self.updates,
            rounds=self.rounds, sampled_actions=self.sampled_actions, committed=self.committed, trace=self.trace))

    def restore(self, state):
        if state["schema"] != 1 or state["code_sha256"] != identity() or state["torch"] != str(torch.__version__):
            raise ValueError("code/schema/environment mismatch")
        if state["plan"] != PLAN or state["world_size"] != 1 or state["stage"] != self.stage:
            raise ValueError("plan/layout/stage mismatch")
        self.model.load_state_dict(state["model"]); self.optimizer.load_state_dict(state["optimizer"])
        self.reference = state["reference"].clone(); self.generator.set_state(state["generator"])
        for key in ["updates", "rounds", "sampled_actions", "committed", "trace"]:
            setattr(self, key, copy.deepcopy(state[key]))


def lifecycle(out):
    checks = []
    base = torch.zeros(4, 3, dtype=DTYPE)
    sft = Learner(base, "sft")
    for _ in range(4):
        sft.update(F.cross_entropy(sft.model(), TARGET))
    dpo = Learner(sft.model().detach(), "dpo")
    assert not dpo.optimizer.state and dpo.updates == 0
    assert weight_id(dpo.model()) == weight_id(sft.model())
    reference_id = weight_id(dpo.reference)
    rejected = (TARGET + 1) % 3; rows = torch.arange(4)
    for _ in range(3):
        logp = dpo.model().log_softmax(-1); ref = dpo.reference.log_softmax(-1)
        margin = .1 * (logp[rows, TARGET] - logp[rows, rejected] - ref[rows, TARGET] + ref[rows, rejected])
        dpo.update(F.softplus(-margin).mean())
    assert reference_id == weight_id(dpo.reference)
    checks.append(dict(name="sft_to_dpo_inherits_weights_resets_optimizer_freezes_reference",
                       sft_updates=4, sft_demo_consumptions=16, dpo_updates=3, pair_consumptions=12,
                       base_weight_id=weight_id(base), sft_weight_id=weight_id(sft.model()),
                       dpo_reference_id=reference_id, dpo_weight_id=weight_id(dpo.model())))
    rl = Learner(dpo.model().detach(), "rl")
    assert not rl.optimizer.state and rl.updates == 0
    first = rl.collect(); rl.train_batch(first)
    checks.append(dict(name="rl_new_run_snapshot_and_fixed_old_logp", first_actor=0, after_updates=2,
                       shape=list(first["actions"].shape), positions=first["valid_count"]))
    rejects = []
    for case, edit in [("duplicate", {}), ("stale", {"batch_id":"bad", "actor_version":-1}),
                       ("future", {"batch_id":"bad", "actor_version":3}),
                       ("scorer", {"batch_id":"bad", "reward_version":"other"}),
                       ("timeout", {"batch_id":"bad", "status":"timeout"}),
                       ("empty", {"batch_id":"bad", "valid_count":0})]:
        try:
            rl.admit({**first, **edit})
        except ValueError:
            rejects.append(case)
        else:
            raise AssertionError(f"accepted {case}")
    checks.append(dict(name="batch_admission_rejections", rejected=rejects))

    state = rl.state()
    temporary = out / "checkpoint.tmp"; saved = out / "checkpoint.pt"
    torch.save(state, temporary); os.replace(temporary, saved)
    # Normal completed-round boundary only; no interrupted-write/fault claim.
    rl.train_batch(rl.collect())
    resumed = Learner(base, "rl")
    resumed.restore(torch.load(saved, map_location="cpu", weights_only=True))
    resumed.train_batch(resumed.collect())
    exact_equal(rl.state(), resumed.state())
    checks.append(dict(name="rl_committed_boundary_restore_matches_weights_adam_rng_trace_counts",
                       rl_updates=rl.updates, rounds=rl.rounds, sampled_actions=rl.sampled_actions,
                       committed_batches=rl.committed, trace=rl.trace, exact=True))
    for field, value in [("stage", "sft"), ("code_sha256", "changed"), ("world_size", 2)]:
        changed = copy.deepcopy(state); changed[field] = value
        try:
            Learner(base, "rl").restore(changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted changed {field}")
    checks.append(dict(name="restore_rejects_changed_stage_code_layout", cases=3))
    # Independent textbook arithmetic, not measured task quality or GPU memory.
    q = torch.tensor([.4, -.1 + .9], dtype=DTYPE); v = q.mean(); a = q - v
    torch.testing.assert_close(a, torch.tensor([-.2, .2], dtype=DTYPE))
    checks.append(dict(name="bellman_value_and_centered_advantage", q=q.tolist(), value=v.item(), advantage=a.tolist()))
    rewards = torch.tensor([0., 1.], dtype=DTYPE)
    p = (rewards / .2).softmax(0); changed = (2 * rewards / .2).softmax(0)
    matched = (2 * rewards / .4).softmax(0)
    assert changed[1] > p[1]; torch.testing.assert_close(p, matched)
    checks.append(dict(name="reward_scaling_changes_regularized_optimum_unless_kl_scales",
                       initial=p.tolist(), reward_only=changed.tolist(), reward_and_beta=matched.tolist()))
    lengths = [2, 3, 4, 5, 2, 3, 4, 5]
    assert sum(lengths) == 28 and len(lengths)*5 == 40 and 8//4*2 == 4
    checks.append(dict(name="architecture_batch_ledger", trajectories=8, valid_positions=28,
                       padded_positions=40, epochs=2, actor_updates=4, target_consumptions=56))
    kv = 2*16*8*1024*4*64*2
    assert kv == 134217728
    checks.append(dict(name="hypothetical_memory_arithmetic", actor_bytes_per_parameter=16,
                       actor_critic_reference_rm_GB=36, kv_bytes=kv, kv_MiB=kv/2**20,
                       note="Assumptions only, no measured GPU memory"))
    return checks


class WindowPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.arange(24, dtype=DTYPE).reshape(2, 4, 3)/20)

    def forward(self, indices):
        return self.logits[indices]


def ppo_sum(logits, indices):
    actions = torch.tensor([[0,1,2,0],[1,0,2,0]])[indices]
    mask = torch.tensor([[1,1,0,0],[1,1,1,1]], dtype=DTYPE)[indices]
    a = torch.tensor([[1,-.5,0,0],[.2,-1,.5,.3]], dtype=DTYPE)[indices]
    old = (torch.arange(24, dtype=DTYPE).reshape(2,4,3)/20)[indices].log_softmax(-1).gather(-1,actions[...,None]).squeeze(-1)
    logp = logits.log_softmax(-1); current = logp.gather(-1, actions[...,None]).squeeze(-1)
    ratio = (current-old).exp()
    kl = (logp.exp()*(logp+torch.log(torch.tensor(3., dtype=DTYPE)))).sum(-1)
    return ((-torch.minimum(ratio*a, ratio.clamp(.8,1.2)*a)+.01*kl)*mask).sum(), mask.sum()


def ddp_worker(rank, port, folder):
    import torch.distributed as dist
    torch.set_num_threads(1)
    store = dist.TCPStore("127.0.0.1", port, 2, rank == 0, timeout=timedelta(seconds=45), use_libuv=False)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2)
    try:
        model = torch.nn.parallel.DistributedDataParallel(WindowPolicy())
        indices = torch.tensor([rank]); local_sum, count = ppo_sum(model(indices), indices)
        dist.all_reduce(count)
        (local_sum * 2/count).backward()
        gradient = model.module.logits.grad.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1); optimizer.step()
        Path(folder, f"rank-{rank}.json").write_text(json.dumps(dict(gradient=gradient.tolist(),
            weights=model.module.logits.detach().tolist(), global_count=count.item())), encoding="utf8")
    finally:
        dist.destroy_process_group()


def distributed_check(out, actual):
    indices = torch.tensor([0,1]); model = WindowPolicy()
    total, count = ppo_sum(model(indices), indices); (total/count).backward()
    expected_grad = model.logits.grad.detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    torch.optim.SGD(model.parameters(), lr=.1).step(); expected_weights = model.logits.detach().clone()
    wrong = WindowPolicy(); naive = 0.
    for rank in range(2):
        i = torch.tensor([rank]); numerator, n = ppo_sum(wrong(i), i)
        naive = naive + numerator/n/2
    naive.backward(); wrong_error = (wrong.logits.grad-expected_grad).abs().max().item()
    assert wrong_error > 1e-3
    result = dict(name="unequal_mask_global_ppo_gradient", global_count=count.item(),
                  local_counts=[2,4], naive_local_mean_max_gradient_error=wrong_error,
                  real_two_process_ddp=False)
    if actual:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        torch.multiprocessing.spawn(ddp_worker, args=(port,str(out)), nprocs=2, join=True)
        errors = []
        for rank in range(2):
            record = json.loads(Path(out,f"rank-{rank}.json").read_text(encoding="utf8"))
            gradient = torch.tensor(record["gradient"],dtype=DTYPE)
            weights = torch.tensor(record["weights"],dtype=DTYPE)
            torch.testing.assert_close(gradient,expected_grad,atol=1e-12,rtol=1e-10)
            torch.testing.assert_close(weights,expected_weights,atol=1e-12,rtol=1e-10)
            errors.append(dict(rank=rank,gradient_max_abs=(gradient-expected_grad).abs().max().item(),
                               weight_max_abs=(weights-expected_weights).abs().max().item()))
        result.update(real_two_process_ddp=True,backend="gloo",ranks=errors)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="runs/posttraining-lifecycle")
    parser.add_argument("--ddp",action="store_true",help="Run two actual local Gloo processes")
    args=parser.parse_args(); out=Path(args.out)
    if out.exists():
        raise ValueError("Use a fresh output directory")
    out.mkdir(parents=True); torch.set_num_threads(1)
    checks=lifecycle(out); checks.append(distributed_check(out,args.ddp))
    report=dict(status="PASS",count=len(checks),python=platform.python_version(),torch=str(torch.__version__),
                device="cpu",dtype="float64",code_sha256=identity(),plan=PLAN,checks=checks,
                scope="12-parameter table lifecycle and separate fixed PPO tensors; no LLM, GPU, RPC, actual tool or production asynchronous queue")
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
