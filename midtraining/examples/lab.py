"""Offline mid-training lab. Fictional byte corpora; CPU only, no downloads.

The locally pretrained tiny base is a mechanism fixture, not an industrial base.
Run from repository root: python midtraining/examples/lab.py --out runs/midtraining
"""
import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

VOCAB, EOS = 257, 256
ENCODING = "utf8-bytes-0-255;eos256;all-shifted-targets;cross-doc-causal-packing"
LR_MULTIPLIER = 1.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_identity():
    return hashlib.sha256(Path(__file__).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()


def corpus():
    tags = ["luma", "nexo", "pavo", "sora"]
    words = ["blue", "calm", "warm", "mild"]
    rows = []
    for source in ["general", "domain"]:
        for index in range(80):
            k = index % 4
            prefix = (f"Unit NX{index:02d} uses Z{k}. Z{k} tag: " if source == "domain"
                      else f"Place GX{index:02d} has W{k}. W{k} tag: ")
            answer = (tags if source == "domain" else words)[k] + ".\n"
            rows.append(dict(id=f"{source}-{index}", group_id=f"{source}-{index}",
                             split="train" if index < 64 else "val", source=source,
                             text=prefix+answer, prefix=prefix, answer=answer))
    return rows


def streams(rows):
    result = {}
    for source in ["general", "domain"]:
        for split in ["train", "val"]:
            ids = []
            for row in rows:
                if row["source"] == source and row["split"] == split:
                    ids.extend(row["text"].encode()); ids.append(EOS)
            result[(source, split)] = torch.tensor(ids, dtype=torch.long)
    return result


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(VOCAB, 32)
        self.position = nn.Embedding(64, 32)
        self.block = nn.TransformerEncoderLayer(32, 4, 128, dropout=0., activation="gelu",
                                               batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, VOCAB, bias=False); self.head.weight = self.token.weight
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, "bias", None) is not None: nn.init.zeros_(module.bias)
        if isinstance(module, nn.MultiheadAttention): nn.init.normal_(module.in_proj_weight, std=.02)

    def forward(self, ids):
        size = ids.shape[1]
        if not 1 <= size <= 64: raise ValueError("sequence outside learned position table 1..64")
        hidden = self.token(ids) + self.position(torch.arange(size))
        causal = torch.ones(size, size, dtype=torch.bool).triu(1)
        return self.head(self.norm(self.block(hidden, src_mask=causal, is_causal=True)))


@dataclass(frozen=True)
class Config:
    name: str = "replay25"
    seed: int = 42
    length: int = 32
    batch: int = 4
    steps: int = 96
    general_fraction: float = .25
    lr: float = .001


def learning_rate(config, step):
    if step < 4: return LR_MULTIPLIER * config.lr * (step + 1) / 4
    progress = (step - 4) / max(config.steps - 5, 1)
    return LR_MULTIPLIER * config.lr * (.1 + .9 * (1 + math.cos(math.pi * progress)) / 2)


def parameter_identity(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()+b"\0"); h.update(value.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def batch(data, config, step, generator):
    count_general = math.floor((step+1)*config.batch*config.general_fraction) - math.floor(step*config.batch*config.general_fraction)
    sources = ["general"]*count_general + ["domain"]*(config.batch-count_general)
    order = torch.randperm(config.batch, generator=generator).tolist()
    windows, ledger = [], []
    for row in order:
        source = sources[row]; stream = data[(source, "train")]
        start = int(torch.randint(len(stream)-config.length, (), generator=generator))
        windows.append(stream[start:start+config.length+1])
        ledger.append(dict(source=source, start=start, targets=config.length))
    ids = torch.stack(windows)
    return ids[:, :-1], ids[:, 1:], ledger


@torch.no_grad()
def evaluate(model, rows, data):
    model.eval(); report = {}
    for length, count in [(32, 16), (64, 8)]:
        for source in ["general", "domain"]:
            stream = data[(source, "val")]
            starts = torch.linspace(0, len(stream)-length-1, count).long().tolist()
            full = torch.stack([stream[start:start+length+1] for start in starts])
            loss = F.cross_entropy(model(full[:, :-1]).reshape(-1, VOCAB), full[:, 1:].reshape(-1), reduction="sum")
            report[f"{source}_{length}"] = dict(nll=float(loss)/(count*length), targets=count*length,
                                                  starts=starts, stream_sha256=digest(stream.tolist()))
    generated = []
    for row in [r for r in rows if r["source"] == "domain" and r["split"] == "val"][:4]:
        tokens = list(row["prefix"].encode()); output = []; ended = False
        for _ in range(12):
            token = int(model(torch.tensor([tokens])).argmax(-1)[0, -1])
            if token == EOS: ended = True; break
            output.append(token); tokens.append(token)
        text = bytes(output).decode("utf8", errors="replace")
        generated.append(dict(id=row["id"], prefix=row["prefix"], expected=row["answer"],
                              output_ids=output, output=text, ended_eos=ended,
                              exact=ended and text == row["answer"]))
    report["generation"] = generated
    return report


class Run:
    def __init__(self, config, rows, initial=None):
        self.config = config; self.rows = rows; self.data = streams(rows)
        torch.manual_seed(config.seed); self.model = TinyBase()
        if initial is not None: self.model.load_state_dict(initial)
        self.initial_identity = parameter_identity(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=.01)
        self.generator = torch.Generator().manual_seed(config.seed+1000)
        self.step = 0; self.targets = 0; self.consumed = dict(general=0, domain=0)
        self.history = []; self.ledger = []

    def contract(self):
        return dict(config=asdict(self.config), data_sha256=digest(self.rows), code_sha256=code_identity(),
                    torch=str(torch.__version__), world_size=1, encoding=ENCODING)

    def advance(self, until=None):
        until = self.config.steps if until is None else until
        if not self.step <= until <= self.config.steps: raise ValueError("invalid target update")
        while self.step < until:
            self.model.train(); x, y, ledger = batch(self.data, self.config, self.step, self.generator)
            lr = learning_rate(self.config, self.step)
            for group in self.optimizer.param_groups: group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(x).reshape(-1, VOCAB), y.reshape(-1))
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            if not torch.isfinite(norm): raise FloatingPointError("nonfinite gradient")
            self.optimizer.step(); self.step += 1
            for item in ledger: self.consumed[item["source"]] += item["targets"]
            self.targets += x.numel(); self.ledger.append(dict(update=self.step, windows=ledger))
            if self.step == 1 or self.step % 12 == 0 or self.step == self.config.steps:
                self.history.append(dict(update=self.step, targets=self.targets, lr=lr, train_nll=float(loss.detach()),
                                         evaluation=evaluate(self.model, self.rows, self.data)))

    def checkpoint(self):
        return dict(contract=self.contract(), initial_identity=self.initial_identity, model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(), generator=self.generator.get_state(),
                    torch_rng=torch.get_rng_state(), step=self.step, targets=self.targets,
                    consumed=self.consumed, history=self.history, ledger=self.ledger)

    def save(self, filename):
        filename = Path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
        temporary = filename.with_suffix(".tmp"); torch.save(self.checkpoint(), temporary); temporary.replace(filename)

    def restore(self, filename):
        saved = torch.load(filename, map_location="cpu", weights_only=False)
        if saved["contract"] != self.contract(): raise ValueError("strict resume identity/config/data/environment mismatch")
        self.model.load_state_dict(saved["model"]); self.optimizer.load_state_dict(saved["optimizer"])
        self.generator.set_state(saved["generator"]); torch.set_rng_state(saved["torch_rng"])
        for field in ["initial_identity", "step", "targets", "consumed", "history", "ledger"]:
            setattr(self, field, saved[field])

    def report(self):
        return dict(config=asdict(self.config), initial_identity=self.initial_identity,
                    final_identity=parameter_identity(self.model), updates=self.step, targets=self.targets,
                    consumed=self.consumed, history=self.history,
                    final=evaluate(self.model, self.rows, self.data), ledger_sha256=digest(self.ledger))


def run_experiment(out):
    rows = corpus(); data = streams(rows)
    report = dict(protocol="EXPERIMENT_PLAN.md fixed before running; matched target budgets; paired seed blocks",
                  code_sha256=code_identity(), data_sha256=digest(rows), encoding=ENCODING,
                  environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                   device="cpu", dtype="float32", threads=1, platform=platform.platform()),
                  streams={f"{s}_{p}":len(t) for (s,p),t in data.items()}, blocks=[])
    (out/"corpus.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    replay_initial = None
    for seed in [42, 43]:
        base = Run(Config("base", seed, steps=120, general_fraction=1., lr=.003), rows)
        base.advance(); base.save(out/f"base-{seed}.pt")
        initial = copy.deepcopy(base.model.state_dict()); arms = {}
        order = ["general", "domain", "replay25"]; random.Random(seed+9).shuffle(order)
        for name in order:
            fraction = dict(general=1., domain=0., replay25=.25)[name]
            arm = Run(Config(name, seed, general_fraction=fraction), rows, initial)
            assert len(arm.optimizer.state) == 0 and arm.step == 0
            arm.advance(); arm.save(out/f"{name}-{seed}.pt"); arms[name] = arm.report()
            assert arm.targets == 12288
            if seed == 42 and name == "replay25": replay_initial = copy.deepcopy(arm.model.state_dict())
        report["blocks"].append(dict(seed=seed, run_order=order, base=base.report(), arms=arms))
    report["length_interface"] = []
    for length, size in [(32,4), (64,2)]:
        arm = Run(Config(f"length{length}",42,length,size,48), rows,replay_initial)
        arm.advance(); arm.save(out/f"length-{length}.pt"); report["length_interface"].append(arm.report())
        assert arm.targets == 6144 and arm.consumed == dict(general=1536,domain=4608)
    report["parameters"] = sum(p.numel() for p in TinyBase().parameters())
    report["scope"] = "Local tiny base; two paired seeds and same-template held-out entity IDs. Length experiment tests interfaces, not long dependency, RoPE/YaRN or general capability. No GPU/DDP."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/midtraining")
    parser.add_argument("--resume", help="strictly restore a default replay25 fixture, for identity rejection checks")
    args = parser.parse_args(); torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    if args.resume:
        run = Run(Config(), corpus()); run.restore(args.resume); run.advance(); return
    out = Path(args.out)
    if out.exists() and any(out.iterdir()): raise ValueError("use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True); start = time.perf_counter()
    report = run_experiment(out); report["wall_seconds_including_evaluation_and_save"] = time.perf_counter()-start
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(dict(status="PASS",parameters=report["parameters"],blocks=len(report["blocks"]),
                          elapsed=report["wall_seconds_including_evaluation_and_save"])))


if __name__ == "__main__": main()
