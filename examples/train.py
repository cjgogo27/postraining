"""CPU offline SFT / original sum-log-prob DPO, strict same-plan update-boundary resume.

Load only trusted checkpoints; full snapshots use Python serialization.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import torch

from postcore import (TinyPolicy, ENCODING, EOS, batch, sequence_logps, sft_loss,
                      dpo_loss, rng_state, restore_rng, source_identity, save_json)

IDENTITY_FILES = ("postcore.py", "train.py")
PLAN = ["stage", "steps", "width", "length", "batch_size", "lr", "beta", "seed", "threads"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sft", "dpo"], default="sft")
    parser.add_argument("--data", default="runs/data")
    parser.add_argument("--out", default="runs/sft")
    init = parser.add_mutually_exclusive_group()
    init.add_argument("--init", help="Trusted matching tiny checkpoint, weights only; starts new stage")
    init.add_argument("--resume", help="Trusted full same-plan snapshot")
    for name, default in [("steps", 300), ("width", 64), ("length", 32), ("batch-size", 8),
                          ("seed", 42), ("threads", 1), ("eval-every", 100)]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()
    if min(args.steps, args.width, args.length, args.batch_size, args.threads, args.eval_every) <= 0 or args.width % 4 or not (0 < args.lr < float("inf")) or not (0 < args.beta < float("inf")):
        parser.error("Invalid size / learning rate / beta")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("stop-after must be in 1..steps")
    if args.stage == "dpo" and not (args.init or args.resume):
        parser.error("DPO requires --init or --resume; reference must have an identity")
    return args


@torch.no_grad()
def generate(model, prompt, max_new=4):
    ids = list(prompt.encode("utf8"))
    answer = []
    for _ in range(max_new):
        if len(ids) > model.length:
            break
        token = int(model(torch.tensor([ids]))[0, -1].argmax())
        if token == EOS:
            return bytes(answer).decode("utf8", errors="replace"), True
        answer.append(token)
        ids.append(token)
    return bytes(answer).decode("utf8", errors="replace"), False


@torch.no_grad()
def evaluate(model, reference, rows, args):
    model.eval()
    if reference is not None:
        reference.eval()
    total_logp, total_count, positive, margins = 0.0, 0, 0, []
    outputs = []
    for start in range(0, len(rows), args.batch_size):
        local = rows[start:start + args.batch_size]
        chosen, counts = sequence_logps(model, batch(local, args.length))
        rejected, _ = sequence_logps(model, batch(local, args.length, "rejected"))
        total_logp += chosen.sum().item()
        total_count += int(counts.sum())
        positive += int((chosen > rejected).sum())
        if reference is not None:
            rc, _ = sequence_logps(reference, batch(local, args.length))
            rr, _ = sequence_logps(reference, batch(local, args.length, "rejected"))
            _, margin = dpo_loss(chosen, rejected, rc, rr, args.beta)
            margins.extend(margin.tolist())
    for row in rows:
        text, stopped = generate(model, row["prompt"])
        outputs.append(dict(id=row["id"], generated=text, eos=stopped, expected=row["chosen"],
                            correct=stopped and text == row["chosen"]))
    result = dict(val_response_nll=-total_logp / total_count, val_response_positions=total_count,
                  val_pair_accuracy=positive / len(rows), greedy_exact=sum(x["correct"] for x in outputs) / len(rows))
    if margins:
        result["val_dpo_margin_mean"] = sum(margins) / len(margins)
    return result, outputs


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This CPU teaching script supports world_size=1 only")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    folder = Path(args.data)
    data, hashes = {}, {}
    for split in ("train", "val"):
        raw = (folder / f"{split}.jsonl").read_bytes()
        data[split] = [json.loads(line) for line in raw.decode().splitlines()]
        if not data[split]:
            raise ValueError("Empty data split")
        hashes[split] = hashlib.sha256(raw).hexdigest()
    if {r["group"] for r in data["train"]} & {r["group"] for r in data["val"]}:
        raise ValueError("Group leakage")
    identity = source_identity(*IDENTITY_FILES)
    model = TinyPolicy(args.width, args.length)
    initialization = dict(kind="random-tiny-fixture", not_pretrained=True)
    step, examples, positions, reference = 0, 0, 0, None
    if args.init:
        source = torch.load(args.init, map_location="cpu", weights_only=False)
        if source["encoding"] != ENCODING or any(source["config"][key] != getattr(args, key) for key in ("width", "length")):
            raise ValueError("Initialization architecture / encoding mismatch")
        model.load_state_dict(source["model"])
        initialization = dict(kind="weights-only-new-stage", source_stage=source["config"]["stage"],
                              source_checkpoint_sha256=hashlib.sha256(Path(args.init).read_bytes()).hexdigest())
    if args.stage == "dpo":
        reference = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed + 1000)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1 or state.get("code_sha256") != identity:
            raise ValueError("Training code identity mismatch")
        if state["encoding"] != ENCODING or state["data_hashes"] != hashes:
            raise ValueError("Data / encoding mismatch")
        if state["torch"] != str(torch.__version__) or state["world_size"] != 1:
            raise ValueError("Environment / world size mismatch")
        if any(state["config"][key] != getattr(args, key) for key in PLAN):
            raise ValueError("Training plan mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if reference is not None:
            reference.load_state_dict(state["reference"])
        step, examples, positions = state["step"], state["examples_seen"], state["response_positions"]
        initialization = state["initialization"]
        restore_rng(state["rng"], generator)
    out = Path(args.out)
    if not args.resume and any((out / name).exists() for name in ("latest.pt", "metrics.jsonl")):
        raise ValueError("New run requires a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    target = args.stop_after or args.steps
    if target <= step:
        raise ValueError("Target must exceed checkpoint step")
    environment = dict(os=platform.platform(), python=platform.python_version(), torch=str(torch.__version__), threads=args.threads, precision="FP32", device="cpu")
    started = time.perf_counter()

    def report(loss=None, save=True):
        metrics, outputs = evaluate(model, reference, data["val"], args)
        record = dict(stage=args.stage, step=step, examples_seen=examples, response_positions=positions,
                      train_loss=loss, elapsed_seconds=time.perf_counter() - started, **metrics)
        with (out / "metrics.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_json(out / "outputs.json", outputs)
        if save:
            state = dict(format_version=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         reference=reference.state_dict() if reference is not None else None,
                         step=step, examples_seen=examples, response_positions=positions,
                         rng=rng_state(generator), config=vars(args), initialization=initialization,
                         data_hashes=hashes, encoding=ENCODING, code_sha256=identity,
                         world_size=1, torch=str(torch.__version__), environment=environment)
            temporary = out / "latest.pt.tmp"
            torch.save(state, temporary)
            os.replace(temporary, out / "latest.pt")
        print(json.dumps(record), flush=True)

    if step == 0:
        report(save=False)
    while step < target:
        model.train()
        indices = torch.randint(len(data["train"]), (args.batch_size,), generator=generator).tolist()
        rows = [data["train"][index] for index in indices]
        chosen = batch(rows, args.length)
        optimizer.zero_grad(set_to_none=True)
        if args.stage == "sft":
            loss = sft_loss(model, chosen)
            count = int(chosen[2].sum())
        else:
            rejected = batch(rows, args.length, "rejected")
            pc, _ = sequence_logps(model, chosen)
            pr, _ = sequence_logps(model, rejected)
            with torch.no_grad():
                rc, _ = sequence_logps(reference, chosen)
                rr, _ = sequence_logps(reference, rejected)
            loss, _ = dpo_loss(pc, pr, rc, rr, args.beta)
            count = int(chosen[2].sum() + rejected[2].sum())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        examples += len(rows)
        positions += count
        if step % args.eval_every == 0 or step == target:
            report(loss.item())


if __name__ == "__main__":
    run(parse_args())
