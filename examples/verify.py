"""Run meaningful offline numerical / mask / stage / strict-resume checks on CPU."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from postcore import (TinyPolicy, batch, sft_loss, sequence_logps, dpo_loss,
                      group_advantages, verify_answer, source_identity, save_json)

HERE = Path(__file__).resolve().parent


def execute(args, error=None):
    result = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                            encoding="utf8", errors="replace", timeout=180,
                            env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
    if error is None:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
    return result


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/checks")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    checks = []
    rows = [dict(prompt="Q:2+3\nA:", chosen="5", rejected="6")]
    inputs, labels, mask = batch(rows)
    assert inputs[0, :9].tolist() == [81, 58, 50, 43, 51, 10, 65, 58, 53]
    assert mask[0].nonzero().flatten().tolist() == [7, 8] and labels[0, 8] == 256
    checks.append("real_byte_ids_shift_response_mask_and_eos")
    model = TinyPolicy(16).eval()
    changed_labels = labels.clone()
    changed_labels[~mask] = 123
    assert sft_loss(model, (inputs, labels, mask)).item() == sft_loss(model, (inputs, changed_labels, mask)).item()
    checks.append("prompt_and_padding_labels_excluded")
    future = inputs.clone()
    future[:, 9:] = 111
    with torch.no_grad():
        assert torch.equal(model(inputs)[:, :9], model(future)[:, :9])
    checks.append("causal_future_invariance")
    uniform = TinyPolicy(16)
    with torch.no_grad():
        for p in uniform.parameters():
            p.zero_()
    varied = batch([rows[0], dict(prompt="Q:9+9\nA:", chosen="18")])
    values, counts = sequence_logps(uniform, varied)
    torch.testing.assert_close(values, -counts * math.log(257), atol=1e-5, rtol=1e-5)
    assert counts.tolist() == [2, 3]
    checks.append("sum_logps_include_variable_length_and_eos")
    pc, pr = torch.tensor([-3.0], requires_grad=True), torch.tensor([-5.0], requires_grad=True)
    rc, rr = torch.tensor([-4.0]), torch.tensor([-4.0])
    loss, margin = dpo_loss(pc, pr, rc, rr, 0.1)
    assert abs(loss.item() - 0.5981388693815918) < 1e-6 and abs(margin.item() - 0.2) < 1e-6
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()
    assert abs(dpo_loss(rc, rr, rc, rr, 0.5)[0].item() - math.log(2)) < 1e-6
    checks.append("dpo_ln2_hand_case_and_gradient_direction")
    torch.testing.assert_close(group_advantages(torch.tensor([[0.0, 0, 1, 1]])), torch.tensor([[-1.0, -1, 1, 1]]))
    assert torch.equal(group_advantages(torch.ones(2, 8)), torch.zeros(2, 8))
    checks.append("group_population_std_and_constant_reward_zero")
    ratio, advantage = torch.tensor([1.5, 1.5]), torch.tensor([1.0, -1.0])
    torch.testing.assert_close(torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage), torch.tensor([1.2, -1.5]))
    checks.append("ppo_clip_positive_and_negative_advantages")
    assert verify_answer("5", 5) == 1 and all(verify_answer(text, 5) == 0 for text in ["05", "50", "answer=5", "5\n"])
    checks.append("verifier_exact_format_counterexamples")
    with tempfile.TemporaryDirectory(prefix="posttraining-check-") as temporary:
        root = Path(temporary)
        data = root / "data"
        execute([HERE / "prepare_data.py", "--out", data])
        common = ["--data", data, "--steps", 8, "--width", 16, "--batch-size", 2, "--eval-every", 4]
        for stage in ("sft", "dpo"):
            extra = ["--stage", stage]
            if stage == "dpo":
                extra += ["--init", root / "sft-continuous/latest.pt", "--lr", 0.0001]
            execute([HERE / "train.py", *common, *extra, "--out", root / f"{stage}-continuous"])
            execute([HERE / "train.py", *common, *extra, "--out", root / f"{stage}-paused", "--stop-after", 4])
            # Init is not needed for resume: the frozen reference is in snapshot.
            resumed_extra = ["--stage", stage] + (["--lr", 0.0001] if stage == "dpo" else [])
            execute([HERE / "train.py", *common, *resumed_extra, "--out", root / f"{stage}-resumed", "--resume", root / f"{stage}-paused/latest.pt"])
            a = torch.load(root / f"{stage}-continuous/latest.pt", weights_only=False)
            b = torch.load(root / f"{stage}-resumed/latest.pt", weights_only=False)
            for key in ("model", "reference", "optimizer", "rng", "step", "examples_seen", "response_positions", "initialization"):
                equal(a[key], b[key])
            assert a["step"] == 8 and a["examples_seen"] == 16
            checks.append(stage + "_exact_update_boundary_resume")
            if stage == "dpo":
                original = torch.load(root / "sft-continuous/latest.pt", weights_only=False)
                equal(a["reference"], original["model"])
                assert all(item["step"].item() == 8 for item in a["optimizer"]["state"].values())
                checks.append("dpo_frozen_reference_and_new_stage_optimizer")
        paused = root / "sft-paused/latest.pt"
        execute([HERE / "train.py", *common, "--lr", 0.002, "--out", root / "plan", "--resume", paused], error="Training plan mismatch")
        checks.append("changed_training_plan_rejected")
        for key, value in (("torch", "changed"), ("world_size", 2)):
            state = torch.load(paused, weights_only=False)
            state[key] = value
            path = root / f"bad-{key}.pt"
            torch.save(state, path)
            execute([HERE / "train.py", *common, "--out", root / key, "--resume", path], error="Environment / world size mismatch")
        checks.append("changed_torch_and_world_size_rejected")
        (root / "postcore.py").write_text((HERE / "postcore.py").read_text(encoding="utf8"), encoding="utf8")
        changed = (HERE / "train.py").read_text(encoding="utf8").replace("lr=args.lr,", "lr=args.lr * 1.01,")
        assert changed != (HERE / "train.py").read_text(encoding="utf8")
        (root / "train.py").write_text(changed, encoding="utf8")
        execute([root / "train.py", *common, "--out", root / "code", "--resume", paused], error="Training code identity mismatch")
        checks.append("changed_executable_lr_code_rejected")
        raw = (data / "train.jsonl").read_bytes()
        (data / "train.jsonl").write_bytes(raw.replace(b'"Q:', b'"X:', 1))
        execute([HERE / "train.py", *common, "--out", root / "data-rejected", "--resume", paused], error="Data / encoding mismatch")
        checks.append("changed_data_rejected")
        for name, extra in (("rl-continuous", []), ("rl-paused", ["--stop-after", 4]),
                            ("rl-resumed", ["--resume", root / "rl-paused/latest.pt"])):
            execute([HERE / "rlvr.py", "--iterations", 8, "--out", root / name, *extra])
        a = torch.load(root / "rl-continuous/latest.pt", weights_only=False)
        b = torch.load(root / "rl-resumed/latest.pt", weights_only=False)
        for key in ("logits", "optimizer", "rng", "iteration", "updates", "records"):
            equal(a[key], b[key])
        checks.append("finite_action_rl_exact_rollout_boundary_resume")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status="PASS", torch=str(torch.__version__), checks=checks, count=len(checks),
                  training_code_sha256=source_identity("postcore.py", "train.py"),
                  rl_code_sha256=source_identity("postcore.py", "rlvr.py"),
                  scope="CPU FP32; same layout; normal update/rollout boundary restart; no CUDA/DDP/production fault test")
    save_json(out / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
