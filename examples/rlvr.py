"""Finite-action contextual bandit with group advantages / clipped surrogate.

Not autoregressive LLM GRPO. Each of 12 prompt rows owns 4 logits; there is no
held-out generalization. Exact categorical KL, not a sampled token KL estimator.
"""
import argparse
import json
import os
from pathlib import Path
import platform

import torch
from postcore import group_advantages, source_identity, rng_state, restore_rng, save_json, verify_answer


def fixture():
    tasks = []
    for index in range(12):
        answer = index + 1
        choices = [str(answer), str(answer + 1), str(answer * 10), f"answer={answer}"]
        shift = index % 4
        choices = choices[shift:] + choices[:shift]
        tasks.append(dict(prompt=f"{index}+1", expected=answer, choices=choices,
                          correct_index=choices.index(str(answer))))
    return tasks


def run(args):
    if not 0 < args.iterations or not 1 < args.group or not 0 < args.epochs or not 0 < args.lr < float("inf"):
        raise ValueError("Invalid experiment configuration")
    if args.stop_after is not None and not 0 < args.stop_after <= args.iterations:
        raise ValueError("stop-after out of range")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)
    tasks = fixture()
    rewards = torch.tensor([[verify_answer(choice, task["expected"]) for choice in task["choices"]] for task in tasks])
    logits = torch.nn.Parameter(torch.zeros(12, 4))
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed + 1000)
    identity = source_identity("postcore.py", "rlvr.py")
    iteration, updates, records = 0, 0, []
    if args.resume:
        state = torch.load(args.resume, weights_only=False, map_location="cpu")
        if state["code_sha256"] != identity or state["torch"] != str(torch.__version__) or state["tasks"] != tasks:
            raise ValueError("Code / environment / task identity mismatch")
        if any(state["config"][key] != getattr(args, key) for key in ("iterations", "group", "epochs", "lr", "seed")):
            raise ValueError("Training plan mismatch")
        with torch.no_grad():
            logits.copy_(state["logits"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], generator)
        iteration, updates, records = state["iteration"], state["updates"], state["records"]
    root = Path(args.out)
    if not args.resume and (root / "latest.pt").exists():
        raise ValueError("Use a fresh output directory")
    root.mkdir(parents=True, exist_ok=True)
    target = args.stop_after or args.iterations
    if target <= iteration:
        raise ValueError("Target must exceed saved iteration")
    initial = dict(expected_reward=0.25, greedy_accuracy=0.25)
    first_group = None
    while iteration < target:
        # A fresh on-policy batch every outer iteration; old logp stays frozen
        # across two optimizer epochs. Reference is uniform throughout the run.
        with torch.no_grad():
            old_log_distribution = logits.log_softmax(-1)
            actions = torch.multinomial(logits.softmax(-1), args.group, replacement=True, generator=generator)
            old_logp = old_log_distribution.gather(-1, actions)
            sampled_rewards = rewards.gather(-1, actions)
            advantages = group_advantages(sampled_rewards)
            version = updates
            if first_group is None:
                first_group = dict(prompt=tasks[0], actions=actions[0].tolist(), rewards=sampled_rewards[0].tolist(),
                                   advantages=advantages[0].tolist(), old_logps=old_logp[0].tolist(), actor_version=version)
        clip_fractions = []
        for _ in range(args.epochs):
            distribution = logits.log_softmax(-1)
            current = distribution.gather(-1, actions)
            ratio = (current - old_logp).exp()
            clipped = ratio.clamp(0.8, 1.2)
            surrogate = torch.minimum(ratio * advantages, clipped * advantages).mean()
            kl = (distribution.exp() * (distribution + torch.log(torch.tensor(4.0)))).sum(-1).mean()
            loss = -surrogate + 0.02 * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            clip_fractions.append(float(((ratio < 0.8) | (ratio > 1.2)).float().mean().detach()))
        iteration += 1
        with torch.no_grad():
            p = logits.softmax(-1)
            expected_reward = float((p * rewards).sum(-1).mean())
            accuracy = float(rewards.gather(-1, logits.argmax(-1, keepdim=True)).mean())
            final_kl = float((p * (p.log() + torch.log(torch.tensor(4.0)))).sum(-1).mean())
        records.append(dict(iteration=iteration, actor_version=version, optimizer_updates=updates,
                            sampled_actions=12 * args.group, sampled_reward_mean=float(sampled_rewards.mean()),
                            constant_reward_groups=int((sampled_rewards.std(-1, unbiased=False) == 0).sum()),
                            expected_reward=expected_reward, greedy_accuracy=accuracy,
                            exact_kl_to_uniform=final_kl, clip_fraction_by_epoch=clip_fractions))
    snapshot = dict(logits=logits.detach(), optimizer=optimizer.state_dict(), rng=rng_state(generator),
                    iteration=iteration, updates=updates, records=records, config=vars(args), tasks=tasks,
                    torch=str(torch.__version__), code_sha256=identity)
    torch.save(snapshot, root / "latest.pt.tmp")
    os.replace(root / "latest.pt.tmp", root / "latest.pt")
    report = dict(scope="12x4 table policy, no language generation or unseen tasks; not full LLM GRPO", parameters=48,
                  environment=dict(os=platform.platform(), python=platform.python_version(), torch=str(torch.__version__), threads=1),
                  config=vars(args), code_sha256=identity, tasks=tasks, initial=initial, first_group_of_this_process=first_group,
                  records=records, final=records[-1], sampled_actions_total=iteration * 12 * args.group,
                  conventions=dict(std="population; eps=1e-8", clip=[0.8, 1.2], kl="exact categorical KL(policy || uniform)", kl_beta=0.02))
    save_json(root / "report.json", report)
    print(json.dumps({"final": records[-1], "sampled_actions_total": report["sampled_actions_total"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("iterations", 80), ("group", 8), ("epochs", 2), ("seed", 42)]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--out", default="runs/rlvr")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    run(parser.parse_args())
