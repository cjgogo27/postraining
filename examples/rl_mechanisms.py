"""Offline CPU FP64 RL mechanism checks, not an LLM/agent training benchmark.

Every action/state in this file is enumerated. No downloads, APIs or extra deps.
Run: python examples/rl_mechanisms.py --out runs/rl-mechanisms
"""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import torch


DTYPE = torch.float64


def tensor(values, **kwargs):
    return torch.tensor(values, dtype=DTYPE, **kwargs)


def close(actual, expected):
    torch.testing.assert_close(torch.as_tensor(actual, dtype=DTYPE),
                               torch.as_tensor(expected, dtype=DTYPE), atol=1e-12, rtol=1e-10)


def gae(rewards, values, next_values, terminated, trace_continues, gamma=1., lam=.95):
    # Bootstrap and trace masks have different jobs. At a rollout boundary a
    # nonterminal state can bootstrap even though there is no next recorded delta.
    deltas = rewards + gamma * (1 - terminated) * next_values - values
    advantages = torch.zeros_like(rewards)
    carry = tensor(0.)
    for index in reversed(range(len(rewards))):
        carry = deltas[index] + gamma * lam * trace_continues[index] * carry
        advantages[index] = carry
    return deltas, advantages


def require_support(behavior, target):
    if any(b == 0 and p > 0 for b, p in zip(behavior, target)):
        raise ValueError("target has mass outside behavior support")


def run_checks():
    checks = []
    d, a = gae(tensor([0., 1.]), tensor([.2, .6]), tensor([.6, 0.]),
               tensor([0., 1.]), tensor([1., 0.]))
    close(d, [.4, .4]); close(a, [.78, .4]); close(a + tensor([.2, .6]), [.98, 1.])
    checks.append(dict(name="two_action_gae_and_value_targets", deltas=d.tolist(),
                       advantages=a.tolist(), value_targets=(a + tensor([.2, .6])).tolist()))

    terminal, _ = gae(tensor([0.]), tensor([.6]), tensor([.8]), tensor([1.]), tensor([0.]))
    truncated, _ = gae(tensor([0.]), tensor([.6]), tensor([.8]), tensor([0.]), tensor([0.]))
    close(terminal, [-.6]); close(truncated, [.2])
    checks.append(dict(name="termination_vs_rollout_cut_bootstrap", terminal_delta=terminal.item(),
                       nonterminal_cut_delta=truncated.item(), note="Whether a time limit ends the task is a task definition"))

    cases = []
    for ratio, advantage, expected, slope in [(1.5, 1., 1.2, 0.), (1.5, -1., -1.5, -1.),
                                               (.5, 1., .5, 1.), (.5, -1., -.8, 0.)]:
        rho = tensor(ratio, requires_grad=True)
        objective = torch.minimum(rho * advantage, rho.clamp(.8, 1.2) * advantage)
        objective.backward(); close(objective.detach(), expected); close(rho.grad, slope)
        cases.append(dict(ratio=ratio, advantage=advantage, objective=objective.item(),
                          d_objective_d_ratio=rho.grad.item()))
    checks.append(dict(name="ppo_four_clip_regions_and_gradients", cases=cases))

    behavior, target, reward = tensor([.4, .6]), tensor([.6, .4]), tensor([1., 0.])
    weighted = (behavior * (target / behavior) * reward).sum()
    clipped = (behavior * (target / behavior).clamp(max=1.2) * reward).sum()
    close(weighted, .6); close(clipped, .48)
    close((.6 / .5) * (.5 / .4), .6 / .4)
    checks.append(dict(name="behavior_old_current_ratios_and_is_bias", behavior_reward=.4,
                       exact_target_reward=weighted.item(), capped_is_estimate=clipped.item(),
                       current_over_old=1.2, old_over_behavior=1.25, current_over_behavior=1.5))

    try:
        require_support([1., 0.], [.5, .5])
    except ValueError:
        pass
    else:
        raise AssertionError("missing support was accepted")
    # Both worlds agree on every observed action A (reward .6). B was never seen.
    world_values = [.5 * .6 + .5 * unseen for unseen in [0., 1.]]
    close(world_values, [.3, .8])
    checks.append(dict(name="offline_missing_support_is_unidentifiable", target_values=world_values,
                       dataset="only action A with reward .6; no evidence about B", missing_support_rejected=True))

    z = torch.zeros(10, dtype=DTYPE, requires_grad=True)
    advantage = tensor([1.] * 2 + [-1.] * 8)
    terms = z.exp() * advantage
    seq_mean = (terms[:2].mean() + terms[2:].mean()) / 2
    token_mean = terms.mean()
    seq_grad = torch.autograd.grad(seq_mean, z, retain_graph=True)[0]
    tok_grad = torch.autograd.grad(token_mean, z)[0]
    close(seq_mean.detach(), 0.); close(token_mean.detach(), -.6)
    close(seq_grad, [.25] * 2 + [-.0625] * 8)
    close(tok_grad, [.1] * 2 + [-.1] * 8)
    checks.append(dict(name="sequence_vs_token_reduction_changes_gradient", lengths=[2, 8],
                       sequence_objective=seq_mean.item(), token_objective=token_mean.item(),
                       sequence_gradient=seq_grad.tolist(), token_gradient=tok_grad.tolist()))

    ratios = tensor([4., .25])
    product = ratios.prod(); geometric = ratios.log().mean().exp()
    token_clip = torch.minimum(ratios, ratios.clamp(.8, 1.2)).mean()
    close(product, 1.); close(geometric, 1.); close(token_clip, .725)
    # A geometric likelihood ratio is a surrogate, not an exact IS correction.
    mu, pi = tensor([.5, .5]), tensor([.8, .2])
    geometric_normalization = (mu * (pi / mu).sqrt()).sum()
    assert abs(geometric_normalization.item() - 1) > .01
    checks.append(dict(name="gspo_geometric_ratio_is_not_token_clip_or_exact_is", token_ratios=ratios.tolist(),
                       joint_ratio=product.item(), geometric_ratio=geometric.item(),
                       mean_token_clipped_objective=token_clip.item(),
                       expectation_of_square_root_ratio=geometric_normalization.item()))

    rewards = tensor([0., 0., 1., 1.])
    centered = rewards - rewards.mean()
    loo = rewards - (rewards.sum() - rewards) / 3
    close(loo, [-2/3, -2/3, 2/3, 2/3]); close(loo, centered * 4/3)
    checks.append(dict(name="rloo_excludes_own_reward", group_centered=centered.tolist(), loo=loo.tolist()))

    z = tensor([math.log(.8), math.log(.2)], requires_grad=True)
    p = z.softmax(0); logp = z.log_softmax(0); logq = tensor([.6, .4]).log()
    kl = (p * (logp - logq)).sum()
    full_gradient = torch.autograd.grad(kl, z, retain_graph=True)[0]
    # Score-function estimator, exactly enumerated at one fixed state.
    score_objective = (p.detach() * (logp - logq).detach() * logp).sum()
    score_gradient = torch.autograd.grad(score_objective, z, retain_graph=True)[0]
    naive = (p.detach() * (logp - logq)).sum()
    naive_gradient = torch.autograd.grad(naive, z)[0]
    close(full_gradient, score_gradient); close(naive_gradient, [0., 0.])
    assert full_gradient.abs().max().item() > .1
    checks.append(dict(name="fixed_state_reverse_kl_score_gradient_and_naive_cancellation", kl=kl.item(),
                       full_gradient=full_gradient.tolist(), enumerated_score_gradient=score_gradient.tolist(),
                       naive_gradient=naive_gradient.tolist(), note="Not a full trajectory/prefix-distribution gradient"))

    student = tensor([math.log(.8), math.log(.2)], requires_grad=True)
    teacher = tensor([math.log(.95), math.log(.05)], requires_grad=True)
    sp = student.softmax(0); tq = teacher.softmax(0).detach()
    objective = (sp * (student.log_softmax(0) - tq.log())).sum()
    objective.backward()
    assert teacher.grad is None and student.grad[0] < 0
    same = student.detach().softmax(0)
    close((same * (same.log() - same.log())).sum(), 0.)
    checks.append(dict(name="opsd_privileged_teacher_stop_gradient", student_gradient=student.grad.tolist(),
                       teacher_gradient_absent=True, same_context_equal_distribution_kl=0.,
                       note="Constructed distributions, not a neural teacher or answer-conditioned LLM"))

    z1 = tensor(math.log(.25/.75), requires_grad=True); z2 = tensor(0., requires_grad=True)
    p1, p2 = z1.sigmoid(), z2.sigmoid()
    team = p1 * p2
    gradients = torch.autograd.grad(team, (z1, z2), retain_graph=True)
    # Joint action probabilities and AND reward, conditional counterfactual baselines.
    q1 = torch.stack([1-p1, p1]); q2 = torch.stack([1-p2, p2])
    counterfactual = tensor(0.)
    for a1 in range(2):
        for a2 in range(2):
            probability = (q1[a1] * q2[a2]).detach()
            r = float(a1 * a2)
            b1 = (p1 * a2).detach(); b2 = (p2 * a1).detach()
            counterfactual = counterfactual + probability * ((r-b1)*q1[a1].log() + (r-b2)*q2[a2].log())
    cf_grad = torch.autograd.grad(counterfactual, (z1, z2))
    close(torch.stack(gradients), [.09375, .0625]); close(torch.stack(cf_grad), torch.stack(gradients))
    checks.append(dict(name="two_agent_exact_team_gradient_and_conditional_baseline", team_reward=team.item(),
                       gradients=[g.item() for g in gradients],
                       note="Enumerated two binary policies; not LLM collaboration, CTDE training or a C3 reproduction"))

    returns = tensor([1., 0., 1., 1.])
    local_loo = returns - (returns.sum() - returns) / 3
    close(local_loo, [1/3, -1., 1/3, 1/3])
    checks.append(dict(name="fixed_context_local_replay_loo", returns=returns.tolist(), advantages=local_loo.tolist(),
                       context="same question, prefix and continuation rule; constructed returns, no environment replay"))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/rl-mechanisms")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks = run_checks()
    report = dict(status="PASS", count=len(checks), python=platform.python_version(),
                  torch=str(torch.__version__), device="cpu", dtype="float64", threads=1,
                  checks=checks, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Numerical/interface fixtures only; no LLM PPO/GRPO, agent, MARL or OPD/OPSD training")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
