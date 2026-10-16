"""CPU hand cases for foundations, reward modelling and assistant-only tool traces.

No reward-model or tool-policy training claim: these are numerical/interface fixtures.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F


def tool_fixture():
    messages = [dict(role="user", content="Order A17 status?"),
                dict(role="assistant", content='lookup_order({"id":"A17"})'),
                dict(role="tool", content='{"id":"A17","status":"shipped"}'),
                dict(role="assistant", content="Order A17 is shipped.")]
    ids, actions, segments = [], [], []
    # Original byte fixture, not a real model's chat template. Delimiters are
    # condition-only; assistant content and its EOS are the chosen action targets.
    for message in messages:
        header=list(("<"+message["role"]+">").encode())
        ids.extend(header); actions.extend([False]*len(header))
        content=list(message["content"].encode())+[256]
        segments.append(dict(role=message["role"], start=len(ids), end=len(ids)+len(content)))
        ids.extend(content); actions.extend([message["role"]=="assistant"]*len(content))
    return dict(messages=messages, input_ids=ids[:-1], labels=ids[1:],
                action_mask=actions[1:], segments=segments,
                note="Original byte fixture; assistant content+EOS only; no LLM rollout or actual tool execution")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/teaching-checks")
    args=parser.parse_args()
    torch.set_num_threads(1)
    checks=[]
    z=torch.tensor([.2,.3,.5],dtype=torch.float64).log().requires_grad_()
    loss=F.cross_entropy(z[None,:],torch.tensor([0])); loss.backward()
    torch.testing.assert_close(z.grad,torch.tensor([-.8,.3,.5],dtype=torch.float64))
    new_p=(z-.1*z.grad).softmax(-1)
    checks.append(dict(name="sft_logit_gradient_and_sgd", loss=float(loss.detach()), new_target_probability=float(new_p[0].detach())))
    scores=torch.tensor([2.,1.],dtype=torch.float64,requires_grad=True)
    rm=F.softplus(-(scores[0]-scores[1])); rm.backward()
    assert abs(float(rm.detach())-.31326168751822286)<1e-12
    torch.testing.assert_close(scores.grad,torch.tensor([-1/(1+math.e),1/(1+math.e)],dtype=torch.float64))
    checks.append(dict(name="reward_model_bt_loss_and_gradient", probability=1/(1+math.exp(-1)), loss=float(rm.detach())))
    # Exact expectation over two actions: score-function identity, action-independent baseline.
    z=torch.tensor(math.log(.25/.75),dtype=torch.float64,requires_grad=True)
    p=torch.stack([z.sigmoid(),1-z.sigmoid()]); reward=torch.tensor([1.,0.],dtype=torch.float64)
    exact=torch.autograd.grad((p*reward).sum(),z,retain_graph=True)[0]
    surrogate=(p.detach()*(reward-.7)*p.log()).sum()
    estimate=torch.autograd.grad(surrogate,z)[0]
    torch.testing.assert_close(exact,estimate); assert abs(float(exact)-.1875)<1e-12
    checks.append(dict(name="policy_gradient_exact_expectation_and_baseline", gradient=float(exact)))
    # Match the JS visualisation: one distinct sample per action, A fixed.
    z=torch.zeros(4,dtype=torch.float64,requires_grad=True)
    a=torch.tensor([-1.,-1.,1.,1.],dtype=torch.float64)
    pg=-(a*z.log_softmax(0)).mean();pg.backward()
    torch.testing.assert_close(z.grad,torch.tensor([.25,.25,-.25,-.25],dtype=torch.float64))
    probability=(z-.4*z.grad).softmax(0)
    checks.append(dict(name="visual_group_step_matches_autograd", probabilities=probability.detach().tolist()))
    fixture=tool_fixture()
    for segment in fixture["segments"]:
        # labels[k] is original token k+1: map original content indices back once.
        mask=fixture["action_mask"][segment["start"]-1:segment["end"]-1]
        assert all(mask) if segment["role"]=="assistant" else not any(mask)
    first,tool,last=fixture["segments"][1:]
    assert tool["end"]<=last["start"] and tool["start"]>first["start"]
    checks.append(dict(name="two_assistant_spans_tool_observation_excluded_shifted_mask", action_positions=sum(fixture["action_mask"])))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"tool-trace.json").write_text(json.dumps(fixture,indent=2)+"\n",encoding="utf8")
    report=dict(status="PASS",count=len(checks),torch=str(torch.__version__),checks=checks,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope="CPU FP64 hand cases and byte tool-message fixture; no trained neural reward model or tool LLM RL")
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report))


if __name__=="__main__": main()
