"""Optional real pretrained-base bridge; CPU, pinned SmolLM2, no service/API calls.

--download explicitly permits an initial public model download. Default is cache-only.
This is a new SFT run, not strict resume; weights remain in ignored runs/.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import torch
from torch.nn import functional as F

MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def records():
    def row(index, status):
        return dict(id=f"order-{index}", prompt=f"Return only the order status.\nRecord: order={index}; status={status}\nAnswer:", target=" " + status)
    statuses = ["shipped", "pending", "cancelled"]
    train = [row(100 + i, statuses[i % 3]) for i in range(18)]
    target = [row(300 + i, statuses[i % 3]) for i in range(6)]
    retention = [dict(id="capital", prompt="The capital of France is", target=" Paris."),
                 dict(id="freezing", prompt="Water freezes at", target=" 0 degrees Celsius."),
                 dict(id="days", prompt="There are seven days in a", target=" week.")]
    return train, target, retention


def encode(tokenizer, row):
    # Explicit token boundary: prompt and completion encoded separately; both
    # evaluation and training use this same token protocol, with no chat template.
    prefix = tokenizer.encode(row["prompt"], add_special_tokens=False)
    answer = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
    if not prefix or len(prefix + answer) > 128:
        raise ValueError("Empty/oversize sample; no silent truncation")
    ids = torch.tensor([prefix + answer])
    labels = ids[:, 1:]
    mask = torch.arange(labels.shape[1]) >= len(prefix) - 1
    return ids[:, :-1], labels, mask, prefix


def nll(model, encoded):
    inputs, labels, mask, _ = encoded
    values = F.cross_entropy(model(input_ids=inputs, use_cache=False).logits[0], labels[0], reduction="none")
    return values[mask].sum(), int(mask.sum())


@torch.no_grad()
def evaluate(model, tokenizer, rows):
    model.eval()
    outputs, total, count = [], 0.0, 0
    for row in rows:
        encoded = encode(tokenizer, row)
        value, size = nll(model, encoded)
        total += float(value)
        count += size
        ids = torch.tensor([encoded[3]])
        generated, ended = [], False
        # Manual greedy loop makes the EOS versus length-cap distinction explicit.
        for _ in range(12):
            token = int(model(input_ids=ids, use_cache=False).logits[0, -1].argmax())
            if token == tokenizer.eos_token_id:
                ended = True
                break
            generated.append(token)
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
        text = tokenizer.decode(generated, skip_special_tokens=False)
        normalized = text.strip()
        outputs.append(dict(**row, generated=text, generated_ids=generated,
                            end_reason="eos" if ended else "max_new_tokens",
                            format_ok=normalized in {"shipped", "pending", "cancelled"},
                            content_exact=normalized == row["target"].strip(),
                            strict_success=ended and normalized == row["target"].strip(),
                            target_positions=size, target_nll=float(value) / size))
    return dict(response_nll=total/count, response_positions=count, samples=len(rows),
                content_exact=sum(x["content_exact"] for x in outputs),
                strict_success=sum(x["strict_success"] for x in outputs), outputs=outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/pretrained-sft")
    parser.add_argument("--cache", default="runs/model-cache")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00002)
    args = parser.parse_args()
    if args.steps <= 0 or args.threads <= 0 or not 0 < args.lr < float("inf"):
        parser.error("Invalid steps, threads or learning rate")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh output directory")
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started = time.perf_counter()
    snapshot = snapshot_download(MODEL, revision=REVISION, cache_dir=args.cache,
                                 local_files_only=not args.download,
                                 allow_patterns=["*.json", "*.txt", "*.safetensors"])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                                               dtype=torch.float32, attn_implementation="eager")
    model.config.use_cache = False
    train, target, retention = records()
    encoded = [encode(tokenizer, row) for row in train]
    before = dict(target=evaluate(model, tokenizer, target), retention=evaluate(model, tokenizer, retention))
    print(json.dumps(dict(event="before", nll=before["target"]["response_nll"])), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    generator = torch.Generator().manual_seed(1042)
    history, positions = [], 0
    for step in range(1, args.steps + 1):
        model.train()
        # Two variable-length examples; reduce by total response positions.
        indices = torch.randint(len(train), (2,), generator=generator).tolist()
        denominator = sum(int(encoded[i][2].sum()) for i in indices)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for index in indices:
            value, _ = nll(model, encoded[index])
            (value / denominator).backward()
            loss_value += float(value.detach()) / denominator
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        positions += denominator
        history.append(dict(step=step, loss=loss_value, indices=indices, response_positions=positions))
        if step % 4 == 0:
            print(json.dumps(history[-1]), flush=True)
    after = dict(target=evaluate(model, tokenizer, target), retention=evaluate(model, tokenizer, retention))
    model.save_pretrained(out / "model", safe_serialization=True)
    tokenizer.save_pretrained(out / "model")
    # Bind downloaded model/tokenizer/config contents; do not trust a name alone.
    files = {p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
             for p in Path(snapshot).iterdir() if p.is_file()}
    report = dict(model=MODEL, revision=REVISION, model_files_sha256=files,
                  parameters=sum(p.numel() for p in model.parameters()), config=vars(args), seed=42,
                  environment=dict(os=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
                                   transformers=importlib.metadata.version("transformers"),
                                   huggingface_hub=importlib.metadata.version("huggingface_hub"), device="cpu", precision="FP32"),
                  template="plain-order-status-v1; separately-encoded prompt/response; no chat template; EOS included",
                  data=dict(train=train, target=target, retention=retention), before=before, after=after,
                  updates=args.steps, examples_seen=2*args.steps, response_positions=positions, history=history,
                  elapsed_seconds=time.perf_counter()-started,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="One seed; shared English template; 6 new order IDs; 3 retention probes, not a general capability or safety benchmark; no RL/tool training")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps(dict(event="done", before=before["target"]["strict_success"], after=after["target"]["strict_success"], seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
