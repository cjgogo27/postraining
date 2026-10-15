"""Offline post-training primitives; byte causal LM, explicit response-only masks.

The random tiny model is a mechanism fixture, not a pretrained foundation model.
"""
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

VOCAB, EOS = 257, 256
ENCODING = "utf8-bytes-0-255;eos=256;right-padding-id=0-with-loss-mask"


class TinyPolicy(nn.Module):
    def __init__(self, width=64, length=32):
        super().__init__()
        self.length = length
        self.token = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(length, width)
        self.block = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, VOCAB, bias=False)
        self.head.weight = self.token.weight
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, std=0.02)

    def forward(self, ids):
        length = ids.shape[1]
        if length > self.length:
            raise ValueError("Input exceeds position table")
        hidden = self.token(ids) + self.position(torch.arange(length))
        causal = torch.ones(length, length, dtype=torch.bool).triu(1)
        # Padding is only at the right; earlier real positions cannot see it.
        return self.head(self.norm(self.block(hidden, src_mask=causal, is_causal=True)))


def batch(records, length=32, field="chosen"):
    inputs = torch.zeros(len(records), length, dtype=torch.long)
    labels = torch.zeros_like(inputs)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for row, item in enumerate(records):
        prefix = list(item["prompt"].encode("utf8"))
        response = list(item[field].encode("utf8")) + [EOS]
        sequence = prefix + response
        if not prefix or len(sequence) - 1 > length:
            raise ValueError("Empty prompt or sequence too long; no silent truncation")
        size = len(sequence) - 1
        inputs[row, :size] = torch.tensor(sequence[:-1])
        labels[row, :size] = torch.tensor(sequence[1:])
        mask[row, len(prefix) - 1:size] = True
    return inputs, labels, mask


def sequence_logps(model, tensors):
    inputs, labels, mask = tensors
    selected = model(inputs).log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(-1), mask.sum(-1)


def sft_loss(model, tensors):
    values, counts = sequence_logps(model, tensors)
    return -values.sum() / counts.sum()


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    margin = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return F.softplus(-margin).mean(), margin


def group_advantages(rewards):
    # Teaching convention: population standard deviation, eps=1e-8.
    return (rewards - rewards.mean(-1, keepdim=True)) / rewards.std(-1, keepdim=True, unbiased=False).clamp_min(1e-8)


def verify_answer(text, expected):
    # Exact canonical decimal string; no substring/last-number extraction.
    return float(text == str(expected))


def source_identity(*names):
    digest = hashlib.sha256()
    for name in names:
        source = (Path(__file__).parent / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        digest.update(name.encode() + b"\0" + source.encode("utf8") + b"\0")
    return digest.hexdigest()


def rng_state(generator):
    return dict(python=random.getstate(), torch=torch.get_rng_state(), sampler=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["sampler"])


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf8")
