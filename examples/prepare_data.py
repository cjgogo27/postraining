"""Project-original small addition fixture; group commuted pairs before splitting."""
import argparse
import hashlib
import json
from pathlib import Path
from postcore import ENCODING, save_json


def prepare(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": []}
    for a in range(10):
        for b in range(10):
            group = f"{min(a,b)}+{max(a,b)}"
            # Split by a fixed hash of the commutative pair, not a formula that
            # accidentally reserves an entire class of answers (e.g. sums % 5).
            split = "val" if hashlib.sha256(group.encode()).digest()[0] % 5 == 0 else "train"
            splits[split].append(dict(id=f"add-{a}-{b}", group=group, prompt=f"Q:{a}+{b}\nA:",
                                      chosen=str(a + b), rejected=str((a + b + 1) % 19)))
    assert not {r["group"] for r in splits["train"]} & {r["group"] for r in splits["val"]}
    files = {}
    for name, rows in splits.items():
        raw = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows).encode()
        (folder / f"{name}.jsonl").write_bytes(raw)
        files[name] = dict(rows=len(rows), groups=len({r["group"] for r in rows}),
                           answer_values=sorted({int(r["chosen"]) for r in rows}), sha256=hashlib.sha256(raw).hexdigest())
    report = dict(version="addition-original-v1", encoding=ENCODING, files=files,
                  scope="Shared ASCII template and digits; grouped commuted pairs, not a general arithmetic benchmark")
    save_json(folder / "manifest.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/data")
    prepare(parser.parse_args().out)
