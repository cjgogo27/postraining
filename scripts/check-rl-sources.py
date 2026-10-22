"""Optional network metadata/anchor audit. Does NOT validate scientific claims.

The manual claim-to-section review is in SOURCE_AUDIT.md. No external packages.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urldefrag, unquote
from urllib.request import Request, urlopen


SOURCES = [
    ("verl PPO", "https://verl.readthedocs.io/en/latest/algo/ppo.html", "Proximal Policy Optimization", None),
    ("verl Rollout Correction", "https://verl.readthedocs.io/en/latest/algo/rollout_corr.html", "Rollout Correction", "Yingru Li"),
    ("DAPO", "https://arxiv.org/abs/2503.14476v2", "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", "Qiying Yu"),
    ("GSPO", "https://arxiv.org/abs/2507.18071v2", "Group Sequence Policy Optimization", "Chujie Zheng"),
    ("Agentic survey", "https://arxiv.org/abs/2509.02547v5", "The Landscape of Agentic Reinforcement Learning for LLMs: A Survey", "Guibin Zhang"),
    ("Agent Lightning", "https://arxiv.org/abs/2508.03680v1", "Agent Lightning: Train ANY AI Agents with Reinforcement Learning", "Xufang Luo"),
    ("C3", "https://arxiv.org/abs/2603.06859v1", "Contextual Counterfactual Credit Assignment for Multi-Agent Reinforcement Learning in LLM Collaboration", "Yanjun Chen"),
    ("OPSD", "https://arxiv.org/abs/2601.18734v3", "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models", "Siyan Zhao"),
    ("Rethinking OPD", "https://arxiv.org/abs/2604.13016v2", "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe", "Yaxuan Li"),
    ("Semi-OPD", "https://arxiv.org/abs/2610.11291v1", "When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better", "Siyan Zhao"),
]


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__(); self.ids = set(); self.meta = {}; self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "meta" and "name" in attrs and "content" in attrs:
            self.meta.setdefault(attrs["name"], []).append(attrs["content"])


def fetch(url):
    try:
        with urlopen(Request(url, headers={"User-Agent": "PostrainingCitationCheck/1.0"}), timeout=45) as response:
            raw = response.read(); html = raw.decode("utf8")
            return dict(url=url, status="PASS", http=response.status,
                        sha256=hashlib.sha256(raw).hexdigest(), html=html)
    except Exception as error:
        return dict(url=url, status="FAIL", error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/rl-source-checks")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    files = [repo / "src/content/docs/00-reading-guides/06-rl-map.mdx",
             repo / "src/content/docs/07-appendix/e-agentic-multi-agent-rl.mdx",
             repo / "src/content/docs/07-appendix/f-rl-reading-list.mdx"]
    files.extend((repo / "src/content/docs/04-online-rl-distillation").glob("*.mdx"))
    files.append(repo / "src/content/docs/05-evaluation-diagnosis/04-training-rollout-orchestration.mdx")
    links = set()
    for file in files:
        for url in re.findall(r"\]\((https://(?:arxiv\.org|verl\.readthedocs\.io)/[^\s)]+)\)", file.read_text(encoding="utf8")):
            if "#" in url:
                links.add(url)
    urls = sorted({x[1] for x in SOURCES} | {urldefrag(x)[0] for x in links})
    with ThreadPoolExecutor(max_workers=3) as pool:
        fetched = dict(zip(urls, pool.map(fetch, urls)))
    initial_failures = [dict(url=url, error=item.get("error")) for url, item in fetched.items() if item["status"] != "PASS"]
    # One bounded retry for transient fetch failures; failed outcomes remain errors.
    if initial_failures:
        with ThreadPoolExecutor(max_workers=2) as pool:
            retry_urls = [item["url"] for item in initial_failures]
            fetched.update(zip(retry_urls, pool.map(fetch, retry_urls)))
    metadata = []
    for name, url, title, author in SOURCES:
        item = fetched[url]
        result = dict(name=name, url=url, status="FAIL")
        if item["status"] == "PASS":
            page = Page(item["html"])
            visible = unescape(re.sub(r"<[^>]+>", " ", item["html"]))
            norm = lambda s: re.sub(r"\s+", " ", s).strip().casefold()
            actual_title = page.meta.get("citation_title", [title if norm(title) in norm(visible) else ""])[0]
            authors = page.meta.get("citation_author", [])
            valid = norm(actual_title) == norm(title) and (author is None or author in authors or author in visible)
            result.update(status="PASS" if valid else "FAIL", title=actual_title, authors=authors,
                          citation_date=page.meta.get("citation_date"), sha256=item["sha256"])
        else:
            result["error"] = item.get("error")
        metadata.append(result)
    anchors = []
    for url in sorted(links):
        base, fragment = urldefrag(url); item = fetched[base]
        ok = item["status"] == "PASS" and unquote(fragment) in Page(item["html"]).ids
        anchors.append(dict(url=url, status="PASS" if ok else "FAIL", error=item.get("error")))
    ok = all(x["status"] == "PASS" for x in metadata + anchors)
    report = dict(status="PASS" if ok else "FAIL", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  primary_count=len(metadata), primary_sources=metadata, external_anchors=anchors,
                  fetch_failures_before_one_retry=initial_failures,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Metadata title/author and HTML fragment presence only; manual claim review in SOURCE_AUDIT.md")
    out = repo / args.out; out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(dict(status=report["status"], primary_sources=len(metadata), anchors=len(anchors),
                          failures=[x for x in metadata + anchors if x["status"] != "PASS"]), ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
