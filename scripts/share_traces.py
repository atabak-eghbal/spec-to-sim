"""Make the featured runs' LangSmith traces public and point their links at the public pages.

Run on submission day (it publishes the traces, prompts and model outputs of these runs):
    uv run python scripts/share_traces.py            # list what would be shared
    uv run python scripts/share_traces.py --apply    # share and rewrite app/featured/*/events.jsonl
Undo for one trace: langsmith.Client().unshare_run(<run id>).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    client = None
    if args.apply:
        from langsmith import Client
        client = Client()
    for f in sorted((ROOT / "app/featured").glob("*/events.jsonl")):
        events = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        start = next(e for e in events if e["type"] == "start")
        url = start.get("trace_url") or ""
        m = re.search(r"/run/([0-9a-f-]{36})", url)
        if not m or "public" in url:
            print(f"{f.parent.name}: {'already public' if 'public' in url else 'no trace link'}")
            continue
        if not client:
            print(f"{f.parent.name}: would share run {m.group(1)}")
            continue
        start["trace_url"] = client.share_run(m.group(1))
        f.write_text("".join(json.dumps(e) + "\n" for e in events))
        print(f"{f.parent.name}: {start['trace_url']}")


if __name__ == "__main__":
    main()
