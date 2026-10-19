"""Talk to the running agent from the command line, the way the app does (same API, same session token).

For reproducing what someone saw on the laptop before touching anything, and for checking a fix against the real model.
Runs inside the eval container, where the runtime is reachable as http://runtime:8000:

    docker compose run --rm eval python scripts/talk.py --as u-ahmed "find me the cheapest flight from Dubai to Beirut next week"
    docker compose run --rm eval python scripts/talk.py --as u-fatima --approve "cancel my Cairo flight, credit please" "yes"
    docker compose run --rm eval python scripts/talk.py --as u-omar --no-reset "what does the policy say about medical cancellations?"

Each message prints the reply, the tools called with their arguments, the reviewer's outcome and the guardrail label.
--approve answers every confirmation card with yes (default: no, nothing is changed); --reset (default) puts the demo
data back to the seed first, exactly as picking the customer in the app does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

from falcon.core.config import settings


async def sse(client: httpx.AsyncClient, path: str, headers: dict, body: dict) -> dict:
    events, ev = [], None
    async with client.stream("POST", f"{settings.runtime_public_url}{path}", headers=headers, json=body, timeout=240) as resp:
        if resp.status_code != 200:
            raise SystemExit(f"{path} -> {resp.status_code}: {(await resp.aread())[:300]!r}")
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                events.append((ev, json.loads(line[5:].strip())))
    for e, d in events:
        if e == "review":
            print(f"  [reviewer round {d.get('round')}: {', '.join(d.get('problems', []))}]")
    return next(d for e, d in events if e == "done")


def show(done: dict) -> None:
    for t in done.get("what_i_checked", []):
        print(f"  tool {t['tool']}({json.dumps(t.get('args'), ensure_ascii=False)}) ok={t.get('ok')} {str(t.get('summary', ''))[:160]!r}")
    v = done.get("verification") or {}
    g = done.get("guardrail") or {}
    print(f"  [verification: {v.get('status')}; guardrail: {g.get('label', '-')} ({g.get('action', '-')}); {done.get('latency_ms')} ms; lang {done.get('lang')}]")
    print("AGENT:", done.get("message", "").strip(), "\n")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("messages", nargs="+")
    ap.add_argument("--as", dest="persona", default="u-ahmed")
    ap.add_argument("--approve", action="store_true", help="answer confirmation cards with yes")
    ap.add_argument("--no-reset", action="store_true", help="keep the current demo data")
    ap.add_argument("--thread", default="talk")
    ap.add_argument("--attach", default=None, help="a PDF/JPEG/PNG to attach (as the paperclip would) before the first message")
    args = ap.parse_args()
    admin = {"X-Admin-Token": settings.admin_token}
    async with httpx.AsyncClient(timeout=240) as client:
        r = await client.post(f"{settings.runtime_public_url}/api/demo/switch", json={"user_id": args.persona, "reset": not args.no_reset}, headers=admin)
        if r.status_code != 200:
            raise SystemExit(f"switch -> {r.status_code}: {r.text[:300]}")
        headers = {"Authorization": f"Bearer {r.json()['token']}", **admin, "X-Falcon-Tags": "talk"}
        print(f"signed in as {args.persona}" + ("" if args.no_reset else " (demo data reset)") + "\n")
        if args.attach:
            import pathlib
            path = pathlib.Path(args.attach)
            ctype = "application/pdf" if path.suffix.lower() == ".pdf" else ("image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png")
            up = await client.post(f"{settings.runtime_public_url}/api/attachments", headers=headers, files={"file": (path.name, path.read_bytes(), ctype)})
            print(f"attached {path.name}: {up.status_code} {up.text[:120]}\n")
        for msg in args.messages:
            print("YOU:", msg)
            done = await sse(client, "/api/chat", headers, {"message": msg, "thread": args.thread})
            show(done)
            while done.get("pending"):
                acts = [a["tool"] for a in done["pending"].get("actions", [])]
                print(f"  [confirmation card: {acts} -> {'approved' if args.approve else 'declined'}]")
                done = await sse(client, "/api/chat/confirm", headers, {"approved": args.approve, "note": "", "thread": args.thread})
                show(done)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
