"""Phase 0 smoke test: prove chat + embeddings round-trip through Foundry Local."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from malimusavir import foundry  # noqa: E402


def main() -> int:
    print(f"endpoint    : {foundry.discover_endpoint()}")

    try:
        chat_id = foundry.resolve_model(foundry.CHAT_ALIAS)
        embed_id = foundry.resolve_model(foundry.EMBED_ALIAS)
    except foundry.FoundryError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"chat model  : {chat_id}")
    print(f"embed model : {embed_id}")

    t0 = time.perf_counter()
    answer = foundry.chat(
        "Tek kelimeyle cevapla: Turkiye'nin baskenti neresi?",
        max_tokens=32,
    )
    chat_s = time.perf_counter() - t0
    print(f"\nchat  ({chat_s:5.1f}s): {answer!r}")

    t0 = time.perf_counter()
    vectors = foundry.embed(["merhaba dunya", "fatura tutari"])
    embed_s = time.perf_counter() - t0
    print(f"embed ({embed_s:5.1f}s): {len(vectors)} vectors, dim={len(vectors[0])}")

    ok = bool(answer) and len(vectors) == 2 and len(vectors[0]) > 0
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
