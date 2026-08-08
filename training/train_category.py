"""Train the category classifier on frozen qwen3-embedding-0.6b vectors.

The encoder is not touched. Only a multinomial logistic-regression head is trained on
its 1024-dim output, which is why this runs on CPU in seconds rather than needing the
GPU the RX 5600 XT cannot provide for training.

Measured baseline this replaces, on 6 held-out vendors:
    qwen3-4b generative   1/6   ~30s per call
    qwen3-8b generative   2/6   ~63s per call
    embedding zero-shot   3/6   ~0.2s per call

    python training/train_category.py [--force-embed]
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from malimusavir import foundry, rag  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "categories.jsonl"
CACHE = HERE / "data" / "embeddings.npz"
MODEL_OUT = HERE.parent / "malimusavir" / "models" / "category_clf.npz"

#: "diğer" is excluded from training on purpose. It is not a semantic class -- it means
#: "none of the above" -- and the dataset has only 10 such examples because no keyword
#: rule produces them. A classifier cannot learn the absence of a concept from a handful
#: of miscellaneous businesses. Instead the confidence floor in category.py expresses it:
#: when no real category scores highly, the answer is "diğer".
EXCLUDED = {"diğer"}


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_examples() -> tuple[list[str], list[str]]:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA} -- run training/build_dataset.py first")
    texts, labels = [], []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        if ex["category"] in EXCLUDED:
            continue
        texts.append(ex["text"])
        labels.append(ex["category"])
    return texts, labels


def embed_cached(texts: list[str], *, force: bool = False) -> np.ndarray:
    """Embed with an on-disk cache keyed by text hash.

    Embedding is the only slow step here (~1800 texts through a CPU model). Caching by
    content hash means editing the dataset re-embeds only what actually changed, so
    iterating on the data costs seconds rather than minutes.
    """
    cache: dict[str, np.ndarray] = {}
    if CACHE.exists() and not force:
        with np.load(CACHE) as stored:
            cache = {k: stored[k] for k in stored.files}

    missing = [t for t in texts if _key(t) not in cache]
    if missing:
        print(f"embedding {len(missing)} new text(s) ({len(texts) - len(missing)} cached)...")
        for start in range(0, len(missing), rag.BATCH_SIZE):
            batch = missing[start:start + rag.BATCH_SIZE]
            for text, vec in zip(batch, foundry.embed(batch)):
                cache[_key(text)] = np.asarray(vec, dtype=np.float32)
            print(f"  {min(start + rag.BATCH_SIZE, len(missing))}/{len(missing)}", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **cache)
    else:
        print(f"all {len(texts)} embeddings cached")

    matrix = np.vstack([cache[_key(t)] for t in texts]).astype(np.float32)
    # L2-normalise, matching rag.search()'s cosine convention so the geometry the
    # classifier learns is the same geometry retrieval uses.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix


def main() -> int:
    force = "--force-embed" in sys.argv
    texts, labels = load_examples()
    counts = Counter(labels)
    print(f"{len(texts)} examples across {len(counts)} categories "
          f"(excluded: {', '.join(sorted(EXCLUDED))})\n")

    X = embed_cached(texts, force=force)
    y = np.array(labels)
    print(f"\nfeature matrix: {X.shape}")

    # Cross-validated predictions give an honest per-category picture; a single
    # train/test split on 1800 synthetic examples would flatter the result.
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    predicted = cross_val_predict(clf, X, y, cv=folds, n_jobs=1)

    overall = float((predicted == y).mean())
    print(f"\n5-fold cross-validated accuracy: {overall:.1%}\n")
    print(f"{'category':14} {'n':>5} {'acc':>7}   most common confusion")
    print("-" * 62)
    for category in sorted(counts):
        mask = y == category
        acc = float((predicted[mask] == category).mean())
        wrong = Counter(predicted[mask][predicted[mask] != category])
        worst = f"{wrong.most_common(1)[0][0]} x{wrong.most_common(1)[0][1]}" if wrong else "-"
        flag = "  <-- weak" if acc < 0.75 else ""
        print(f"{category:14} {mask.sum():5} {acc:7.1%}   {worst}{flag}")

    clf.fit(X, y)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_OUT,
        coef=clf.coef_.astype(np.float32),
        intercept=clf.intercept_.astype(np.float32),
        labels=np.array(clf.classes_, dtype=object),
        cv_accuracy=np.array([overall], dtype=np.float32),
    )
    size_kb = MODEL_OUT.stat().st_size / 1024
    print(f"\nsaved {MODEL_OUT.name} ({size_kb:.0f} KB, {len(clf.classes_)} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
