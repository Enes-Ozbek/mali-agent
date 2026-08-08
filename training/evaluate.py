"""Compare category strategies on the held-out cases.

These six vendors are the ones both generative models were measured on, and none of
them appear in the training set -- build_dataset.py seeds the *kinds* of business
(nalbur, kuafor, optik, pastane, oto servis, fidanci) without reusing these strings.

    python training/evaluate.py            # keyword / zero-shot / trained classifier
    python training/evaluate.py --llm      # also re-measure the generative baseline
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from malimusavir import classifier, foundry  # noqa: E402
from malimusavir.category import (  # noqa: E402
    CATEGORY_GLOSS,
    CONFIDENCE_MIN,
    _match_keywords,
    classify,
)
from malimusavir.normalize import fold_tr  # noqa: E402

#: (vendor, items, acceptable categories). Several accept more than one label because
#: a plant nursery is defensibly "ev", "market" or "diğer" -- the test is whether the
#: answer is reasonable, not whether it matches one arbitrary choice.
CASES = [
    ("Zeytinburnu Nalbur", "Vida, civata ve matkap ucu", {"ev", "diğer"}),
    ("Bahar Kuafor", "Sac kesimi ve bakim uygulamasi", {"hizmet"}),
    ("Ege Fidancilik", "Meyve fidani ve saksi topragi", {"ev", "diğer", "market"}),
    ("Deniz Optik", "Numarali gozluk cami ve cerceve", {"sağlık"}),
    ("Yildiz Pastanesi", "Yas pasta ve kurabiye", {"yeme-içme", "market"}),
    ("Guven Oto Servis", "Periyodik bakim ve yag degisimi", {"ulaşım", "hizmet"}),
]


def run(name: str, predict) -> tuple[int, float]:
    hits, elapsed = 0, 0.0
    print(f"\n=== {name} ===")
    for vendor, items, expected in CASES:
        t0 = time.perf_counter()
        got = predict(vendor, items)
        dt = time.perf_counter() - t0
        elapsed += dt
        ok = got in expected
        hits += ok
        print(f"  {'OK ' if ok else '   '}{dt:6.2f}s {str(got):12} "
              f"(bekl. {'/'.join(sorted(expected)):18}) {vendor}")
    print(f"  -> {hits}/{len(CASES)}, ortalama {elapsed / len(CASES):.2f}s")
    return hits, elapsed / len(CASES)


def keyword_only(vendor: str, items: str) -> str | None:
    return _match_keywords(fold_tr(f"{vendor}\n{items}"))


def zero_shot(vendor: str, items: str) -> str | None:
    """Nearest category gloss by cosine -- the untrained embedding baseline (3/6)."""
    labels = [label for label, _ in CATEGORY_GLOSS]
    refs = np.asarray(
        foundry.embed([f"{label}: {gloss}" for label, gloss in CATEGORY_GLOSS]),
        dtype=np.float32,
    )
    refs /= np.linalg.norm(refs, axis=1, keepdims=True) + 1e-12
    q = np.asarray(foundry.embed([f"{vendor}. {items}"])[0], dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-12
    return labels[int(np.argmax(refs @ q))]


def trained(vendor: str, items: str) -> str | None:
    result = classifier.classify(f"{vendor}. {items}")
    if result is None:
        return None
    label, confidence = result
    return label if confidence >= CONFIDENCE_MIN else f"(düşük:{label})"


def pipeline(vendor: str, items: str) -> str:
    """The real thing: keyword rules, then classifier, no generative model."""
    return classify(vendor, items, use_llm=False)[0]


def main() -> int:
    if not classifier.is_available():
        print("no trained model -- run training/train_category.py first")
        return 1

    model = classifier.load_model()
    print(f"model: {len(model.labels)} classes, "
          f"{model.cv_accuracy:.1%} cross-validated on training data")

    results = {
        "keyword rules only": run("KEYWORD ONLY", keyword_only),
        "embedding zero-shot": run("EMBEDDING ZERO-SHOT (untrained)", zero_shot),
        "trained classifier": run("TRAINED CLASSIFIER", trained),
        "full pipeline (kw+clf)": run("FULL PIPELINE, no LLM", pipeline),
    }

    if "--llm" in sys.argv:
        from malimusavir.category import classify as full
        results["qwen3-4b generative"] = run(
            "GENERATIVE qwen3-4b",
            lambda v, i: full(v, i, use_llm=True, use_classifier=False)[0],
        )

    print(f"\n{'=' * 58}\n{'method':26} {'acc':>7} {'avg latency':>14}")
    print("-" * 58)
    for name, (hits, avg) in results.items():
        print(f"{name:26} {hits}/{len(CASES):<5} {avg:11.2f}s")
    print("\nbaselines measured earlier: qwen3-4b 1/6 @ ~30s, qwen3-8b 2/6 @ ~63s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
