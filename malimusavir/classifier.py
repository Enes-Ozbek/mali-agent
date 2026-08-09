"""Trained category classifier: a logistic-regression head over frozen embeddings.

The encoder (qwen3-embedding-0.6b) is not fine-tuned -- only this head is trained, by
training/train_category.py. That is what lets the whole thing train on CPU in seconds
and classify in ~0.2s, against ~30s for asking qwen3-4b to do the same job.

Measured on 6 held-out vendors: qwen3-4b 1/6, qwen3-8b 2/6, embeddings zero-shot 3/6.

The artifact is ~200 KB of coefficients, committed to the repo, so nothing needs
retraining to use the project.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from . import foundry
from . import paths

MODEL_PATH = paths.resource("malimusavir", "models", "category_clf.npz")


class _Model:
    __slots__ = ("coef", "intercept", "labels", "cv_accuracy")

    def __init__(self, coef: np.ndarray, intercept: np.ndarray,
                 labels: list[str], cv_accuracy: float) -> None:
        self.coef = coef
        self.intercept = intercept
        self.labels = labels
        self.cv_accuracy = cv_accuracy


@functools.lru_cache(maxsize=1)
def load_model() -> _Model | None:
    """The trained head, or None when it has not been built yet.

    Returning None rather than raising keeps the project usable without the artifact:
    category.py simply falls through to its other strategies, exactly as it does when
    Foundry Local is unreachable.
    """
    if not MODEL_PATH.exists():
        return None
    with np.load(MODEL_PATH, allow_pickle=True) as stored:
        return _Model(
            coef=stored["coef"].astype(np.float32),
            intercept=stored["intercept"].astype(np.float32),
            labels=[str(x) for x in stored["labels"]],
            cv_accuracy=float(stored["cv_accuracy"][0]),
        )


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / (exp.sum() + 1e-12)


def classify(text: str) -> tuple[str, float] | None:
    """Predict a category. Returns (label, confidence) or None if unavailable.

    None means "cannot answer" -- no model artifact, no Foundry Local, or empty input --
    and is distinct from a low-confidence prediction, which is returned so the caller
    can decide what to do with it against its own threshold.
    """
    model = load_model()
    if model is None or not text.strip():
        return None

    try:
        vector = np.asarray(foundry.embed([text])[0], dtype=np.float32)
    except Exception:  # noqa: BLE001 - unreachable model must not break ingest
        return None

    # Same L2 normalisation the head was trained under; skipping it silently shifts
    # every score.
    vector /= np.linalg.norm(vector) + 1e-12
    probabilities = _softmax(model.coef @ vector + model.intercept)
    best = int(np.argmax(probabilities))
    return model.labels[best], float(probabilities[best])


def is_available() -> bool:
    """Whether a trained artifact exists (does not check Foundry reachability)."""
    return load_model() is not None
