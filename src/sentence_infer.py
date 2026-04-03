"""Line-based segmentation HTR (horizontal projection). Used by main.py --segmentation line."""
import os
import string
from collections import Counter
from typing import Dict, Tuple

import cv2
import numpy as np

from spellchecker import SpellChecker

from dataloader_iam import Batch
from model import Model
from preprocessor import Preprocessor

# Same extras as main.infer() for domain words
_DEFAULT_EXTRA_WORDS = ("work", "level", "line", "online", "meeting", "confirm")

_spell: SpellChecker = None
_spell_extra_loaded: set = set()


def _get_spell() -> SpellChecker:
    """One SpellChecker per process (consensus runs many times; avoid reloading dict each time)."""
    global _spell
    if _spell is None:
        _spell = SpellChecker()
        _spell.word_frequency.load_words(list(_DEFAULT_EXTRA_WORDS))
    return _spell


def _spell_token(spell: SpellChecker, token: str) -> str:
    """Correct alphabetic core; keep digits/punctuation as-is."""
    if any(ch.isdigit() for ch in token):
        return token
    lead = ""
    trail = ""
    core = token
    while core and core[0] in string.punctuation + '"\'':
        lead += core[0]
        core = core[1:]
    while core and core[-1] in string.punctuation + '"\'':
        trail = core[-1] + trail
        core = core[:-1]
    if not core:
        return token
    c = spell.correction(core)
    fixed = c if c is not None else core
    return lead + fixed + trail


def apply_spell_multiline(text: str, extra_words=()) -> str:
    """Spell-correct whitespace-separated tokens per line; skip tokens that contain digits (codes)."""
    if not text or not text.strip():
        return text
    spell = _get_spell()
    new_extras = [w for w in extra_words if w and w not in _spell_extra_loaded]
    if new_extras:
        spell.word_frequency.load_words(new_extras)
        _spell_extra_loaded.update(new_extras)
    out_lines = []
    for line in text.split("\n"):
        fixed = [_spell_token(spell, w) for w in line.split()]
        out_lines.append(" ".join(fixed))
    return "\n".join(out_lines).strip()


def imread_gray(path: str):
    """Read grayscale image; works on Windows for paths with # and non-ASCII."""
    path = os.path.abspath(os.path.normpath(path))
    if not os.path.isfile(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def split_lines(img: np.ndarray):
    """Split binarized image into horizontal text lines via projection."""
    hist = np.sum(img, axis=1)
    lines = []
    start = None

    for i, val in enumerate(hist):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            if i - start > 20:
                lines.append(img[start:i, :])
            start = None

    if start is not None:
        lines.append(img[start:, :])

    return lines


def full_image_predict(
    model: Model,
    image_path: str,
    verbose: bool = True,
    spell_check: bool = True,
) -> str:
    """
    Preprocess full page, split into lines, run HTR (single batched infer per image).
    Optional English spell correction (like main.infer word mode).
    """
    img = imread_gray(image_path)
    if img is None:
        if verbose:
            print(f"ERROR: Could not read image: {image_path}")
        return ""

    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, None, fx=2, fy=2)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.dilate(img, kernel, iterations=1)

    lines = split_lines(img)
    if not lines:
        return ""

    preprocessor = Preprocessor((128, 32))
    processed = []
    for line in lines:
        if line.shape[1] < line.shape[0]:
            line = cv2.rotate(line, cv2.ROTATE_90_CLOCKWISE)
        processed.append(preprocessor.process_img(line))

    batch = Batch(processed, None, len(processed))
    recognized, _ = model.infer_batch(batch, False)
    raw = "\n".join(recognized).strip()
    if spell_check and raw:
        raw = apply_spell_multiline(raw)
    return raw


def full_image_predict_consensus(
    model: Model,
    image_path: str,
    n_runs: int,
    spell_check: bool = True,
) -> Tuple[str, Dict[str, int]]:
    """
    Run full_image_predict multiple times and return the most common full output.
    Deterministic models usually repeat the same string; use this when you want a stable
    choice if runs ever differ, or to average out rare numerical noise.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    outputs = []
    for i in range(n_runs):
        # Avoid repeating the same read error message on every pass
        verbose = i == 0
        outputs.append(
            full_image_predict(model, image_path, verbose=verbose, spell_check=spell_check)
        )
    counts = Counter(outputs)
    winner = counts.most_common(1)[0][0]
    return winner, dict(counts)
