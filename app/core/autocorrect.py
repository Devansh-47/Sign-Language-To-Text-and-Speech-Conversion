from __future__ import annotations

from typing import Dict, Iterable, List

from app.config import AUTOCORRECT_VARIANT_CAP

CONFUSION_MATRIX: Dict[str, List[str]] = {
    "c": ["o"],
    "o": ["c"],
    "g": ["h"],
    "h": ["g"],
    "p": ["q", "z"],
    "q": ["p", "z"],
    "z": ["p", "q"],
    "u": ["v", "w"],
    "v": ["u", "w"],
    "w": ["u", "v"],
    "m": ["n"],
    "n": ["m"],
    "y": ["j"],
    "j": ["y"],
}


def _variants(word: str) -> Iterable[str]:
    letters = list(word)
    for idx, ch in enumerate(letters):
        ch_lower = ch.lower()
        if ch_lower not in CONFUSION_MATRIX:
            continue
        for repl in CONFUSION_MATRIX[ch_lower]:
            new_letters = letters.copy()
            new_letters[idx] = repl
            yield "".join(new_letters)


def autocorrect(word: str, lexicon_words: Dict[str, float]) -> List[str]:
    seen = set()
    output: List[str] = []
    for variant in _variants(word):
        if len(output) >= AUTOCORRECT_VARIANT_CAP:
            break
        if variant in seen:
            continue
        seen.add(variant)
        if variant in lexicon_words:
            output.append(variant)
    return output
