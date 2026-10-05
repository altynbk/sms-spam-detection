"""Nonadaptive, character-level attacks on raw SMS; no model access."""

import math
import random

import numpy as np

from .preprocessing import raw_batch

# Original notebook mapping. Only eligible ASCII letters can change.
LEET = {"a": ["@", "4"], "e": ["3"], "i": ["1", "!"], "o": ["0"], "s": ["$", "5"],
        "l": ["1"], "t": ["7"], "c": ["(", "<"], "u": ["v"], "b": ["8"]}
UNICODE = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
           "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
           "O": "О", "P": "Р", "T": "Т", "X": "Х"}
ATTACK_TYPES = ("leet", "separators", "unicode")


def obfuscate(text, rng, intensity=0.3, attack_type="leet"):
    if not isinstance(text, str):
        raise TypeError("Attack input must be a raw SMS string")
    if not math.isfinite(intensity) or not 0 <= intensity <= 1:
        raise ValueError("Attack intensity must be finite and between 0 and 1")
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"Unknown attack: {attack_type}")
    if intensity == 0:
        return text
    result = []
    for i, char in enumerate(text):
        new = char
        if attack_type == "leet" and char.isascii() and char.lower() in LEET:
            if rng.random() < intensity:
                new = rng.choice(LEET[char.lower()])
                if char.isupper():
                    new = new.upper()
        elif attack_type == "unicode" and char in UNICODE:
            if rng.random() < intensity:
                new = UNICODE[char]
        result.append(new)
        if attack_type == "separators" and i + 1 < len(text):
            # Only boundaries between adjacent ASCII alphabetic characters.
            neighbor = text[i + 1]
            if char.isascii() and char.isalpha() and neighbor.isascii() and neighbor.isalpha():
                if rng.random() < intensity:
                    result.append(rng.choice([".", "-", "_"]))
    return "".join(result)


def attack_messages(texts, labels, seed, intensity, attack_type="leet"):
    values = raw_batch(texts)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(values) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("One binary label (ham=0, spam=1) is required per SMS")
    # Validate configuration even when the batch contains only ham.
    rng = random.Random(seed)
    obfuscate("", rng, intensity, attack_type)
    return [obfuscate(text, rng, intensity, attack_type) if label == 1 else text
            for text, label in zip(values, labels)]
