import random

import pytest

from spam_detector.attacks import ATTACK_TYPES, LEET, attack_messages, obfuscate


@pytest.mark.parametrize("attack", ATTACK_TYPES)
def test_zero_intensity_identity_ham_unchanged_and_seed_reproducibility(attack):
    texts = ["FREE CALL £100!", "ham ABC", "Сәлем🙂", "  ", "!!!", ""]
    labels = [1, 0, 1, 0, 1, 1]
    assert attack_messages(texts, labels, 9, 0, attack) == texts
    a = attack_messages(texts, labels, 9, 0.5, attack)
    assert a == attack_messages(texts, labels, 9, 0.5, attack)
    assert a[1] == texts[1] and a[3] == texts[3]
    assert a[2] == texts[2] and a[4] == texts[4] and a[5] == texts[5]


def test_leet_uppercase_and_ineligible_characters():
    chars = "".join(LEET).upper()
    attacked = obfuscate(chars, random.Random(42), 1)
    assert all(a != b for a, b in zip(chars, attacked))
    assert obfuscate("Uu", random.Random(1), 1) == "Vv"
    untouched = "DFGHJKMNPQRVWXYZ 0123456789 @£€$!? Привет Ә🙂"
    assert obfuscate(untouched, random.Random(1), 1) == untouched


def test_separators_only_insert_at_ascii_letter_boundaries():
    raw = "AB 12 éЯ C-D"
    attacked = obfuscate(raw, random.Random(3), 1, "separators")
    assert len(attacked) == len(raw) + 1
    assert attacked[0] == "A" and attacked[1] in ".-_" and attacked[2:] == raw[1:]


def test_unicode_is_case_aware_and_does_not_normalize_unrelated_text():
    assert obfuscate("AaCcXx", random.Random(1), 1, "unicode") == "АаСсХх"
    raw = "éЯ ӘЖ 123 £?🙂"
    assert obfuscate(raw, random.Random(1), 1, "unicode") == raw


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_intensity_rejected_even_for_ham(p):
    with pytest.raises(ValueError, match="intensity"):
        attack_messages(["hello"], [0], 42, p)


def test_invalid_attack_and_alignment():
    with pytest.raises(ValueError, match="Unknown attack"):
        attack_messages(["hello"], [0], 42, 0, "missing")
    with pytest.raises(ValueError, match="binary label"):
        attack_messages(["hello"], [], 42, 0.1)
