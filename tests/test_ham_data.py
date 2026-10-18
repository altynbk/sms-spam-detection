import pandas as pd
import pytest

from spam_detector.adaptation_data import annotate, campaign_key
from spam_detector.ham_data import fresh_partition, near_groups


def frame(texts, labels):
    return annotate(pd.DataFrame({"text": texts, "label": labels,
                                  "source_row": range(len(texts))}), "fixture")


def test_near_copy_chain_is_one_component_even_without_exact_campaign_match():
    a = "Your delivery has arrived at the local collection centre today. Please collect it before closing time."
    b = a + " Thank you for your patience."
    c = b + " Have a wonderful weekend."
    assert len({campaign_key(t) for t in [a, b, c]}) == 3
    grouped, _ = near_groups(frame([a, b, c, "Lunch at noon?"], [0] * 4), similarity=0.85)
    assert grouped.iloc[:3].group_id.nunique() == 1
    assert grouped.group_id.nunique() == 2
    reordered, _ = near_groups(frame([c, b, a, "Lunch at noon?"], [0] * 4), similarity=0.85)
    assert grouped.iloc[0].group_id == reordered.iloc[0].group_id


def test_conflicting_near_copy_component_is_excluded_in_full():
    text = "Your package is ready for collection from our service desk today."
    result, audit = near_groups(frame([text, text + " Thanks!", "Lunch at noon?"], [0, 1, 0]))
    assert result.source_row.tolist() == [2]
    assert audit["conflicting_components"] == 1
    assert audit["conflicting_rows"] == 2


def test_number_only_and_short_messages_still_keep_campaign_groups_together():
    result, _ = near_groups(frame(["42", "73", "Hi", "Hi!", "Bye"], [0] * 5))
    assert result.group_id.iloc[0] == result.group_id.iloc[1]
    assert result.group_id.iloc[2] == result.group_id.iloc[3]
    assert result.group_id.nunique() == 3


@pytest.mark.parametrize("limit", [0, -0.1, 1.1])
def test_similarity_bound_is_validated(limit):
    with pytest.raises(ValueError):
        near_groups(frame(["hello"], [0]), similarity=limit)


def test_new_holdout_partition_has_declared_group_fractions():
    data = pd.DataFrame({"group_id": [f"group{i}" for i in range(100)], "label": [0, 1] * 50})
    result = fresh_partition(data, seed=20261019)
    assert result.split.value_counts().to_dict() == {"train": 50, "test": 30, "validation": 20}
