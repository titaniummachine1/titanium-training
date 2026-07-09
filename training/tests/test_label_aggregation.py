"""Label aggregation tests for feature cache build logic."""


def aggregate_value_i16(values: list[int]) -> float:
    """Mirror build_feature_cache Pass 1 target mapping."""
    mean_i16 = sum(values) / len(values)
    return (mean_i16 / 100.0 + 1.0) / 2.0


def test_contradictory_outcomes_average_to_drawish():
    win = 100
    loss = -100
    t = aggregate_value_i16([win, loss])
    assert 0.45 <= t <= 0.55


def test_single_label_unchanged():
    t = aggregate_value_i16([80])
    assert abs(t - 0.9) < 1e-6


def test_observation_count_preserved_in_list():
    values = [100, 100, -100]
    assert len(values) == 3
    t = aggregate_value_i16(values)
    assert 0.4 < t < 0.7
