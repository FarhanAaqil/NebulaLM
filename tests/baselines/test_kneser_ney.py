import math

import numpy as np
import pytest

from nebulalm.baselines.kneser_ney import KneserNey, score_windows, windows_to_stream

V = 12


def _corpus(seed: int = 0, length: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tokens = [int(rng.integers(0, V))]
    for _ in range(length - 1):
        # A Markov chain with structure an n-gram model can learn.
        tokens.append((tokens[-1] * 3 + int(rng.integers(0, 3))) % V)
    docs = np.repeat(np.arange(length // 100), 100)
    return np.array(tokens), docs


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_distribution_sums_to_one_for_any_history(order):
    tokens, docs = _corpus()
    model = KneserNey(order, V).fit(tokens, docs)
    for history in ([], [4], [4, 0], [11, 11, 11, 11], [7, 2, 9, 1]):
        hist = list(history)[-(order - 1):] if order > 1 else []
        width = max(order - 1, 0)
        row = np.zeros(width, dtype=np.int64)
        if hist:
            row[width - len(hist):] = hist
        histories = np.tile(row, (V, 1)) if width else np.zeros((V, 0), dtype=np.int64)
        lengths = np.full(V, len(hist))
        probs = np.exp(model.token_log_probs(histories, np.arange(V), lengths))
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)
        assert (probs > 0).all()


def test_higher_order_beats_unigram_on_structured_text():
    tokens, docs = _corpus(0)
    held, held_docs = _corpus(1, 1000)
    windows = held.reshape(-1, 100)
    segments = held_docs.reshape(-1, 100)
    mask = np.ones_like(windows)
    results = {}
    for order in (1, 3):
        model = KneserNey(order, V).fit(tokens, docs)
        nll, count = score_windows(model, windows, mask, segments)
        results[order] = nll / count
    assert results[3] < results[1] - 0.3
    assert results[1] == pytest.approx(math.log(V), abs=0.2)


def test_history_never_crosses_a_document_boundary():
    tokens, docs = _corpus()
    model = KneserNey(3, V).fit(tokens, docs)
    windows = np.array([[1, 4, 0, 5, 3, 3]])
    mask = np.ones_like(windows)
    joined = score_windows(model, windows, mask, np.array([[0, 0, 0, 0, 0, 0]]))
    split = score_windows(model, windows, mask, np.array([[0, 0, 0, 1, 1, 1]]))
    assert joined[1] == split[1] == 5
    assert joined[0] != split[0]


def test_pad_positions_are_dropped_from_the_training_stream():
    ids = np.array([[5, 6, 7, 0, 0]])
    segments = np.array([[3, 3, 3, -1, -2]])
    tokens, docs = windows_to_stream(ids, segments)
    assert tokens.tolist() == [5, 6, 7] and docs.tolist() == [3, 3, 3]
