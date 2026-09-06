import numpy as np

from bt4012_fraud.evaluation import evaluate_binary_predictions


def test_evaluation_counts_and_scores() -> None:
    result = evaluate_binary_predictions(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.9, 0.2]), threshold=0.5
    )
    assert result["true_negatives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_positives"] == 1
    assert 0 <= result["pr_auc"] <= 1
