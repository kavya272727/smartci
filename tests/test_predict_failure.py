from scripts.predict_failure import (
    classify_risk,
    load_dataset,
    predict_failure_probability,
    train_model,
)


def test_training_data_contains_success_and_failure_examples():
    _, labels = load_dataset()
    assert 0 in labels
    assert 1 in labels


def test_more_previous_failures_increase_predicted_risk():
    model = train_model()
    ordinary = {
        "changed_files": 3,
        "lines_changed": 80,
        "test_files_changed": 1,
        "previous_failures": 0,
    }
    repeated_failures = dict(ordinary, previous_failures=4)
    assert predict_failure_probability(repeated_failures, model) > predict_failure_probability(ordinary, model)


def test_risk_classification_thresholds():
    assert classify_risk(0.20) == "LOW"
    assert classify_risk(0.50) == "MEDIUM"
    assert classify_risk(0.80) == "HIGH"
