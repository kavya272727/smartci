"""Train a small logistic-regression model to estimate CI build-failure risk.

The bundled CSV is synthetic teaching data. Replace it with real, labelled CI
history before using predictions for operational decisions.
"""
import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "build_history.csv"
FEATURES = [
    "changed_files",
    "lines_changed",
    "test_files_changed",
    "previous_failures",
]


def load_dataset(path=DEFAULT_DATA):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = set(FEATURES + ["build_failed"])
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Training CSV is missing required columns.")
        for row in reader:
            rows.append([float(row[name]) for name in FEATURES] + [float(row["build_failed"])])
    if len(rows) < 20:
        raise ValueError("At least 20 labelled build-history rows are required.")
    values = np.asarray(rows, dtype=float)
    labels = values[:, -1]
    if len(np.unique(labels)) < 2:
        raise ValueError("Training data must contain both successful and failed builds.")
    return values[:, :-1], labels


def train_model(path=DEFAULT_DATA, learning_rate=0.15, iterations=3500):
    """Fit logistic regression with NumPy gradient descent."""
    x, y = load_dataset(path)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    x_scaled = (x - means) / scales
    x_augmented = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    weights = np.zeros(x_augmented.shape[1], dtype=float)

    for _ in range(iterations):
        logits = np.clip(x_augmented @ weights, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = (x_augmented.T @ (probabilities - y)) / len(y)
        gradient[1:] += 0.01 * weights[1:]  # L2 regularization
        weights -= learning_rate * gradient

    return weights, means, scales


def predict_failure_probability(metrics, model):
    weights, means, scales = model
    values = np.asarray([metrics[name] for name in FEATURES], dtype=float)
    scaled = (values - means) / scales
    logit = float(weights[0] + scaled @ weights[1:])
    logit = max(-30.0, min(30.0, logit))
    return 1.0 / (1.0 + np.exp(-logit))


def classify_risk(probability):
    if probability < 0.35:
        return "LOW"
    if probability < 0.65:
        return "MEDIUM"
    return "HIGH"


def collect_git_metrics():
    """Estimate change metrics from the latest commit; use safe defaults if unavailable."""
    try:
        result = subprocess.run(
            ["git", "diff", "--numstat", "HEAD~1", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        changed_files = 0
        lines_changed = 0
        test_files_changed = 0
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added, deleted, filename = parts
            changed_files += 1
            if added.isdigit():
                lines_changed += int(added)
            if deleted.isdigit():
                lines_changed += int(deleted)
            if filename.startswith("tests/") and filename.endswith(".py"):
                test_files_changed += 1
        return {
            "changed_files": changed_files,
            "lines_changed": lines_changed,
            "test_files_changed": test_files_changed,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"changed_files": 1, "lines_changed": 0, "test_files_changed": 0}


def main():
    parser = argparse.ArgumentParser(description="Predict CI build-failure risk.")
    parser.add_argument("--changed-files", type=int, default=None)
    parser.add_argument("--lines-changed", type=int, default=None)
    parser.add_argument("--test-files-changed", type=int, default=None)
    parser.add_argument(
        "--previous-failures", type=int, default=0,
        help="Number of recent failures supplied by the user or CI history (default: 0).",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    metrics = collect_git_metrics()
    for name in ("changed_files", "lines_changed", "test_files_changed"):
        supplied = getattr(args, name.replace("-", "_"))
        if supplied is not None:
            metrics[name] = supplied
    metrics["previous_failures"] = args.previous_failures

    model = train_model(args.data)
    probability = predict_failure_probability(metrics, model)
    risk = classify_risk(probability)

    print("## Build-Failure Prediction")
    print()
    print(f"- **Predicted failure probability:** {probability * 100:.1f}%")
    print(f"- **Risk level:** {risk}")
    print("- **Input metrics:** " + ", ".join(f"{key.replace('_', ' ')} = {value}" for key, value in metrics.items()))
    print()
    print("This is an early-warning estimate, not a build result. The CI tests still determine whether the build passes.")
    print("Training data note: the bundled dataset is synthetic demonstration data; replace it with real labelled CI history for production use.")


if __name__ == "__main__":
    main()
