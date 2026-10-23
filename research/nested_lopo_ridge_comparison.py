
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


BASE_DIR = Path(__file__).resolve().parent

NPZ_FILE = BASE_DIR / "lstm_sequences.npz"
METADATA_FILE = BASE_DIR / "lstm_sequence_metadata.csv"
LSTM_FILE = BASE_DIR / "consistent_lopo_augmentation_predictions.csv"

ALPHAS = [0.1, 1.0, 10.0, 100.0]
INNER_SPLITS = 4

PREDICTIONS_FILE = BASE_DIR / "nested_lopo_ridge_predictions.csv"
SUMMARY_FILE = BASE_DIR / "nested_lopo_ridge_summary.csv"
PARTICIPANT_FILE = BASE_DIR / "nested_lopo_ridge_by_participant.csv"
ALPHA_FILE = BASE_DIR / "nested_lopo_ridge_selected_alphas.csv"


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    correlation = np.nan
    if np.std(actual) > 0 and np.std(predicted) > 0:
        correlation = np.corrcoef(actual, predicted)[0, 1]

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "Correlation": correlation,
        "Within_1_Borg_Percent": (
            np.mean(np.abs(actual - predicted) <= 1.0) * 100
        ),
    }


def evaluate_inner_alpha(X, y, groups, alpha):
    """Select alpha using training participants only."""

    group_count = len(np.unique(groups))
    n_splits = min(INNER_SPLITS, group_count)

    if n_splits < 2:
        raise ValueError("Not enough training participants for inner CV.")

    splitter = GroupKFold(n_splits=n_splits)
    actual_all = []
    predicted_all = []

    for train_idx, val_idx in splitter.split(X, y, groups):
        X_inner_train = X[train_idx]
        X_inner_val = X[val_idx]
        y_inner_train = y[train_idx]
        y_inner_val = y[val_idx]

        scaler = StandardScaler()
        X_inner_train_scaled = scaler.fit_transform(X_inner_train)
        X_inner_val_scaled = scaler.transform(X_inner_val)

        model = Ridge(alpha=alpha)
        model.fit(X_inner_train_scaled, y_inner_train)

        predictions = model.predict(X_inner_val_scaled)

        actual_all.extend(y_inner_val)
        predicted_all.extend(predictions)

    return mean_absolute_error(actual_all, predicted_all)


def main():
    data = np.load(NPZ_FILE)
    X = data["X"]
    y = data["y"].astype(float)

    metadata = pd.read_csv(METADATA_FILE)
    lstm = pd.read_csv(LSTM_FILE)

    if X.ndim != 3:
        raise ValueError(f"Expected 3D X, got {X.shape}.")

    if len(X) != len(y) or len(y) != len(metadata):
        raise ValueError("X, y, and metadata row counts differ.")

    if len(lstm) != len(metadata):
        raise ValueError("LSTM prediction row count differs.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X or y contains non-finite values.")

    if not np.allclose(
        y, metadata["target_borg"].to_numpy(dtype=float),
        equal_nan=True
    ):
        raise ValueError("Labels do not match metadata.")

    participants = metadata["participant"].astype(str).to_numpy()

    if not np.array_equal(
        participants,
        lstm["participant"].astype(str).to_numpy()
    ):
        raise ValueError("Participant ordering differs from LSTM file.")

    if not np.allclose(
        y, lstm["actual_borg"].to_numpy(dtype=float),
        equal_nan=True
    ):
        raise ValueError("Actual Borg labels differ from LSTM file.")

    if "baseline_prediction" not in lstm.columns:
        raise ValueError("LSTM prediction column is missing.")

    # Flatten 3 timesteps x 55 features into 165 features.
    X_flat = X.reshape(X.shape[0], -1)

    unique_participants = np.unique(participants)

    prediction_rows = []
    alpha_rows = []

    print("DATA VALIDATION PASSED")
    print("Sequences:", len(y))
    print("Participants:", len(unique_participants))
    print("Feature shape per sequence:", X_flat.shape[1])
    print("Alpha candidates:", ALPHAS)
    print("Inner CV splits:", INNER_SPLITS)
    print()

    for fold, held_out in enumerate(unique_participants, start=1):
        test_mask = participants == held_out
        train_mask = ~test_mask

        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)

        X_train = X_flat[train_indices]
        y_train = y[train_indices]
        groups_train = participants[train_indices]

        X_test = X_flat[test_indices]
        y_test = y[test_indices]

        # Choose alpha using only the outer training participants.
        alpha_scores = {}

        for alpha in ALPHAS:
            score = evaluate_inner_alpha(
                X_train, y_train, groups_train, alpha
            )
            alpha_scores[alpha] = score

        selected_alpha = min(alpha_scores, key=alpha_scores.get)

        # Refit the selected model on the complete outer training set.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=selected_alpha)
        model.fit(X_train_scaled, y_train)

        ridge_predictions = model.predict(X_test_scaled)

        # Training-only median baseline.
        median_prediction = float(np.median(y_train))

        # Existing LSTM predictions, kept unchanged.
        lstm_predictions = lstm.iloc[test_indices][
            "baseline_prediction"
        ].to_numpy(dtype=float)

        alpha_rows.append({
            "held_out_participant": held_out,
            "selected_alpha": selected_alpha,
            "inner_cv_mae_alpha_0.1": alpha_scores[0.1],
            "inner_cv_mae_alpha_1": alpha_scores[1.0],
            "inner_cv_mae_alpha_10": alpha_scores[10.0],
            "inner_cv_mae_alpha_100": alpha_scores[100.0],
        })

        for j, idx in enumerate(test_indices):
            prediction_rows.append({
                "participant": held_out,
                "target_borg": y[idx],
                "median_prediction": median_prediction,
                "ridge_prediction": ridge_predictions[j],
                "lstm_prediction": lstm_predictions[j],
                "selected_alpha": selected_alpha,
            })

        print(
            f"Fold {fold:02d}/{len(unique_participants)} | "
            f"{held_out} | alpha={selected_alpha} | "
            f"inner MAE={alpha_scores[selected_alpha]:.4f}"
        )

    predictions = pd.DataFrame(prediction_rows)
    alpha_results = pd.DataFrame(alpha_rows)

    models = {
        "Median baseline": "median_prediction",
        "Nested Ridge": "ridge_prediction",
        "Existing LSTM": "lstm_prediction",
    }

    summary_rows = []

    for model_name, column in models.items():
        metrics = calculate_metrics(
            predictions["target_borg"],
            predictions[column]
        )
        metrics["Model"] = model_name
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)

    participant_rows = []

    for participant, group in predictions.groupby("participant"):
        row = {"participant": participant}

        for model_name, column in models.items():
            row[f"{model_name}_MAE"] = mean_absolute_error(
                group["target_borg"], group[column]
            )

        participant_rows.append(row)

    participant_summary = pd.DataFrame(participant_rows)

    mean_mae_rows = []
    for model_name in models:
        mean_mae_rows.append({
            "Model": model_name,
            "Mean_Participant_MAE": participant_summary[
                f"{model_name}_MAE"
            ].mean(),
        })

    summary = summary.merge(
        pd.DataFrame(mean_mae_rows), on="Model"
    )

    predictions.to_csv(PREDICTIONS_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    participant_summary.to_csv(PARTICIPANT_FILE, index=False)
    alpha_results.to_csv(ALPHA_FILE, index=False)

    print("\nNESTED LOPO MODEL COMPARISON")
    print(summary.round(4).to_string(index=False))

    print("\nPARTICIPANT-LEVEL RESULTS")
    print(
        "Nested Ridge beats median:",
        int(
            (
                participant_summary["Nested Ridge_MAE"]
                < participant_summary["Median baseline_MAE"]
            ).sum()
        ),
        "/", len(participant_summary)
    )

    print(
        "Nested Ridge beats LSTM:",
        int(
            (
                participant_summary["Nested Ridge_MAE"]
                < participant_summary["Existing LSTM_MAE"]
            ).sum()
        ),
        "/", len(participant_summary)
    )

    print("\nSELECTED ALPHA COUNTS")
    print(alpha_results["selected_alpha"].value_counts().sort_index())

    print("\nSaved files:")
    print(PREDICTIONS_FILE.name)
    print(SUMMARY_FILE.name)
    print(PARTICIPANT_FILE.name)
    print(ALPHA_FILE.name)


if __name__ == "__main__":
    main()
