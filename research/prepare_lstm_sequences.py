
from pathlib import Path
import numpy as np
import pandas as pd

DATA_FILE = Path("fatigue_features_all_subjects.csv")
SEQUENCE_FILE = Path("lstm_sequences.npz")
METADATA_FILE = Path("lstm_sequence_metadata.csv")

SEQUENCE_LENGTH = 3

df = pd.read_csv(DATA_FILE)

metadata_columns = {
    "participant",
    "condition",
    "window_start_s",
    "window_end_s",
    "borg_rating",
}

feature_columns = [
    col for col in df.select_dtypes(include=[np.number]).columns
    if col not in metadata_columns
]

df = df.sort_values(
    ["participant", "condition", "window_start_s"]
).reset_index(drop=True)

sequences = []
labels = []
metadata = []

for (participant, condition), group in df.groupby(
    ["participant", "condition"], sort=False
):
    group = group.sort_values("window_start_s").reset_index(drop=True)

    for end_idx in range(SEQUENCE_LENGTH - 1, len(group)):
        selected = group.iloc[
            end_idx - SEQUENCE_LENGTH + 1 : end_idx + 1
        ]

        # Ensure the windows are consecutive in time.
        if len(selected) != SEQUENCE_LENGTH:
            continue

        starts = selected["window_start_s"].to_numpy(dtype=float)
        ends = selected["window_end_s"].to_numpy(dtype=float)

        # Require a consistent 10-second label interval.
        if not np.allclose(np.diff(starts), 10.0):
            continue

        x = selected[feature_columns].to_numpy(dtype=np.float32)
        y = float(selected.iloc[-1]["borg_rating"])

        if not np.isfinite(x).all() or not np.isfinite(y):
            continue

        sequences.append(x)
        labels.append(y)

        metadata.append({
            "participant": participant,
            "condition": condition,
            "sequence_start_s": starts[0],
            "sequence_end_s": ends[-1],
            "target_borg": y,
        })

if not sequences:
    raise ValueError("No valid sequences were created.")

X = np.stack(sequences)
y = np.asarray(labels, dtype=np.float32)
metadata_df = pd.DataFrame(metadata)

np.savez_compressed(
    SEQUENCE_FILE,
    X=X,
    y=y,
    feature_columns=np.asarray(feature_columns),
)

metadata_df.to_csv(METADATA_FILE, index=False)

print("STEP 43: LSTM SEQUENCE PREPARATION")
print(f"Input feature rows: {len(df)}")
print(f"Sequence length: {SEQUENCE_LENGTH} windows")
print(f"Feature count per window: {len(feature_columns)}")
print(f"Sequence tensor shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Unique participants: {metadata_df['participant'].nunique()}")
print(f"Unique conditions: {metadata_df['condition'].nunique()}")
print(f"Non-finite sequence values: {int((~np.isfinite(X)).sum())}")
print(f"Non-finite labels: {int((~np.isfinite(y)).sum())}")
print(f"Sequences saved to: {SEQUENCE_FILE.resolve()}")
print(f"Metadata saved to: {METADATA_FILE.resolve()}")
print("Step 43 complete.")
