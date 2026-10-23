
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("EMG, IMU, and PPG data")
CLEAN_ROOT = Path("cleaned_data")

WINDOW_SECONDS = 5
STEP_SECONDS = 2.5

SAMPLING_RATES = {
    "EMG": 1000,
    "IMU": 100,
    "PPG": 200,
}

CONDITION = "30-40_ external rotation"
SUBJECT = 9

TASK_MAPPING = {
    "task1_35i": ("30-40_ internal rotation", "Internal", 35),
    "task2_45i": ("40-50_ internal rotation", "Internal", 45),
    "task3_55i": ("50-60_ internal rotation", "Internal", 55),
    "task4_35e": ("30-40_ external rotation", "External", 35),
    "task5_45e": ("40-50_ external rotation", "External", 45),
    "task6_55e": ("50-60_ external rotation", "External", 55),
}


def get_sensor_path(original_path):
    original_path = Path(original_path)

    try:
        relative_path = original_path.relative_to(ROOT)
    except ValueError:
        return original_path

    cleaned_path = CLEAN_ROOT / relative_path

    if cleaned_path.is_file():
        return cleaned_path

    return original_path


def load_window(path, sampling_rate, start_s, end_s):
    path = Path(path)

    data = pd.read_csv(path).apply(
        pd.to_numeric, errors="coerce"
    )

    start_index = int(round(start_s * sampling_rate))
    end_index = int(round(end_s * sampling_rate))

    if start_index < 0 or end_index > len(data):
        raise ValueError(f"Window outside signal boundaries: {path}")

    window = data.iloc[start_index:end_index].to_numpy(dtype=float)

    expected_samples = int(round((end_s - start_s) * sampling_rate))

    if window.shape[0] != expected_samples:
        raise ValueError(f"Unexpected sample count: {path}")

    if not np.isfinite(window).all():
        raise ValueError(f"Invalid sample in window: {path}")

    return window


def add_signal_features(record, prefix, values):
    values = np.asarray(values, dtype=float).reshape(-1)

    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid or empty data in {prefix}")

    record[f"{prefix}_mean"] = float(np.mean(values))
    record[f"{prefix}_std"] = float(np.std(values))
    record[f"{prefix}_rms"] = float(np.sqrt(np.mean(values ** 2)))
    record[f"{prefix}_min"] = float(np.min(values))
    record[f"{prefix}_max"] = float(np.max(values))


def extract_window_features(subject_folder, subject_id, condition,
                            time_seconds, borg_value):
    start_s = time_seconds - WINDOW_SECONDS
    end_s = time_seconds

    record = {
        "participant": subject_id,
        "condition": condition,
        "window_start_s": start_s,
        "window_end_s": end_s,
        "borg_rating": borg_value,
    }

    emg_path = get_sensor_path(
        subject_folder / "EMG data" / "anterior_deltoid.csv"
    )
    emg_window = load_window(
        emg_path, SAMPLING_RATES["EMG"], start_s, end_s
    )
    add_signal_features(
        record, "emg_anterior_deltoid", emg_window
    )

    for sensor_name, filename in [
        ("acc_shoulder", "acc_shoulder.csv"),
        ("gyr_shoulder", "gyr_shoulder.csv"),
        ("mag_shoulder", "mag_shoulder.csv"),
    ]:
        path = get_sensor_path(
            subject_folder / "IMU data" / "Shoulder" / filename
        )
        imu_window = load_window(
            path, SAMPLING_RATES["IMU"], start_s, end_s
        )

        for axis_index in range(imu_window.shape[1]):
            add_signal_features(
                record,
                f"{sensor_name}_axis{axis_index + 1}",
                imu_window[:, axis_index],
            )

    ppg_path = get_sensor_path(
        subject_folder / "PPG data" / "ppg.csv"
    )
    ppg_window = load_window(
        ppg_path, SAMPLING_RATES["PPG"], start_s, end_s
    )
    add_signal_features(record, "ppg", ppg_window)

    return record


print("=" * 65)
print("MODULE 4: FEATURE EXTRACTION")
print("STEPS 33–36: VALIDATION AND CLEANED-PATH SELECTION")

subject_folder = ROOT / CONDITION / f"Subject {SUBJECT}"
borg_path = Path("Borg data") / "borg_data.csv"

if not subject_folder.is_dir():
    raise FileNotFoundError(subject_folder)

if not borg_path.is_file():
    raise FileNotFoundError(borg_path)

borg = pd.read_csv(borg_path)
borg.columns = borg.columns.str.strip()
borg["subject"] = borg["subject"].ffill()

subject_id = f"subject_{SUBJECT}"

subject_borg = borg[
    borg["subject"].astype(str).str.lower() == subject_id
].copy()

if subject_borg.empty:
    raise ValueError(f"No Borg records found for {subject_id}")

matching_rows = []

for _, row in subject_borg.iterrows():
    task_code = str(row.get("task_order", "")).strip()

    if task_code not in TASK_MAPPING:
        continue

    if TASK_MAPPING[task_code][0] == CONDITION:
        matching_rows.append(row)

if len(matching_rows) != 1:
    raise ValueError(
        f"Expected one matching Borg task, found {len(matching_rows)}"
    )

task = matching_rows[0]

rating_times = []

for column in borg.columns:
    if column.endswith("_sec"):
        time_text = column.removesuffix("_sec").strip()

        if time_text.isdigit():
            rating_times.append((int(time_text), column))

rating_times.sort()
ratings = []

for time_seconds, column in rating_times:
    value = pd.to_numeric(
        pd.Series([task[column]]), errors="coerce"
    ).iloc[0]

    if pd.notna(value):
        ratings.append((time_seconds, float(value)))

print(f"Participant: {subject_id}")
print(f"Condition: {CONDITION}")
print(f"Timed Borg ratings: {len(ratings)}")

print("\nSTEP 36: CLEANED-PATH SELECTION CHECK")

test_original = (
    ROOT
    / "40-50_ internal rotation"
    / "Subject 3"
    / "EMG data"
    / "anterior_deltoid.csv"
)

selected_path = get_sensor_path(test_original)

print(f"Original exists: {test_original.is_file()}")
print(f"Selected path: {selected_path}")
print(f"Selected file exists: {selected_path.is_file()}")
print(f"Cleaned copy selected: {selected_path != test_original}")

print("\nSTEP 34: FIRST LABELLED WINDOW")

if not ratings:
    raise ValueError("No timed Borg ratings available.")

first_time, first_borg = ratings[0]

first_record = extract_window_features(
    subject_folder,
    subject_id,
    CONDITION,
    first_time,
    first_borg,
)

print(f"Window: {first_time - WINDOW_SECONDS}–{first_time} seconds")
print(f"Borg label: {first_borg:g}")
print(f"Feature count: {len(first_record)}")

print("\nSTEP 35: ALL LABELLED WINDOWS")

feature_records = []
rejected_windows = []

for time_seconds, borg_value in ratings:
    try:
        record = extract_window_features(
            subject_folder,
            subject_id,
            CONDITION,
            time_seconds,
            borg_value,
        )
        feature_records.append(record)

    except (ValueError, OSError, pd.errors.ParserError) as exc:
        rejected_windows.append({
            "participant": subject_id,
            "condition": CONDITION,
            "window_end_s": time_seconds,
            "reason": str(exc),
        })

all_features = pd.DataFrame(feature_records)

output_path = Path("feature_extraction_subject9.csv")
rejected_path = Path("feature_extraction_rejected_subject9.csv")

if not all_features.empty:
    numeric_features = all_features.select_dtypes(
        include=[np.number]
    )
    missing_count = int(all_features.isna().sum().sum())
    infinite_count = int(
        np.isinf(numeric_features.to_numpy(dtype=float)).sum()
    )

    all_features.to_csv(output_path, index=False)

    print(f"Accepted windows: {len(all_features)}")
    print(f"Feature columns: {len(all_features.columns)}")
    print(f"Missing values: {missing_count}")
    print(f"Infinite values: {infinite_count}")
    print(f"Saved features: {output_path.resolve()}")

    print("\nWindow labels:")
    print(all_features[
        ["window_start_s", "window_end_s", "borg_rating"]
    ].to_string(index=False))
else:
    print("No windows passed validation.")

if rejected_windows:
    pd.DataFrame(rejected_windows).to_csv(
        rejected_path, index=False
    )
    print(f"\nRejected windows: {len(rejected_windows)}")
    print(f"Rejection report: {rejected_path.resolve()}")
    print(pd.DataFrame(rejected_windows).to_string(index=False))
else:
    print("\nRejected windows: 0")

print("\nFeature extraction test complete.")
print("Original sensor files were not modified.")

print("\n" + "=" * 65)
print("STEP 38: CLEANED EMG WINDOW VALIDATION")

test_condition = "40-50_ internal rotation"
test_subject = 3

test_folder = (
    ROOT / test_condition / f"Subject {test_subject}"
)

test_emg_path = get_sensor_path(
    test_folder / "EMG data" / "anterior_deltoid.csv"
)

test_start = 0
test_end = 5

try:
    test_window = load_window(
        test_emg_path,
        SAMPLING_RATES["EMG"],
        test_start,
        test_end,
    )

    print("Unexpected result: window was accepted.")
    print("Check the cleaned file and selected path.")

except ValueError as exc:
    print("Expected result: window rejected.")
    print(f"Reason: {exc}")

print("Step 38 test complete.")

print("\n" + "=" * 65)
print("STEP 39: FULL-DATASET FEATURE EXTRACTION")

CONDITIONS = [
    "30-40_ external rotation",
    "40-50_ external rotation",
    "50-60_ external rotation",
    "30-40_ internal rotation",
    "40-50_ internal rotation",
    "50-60_ internal rotation",
]

borg_all = pd.read_csv(borg_path)
borg_all.columns = borg_all.columns.str.strip()
borg_all["subject"] = borg_all["subject"].ffill()

all_records = []
rejected_records = []
excluded_tasks = []

for participant_number in range(1, 35):
    participant_id = f"subject_{participant_number}"

    participant_rows = borg_all[
        borg_all["subject"].astype(str).str.lower()
        == participant_id
    ]

    for condition in CONDITIONS:
        task_rows = []

        for _, row in participant_rows.iterrows():
            task_code = str(row.get("task_order", "")).strip()

            if (
                task_code in TASK_MAPPING
                and TASK_MAPPING[task_code][0] == condition
            ):
                task_rows.append(row)

        if len(task_rows) != 1:
            rejected_records.append({
                "participant": participant_id,
                "condition": condition,
                "window_end_s": np.nan,
                "reason": (
                    f"Expected one Borg task; found {len(task_rows)}"
                ),
            })
            continue

        task = task_rows[0]
        subject_folder_full = (
            ROOT / condition / f"Subject {participant_number}"
        )

        if not subject_folder_full.is_dir():
            rejected_records.append({
                "participant": participant_id,
                "condition": condition,
                "window_end_s": np.nan,
                "reason": "Task sensor folder missing",
            })
            continue

        # Known unresolved shoulder-IMU duration mismatch.
        if (
            participant_number == 3
            and condition == "30-40_ internal rotation"
        ):
            excluded_tasks.append({
                "participant": participant_id,
                "condition": condition,
                "reason": (
                    "Shoulder IMU streams are 11.05 seconds "
                    "shorter than EMG; alignment unresolved"
                ),
            })
            continue

        task_ratings = []

        for column in borg_all.columns:
            if column.endswith("_sec"):
                time_text = column.removesuffix("_sec").strip()

                if not time_text.isdigit():
                    continue

                value = pd.to_numeric(
                    pd.Series([task[column]]),
                    errors="coerce",
                ).iloc[0]

                if pd.notna(value):
                    task_ratings.append(
                        (int(time_text), float(value))
                    )

        task_ratings.sort()

        for time_seconds, borg_value in task_ratings:
            window_start = time_seconds - WINDOW_SECONDS

            if window_start < 0:
                rejected_records.append({
                    "participant": participant_id,
                    "condition": condition,
                    "window_end_s": time_seconds,
                    "reason": "Window starts before signal",
                })
                continue

            try:
                record = extract_window_features(
                    subject_folder_full,
                    participant_id,
                    condition,
                    time_seconds,
                    borg_value,
                )

                all_records.append(record)

            except (
                ValueError,
                OSError,
                pd.errors.ParserError,
            ) as exc:
                rejected_records.append({
                    "participant": participant_id,
                    "condition": condition,
                    "window_end_s": time_seconds,
                    "reason": str(exc),
                })

full_features = pd.DataFrame(all_records)
full_rejections = pd.DataFrame(rejected_records)
full_exclusions = pd.DataFrame(excluded_tasks)

features_output = Path("fatigue_features_all_subjects.csv")
rejections_output = Path("fatigue_features_rejected_windows.csv")
exclusions_output = Path("fatigue_features_excluded_tasks.csv")

if not full_features.empty:
    full_features.to_csv(features_output, index=False)

if not full_rejections.empty:
    full_rejections.to_csv(rejections_output, index=False)

if not full_exclusions.empty:
    full_exclusions.to_csv(exclusions_output, index=False)

print(f"Feature rows created: {len(full_features)}")
print(f"Participants represented: {full_features['participant'].nunique() if not full_features.empty else 0}")
print(f"Conditions represented: {full_features['condition'].nunique() if not full_features.empty else 0}")
print(f"Rejected windows/tasks: {len(full_rejections)}")
print(f"Excluded unresolved tasks: {len(full_exclusions)}")

if not full_features.empty:
    numeric_features = full_features.select_dtypes(
        include=[np.number]
    )

    print(
        "Missing values in feature table:",
        int(full_features.isna().sum().sum()),
    )
    print(
        "Infinite numeric values:",
        int(np.isinf(numeric_features.to_numpy(dtype=float)).sum()),
    )
    print(f"Features saved to: {features_output.resolve()}")

if not full_rejections.empty:
    print(f"Rejection report: {rejections_output.resolve()}")

if not full_exclusions.empty:
    print(f"Excluded-task report: {exclusions_output.resolve()}")
    print(full_exclusions.to_string(index=False))

print("Step 39 complete.")
print("Original dataset files were not modified.")
