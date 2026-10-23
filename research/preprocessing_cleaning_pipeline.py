
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("EMG, IMU, and PPG data")

# Task 1: Validate dataset paths and required files
print("=" * 65)
print("TASK 1: INPUT VALIDATION")

if not ROOT.exists():
    raise FileNotFoundError(f"Dataset folder not found: {ROOT.resolve()}")

conditions = [
    "30-40_ external rotation",
    "40-50_ external rotation",
    "50-60_ external rotation",
    "30-40_ internal rotation",
    "40-50_ internal rotation",
    "50-60_ internal rotation",
]

required_files = [
    Path("EMG data/anterior_deltoid.csv"),
    Path("IMU data/Shoulder/acc_shoulder.csv"),
    Path("IMU data/Shoulder/gyr_shoulder.csv"),
    Path("IMU data/Shoulder/mag_shoulder.csv"),
    Path("PPG data/ppg.csv"),
]

checked = 0
missing = []

for condition in conditions:
    for subject_number in range(1, 35):
        folder = ROOT / condition / f"Subject {subject_number}"

        for relative_path in required_files:
            path = folder / relative_path
            checked += 1

            if not path.is_file():
                missing.append(str(path))

print(f"Dataset root: {ROOT.resolve()}")
print(f"Required file paths checked: {checked}")
print(f"Missing required files: {len(missing)}")

for path in missing[:20]:
    print("MISSING:", path)

if missing:
    print("Input validation found missing files.")
else:
    print("Input validation passed for all required files.")


# Task 2: Inspect invalid and extreme EMG values
print("\n" + "=" * 65)
print("TASK 2: NON-DESTRUCTIVE EMG QUALITY CHECK")

emg_cases = [
    (
        "40-50_ internal rotation",
        3,
        "anterior_deltoid.csv",
    ),
    (
        "40-50_ internal rotation",
        3,
        "infraspinatus.csv",
    ),
]

for condition, subject_number, filename in emg_cases:
    path = (
        ROOT
        / condition
        / f"Subject {subject_number}"
        / "EMG data"
        / filename
    )

    df = pd.read_csv(path)
    column = df.columns[0]
    values = pd.to_numeric(df[column], errors="coerce")
    array = values.to_numpy(dtype=float)

    non_numeric = int(values.isna().sum())
    non_finite = int((~np.isfinite(array)).sum())
    extreme = int((np.isfinite(array) & (np.abs(array) > 1.0)).sum())

    print(f"\nFile: {path}")
    print(f"Rows: {len(df)}")
    print(f"Non-numeric or missing values: {non_numeric}")
    print(f"Infinite values: {non_finite}")
    print(f"Samples with absolute amplitude above 1 mV: {extreme}")

    valid = array[np.isfinite(array)]

    if valid.size:
        print(f"Minimum finite value: {valid.min():.6g}")
        print(f"Maximum finite value: {valid.max():.6g}")

    print("Original file unchanged; no samples were removed or replaced.")

print("\nStep 23 complete.")

print("\n" + "=" * 65)
print("STEP 24: CORRUPTED REGION AND CLEANING AUDIT")

audit_records = []

for condition, subject_number, filename in emg_cases:
    path = (
        ROOT / condition / f"Subject {subject_number}"
        / "EMG data" / filename
    )

    df = pd.read_csv(path)
    values = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(
        dtype=float
    )

    invalid = ~np.isfinite(values)
    extreme = np.isfinite(values) & (np.abs(values) > 1.0)
    affected = invalid | extreme
    indices = np.flatnonzero(affected)

    if len(indices):
        breaks = np.where(np.diff(indices) > 1)[0]
        starts = np.r_[indices[0], indices[breaks + 1]]
        ends = np.r_[indices[breaks], indices[-1]]

        regions = list(zip(starts, ends))
    else:
        regions = []

    affected_count = int(affected.sum())
    percentage = 100 * affected_count / len(values) if len(values) else 0

    print(f"\nFile: {path}")
    print(f"Total samples: {len(values)}")
    print(f"Affected samples: {affected_count}")
    print(f"Affected percentage: {percentage:.4f}%")
    print(f"Contiguous affected regions: {len(regions)}")

    for start, end in regions:
        print(f"Rows {start}–{end} (inclusive)")

    audit_records.append({
        "condition": condition,
        "subject": subject_number,
        "file": filename,
        "total_samples": len(values),
        "invalid_samples": int(invalid.sum()),
        "extreme_samples": int(extreme.sum()),
        "affected_samples": affected_count,
        "affected_percentage": percentage,
        "regions": str(regions),
        "action": "flagged_only_original_unchanged",
    })

audit_path = Path("emg_cleaning_audit.csv")
pd.DataFrame(audit_records).to_csv(audit_path, index=False)

print(f"\nAudit report saved to: {audit_path.resolve()}")
print("Original sensor files remain unchanged.")

print("\n" + "=" * 65)
print("STEP 25: SAFE EMG CLEANING")

clean_root = Path("cleaned_data")
clean_audit_records = []

for condition, subject_number, filename in emg_cases:
    source_path = (
        ROOT / condition / f"Subject {subject_number}"
        / "EMG data" / filename
    )

    df = pd.read_csv(source_path)
    column = df.columns[0]
    values = pd.to_numeric(df[column], errors="coerce")

    # Only these previously verified regions are marked as corrupted.
    if filename == "anterior_deltoid.csv":
        start, end = 39, 338
    elif filename == "infraspinatus.csv":
        start, end = 0, 237
    else:
        raise ValueError(f"No verified cleaning rule for {filename}")

    if end >= len(df):
        raise ValueError(f"Cleaning region exceeds file length: {source_path}")

    cleaned = df.copy()
    cleaned[column] = values
    cleaned.loc[start:end, column] = np.nan

    relative_path = source_path.relative_to(ROOT)
    output_path = clean_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    clean_audit_records.append({
        "source_file": str(source_path),
        "cleaned_file": str(output_path),
        "column": column,
        "first_corrupted_row": start,
        "last_corrupted_row": end,
        "samples_marked_missing": end - start + 1,
        "reason": "confirmed_extreme_numerical_corruption",
    })

    print(f"\nSource: {source_path}")
    print(f"Cleaned copy: {output_path}")
    print(f"Rows marked NaN: {start}–{end}")
    print(f"Samples marked missing: {end - start + 1}")
    print(f"Original row count: {len(df)}")
    print(f"Cleaned row count: {len(cleaned)}")

audit_output = Path("emg_cleaning_changes.csv")
pd.DataFrame(clean_audit_records).to_csv(audit_output, index=False)

print(f"\nChange audit saved to: {audit_output.resolve()}")
print("Original sensor files were not modified.")
print("Step 25 complete.")

print("\n" + "=" * 65)
print("STEP 26: VERIFY CLEANED EMG FILES")

for condition, subject_number, filename in emg_cases:
    source_path = (
        ROOT / condition / f"Subject {subject_number}"
        / "EMG data" / filename
    )
    output_path = (
        clean_root / condition / f"Subject {subject_number}"
        / "EMG data" / filename
    )

    original = pd.read_csv(source_path)
    cleaned = pd.read_csv(output_path)

    column = original.columns[0]

    if filename == "anterior_deltoid.csv":
        start, end = 39, 338
    else:
        start, end = 0, 237

    original_values = pd.to_numeric(
        original[column], errors="coerce"
    ).to_numpy(dtype=float)

    cleaned_values = pd.to_numeric(
        cleaned[column], errors="coerce"
    ).to_numpy(dtype=float)

    affected_region_is_nan = np.isnan(cleaned_values[start:end + 1]).all()

    unaffected = np.ones(len(original_values), dtype=bool)
    unaffected[start:end + 1] = False

    unaffected_values_match = np.allclose(
        original_values[unaffected],
        cleaned_values[unaffected],
        equal_nan=True
    )

    row_count_matches = len(original) == len(cleaned)

    print(f"\nFile: {filename}")
    print(f"Affected region entirely NaN: {affected_region_is_nan}")
    print(f"Unaffected values unchanged: {unaffected_values_match}")
    print(f"Row count preserved: {row_count_matches}")

    if not (
        affected_region_is_nan
        and unaffected_values_match
        and row_count_matches
    ):
        raise RuntimeError(f"Verification failed: {output_path}")

print("\nStep 26 complete: both cleaned files passed verification.")

print("\n" + "=" * 65)
print("STEP 27: IMU ANOMALY INVESTIGATION")

imu_cases = [
    (
        "30-40_ internal rotation",
        27,
        "Sternum",
        "acc_sternum.csv",
        5090,
        5091,
    ),
    (
        "50-60_ external rotation",
        25,
        "Hand",
        "acc_hand.csv",
        1092,
        1099,
    ),
    (
        "50-60_ external rotation",
        25,
        "Forearm",
        "acc_forearm.csv",
        1092,
        1099,
    ),
    (
        "40-50_ internal rotation",
        26,
        "Hand",
        "acc_hand.csv",
        3568,
        3574,
    ),
]

for condition, subject, location, filename, start, end in imu_cases:
    path = (
        ROOT / condition / f"Subject {subject}"
        / "IMU data" / location / filename
    )

    df = pd.read_csv(path)
    numeric = df.apply(pd.to_numeric, errors="coerce")

    print(f"\nFile: {path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Inspecting rows {start}–{end}, plus surrounding samples:")

    left = max(0, start - 3)
    right = min(len(df), end + 4)

    print(numeric.iloc[left:right].to_string())

    values = numeric.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size:
        print(f"Overall minimum: {finite.min():.6g}")
        print(f"Overall maximum: {finite.max():.6g}")

    print("Original file unchanged.")

print("\nStep 27A complete.")

print("\n" + "=" * 65)
print("STEP 27B: CROSS-LOCATION IMU COMPARISON")

condition = "30-40_ internal rotation"
subject = 27
start, end = 5088, 5093

locations = ["Sternum", "Shoulder", "Upper arm", "Forearm", "Hand"]

for location in locations:
    path = (
        ROOT / condition / f"Subject {subject}"
        / "IMU data" / location
        / f"acc_{location.lower().replace(' ', '_')}.csv"
    )

    if not path.is_file():
        print(f"\nFile not found: {path}")
        continue

    df = pd.read_csv(path)
    numeric = df.apply(pd.to_numeric, errors="coerce")

    print(f"\nLocation: {location}")
    print(f"File: {path}")

    if len(numeric) <= start:
        print("Not enough rows for this comparison.")
        continue

    print(numeric.iloc[start:min(end + 1, len(numeric))].to_string())

print("\nStep 27B complete. No sensor files were modified.")

print("\n" + "=" * 65)
print("STEP 28: IMU MAGNITUDE AND CHANGE ANALYSIS")

for condition, subject, location, filename, start, end in imu_cases:
    path = (
        ROOT / condition / f"Subject {subject}"
        / "IMU data" / location / filename
    )

    df = pd.read_csv(path)
    values = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    magnitude = np.sqrt(np.sum(values ** 2, axis=1))
    change = np.full(len(values), np.nan)

    if len(values) > 1:
        change[1:] = np.sqrt(np.sum(np.diff(values, axis=0) ** 2, axis=1))

    left = max(1, start - 3)
    right = min(len(df), end + 4)

    print(f"\nSubject {subject} | {location} | {condition}")
    print("row | magnitude | change from previous row")

    for i in range(left, right):
        print(f"{i:5d} | {magnitude[i]:10.4f} | {change[i]:12.4f}")

    print("Original file unchanged.")

print("\nStep 28 complete.")

print("\n" + "=" * 65)
print("STEP 29: CLEAN CONFIRMED STERNUM ANOMALY")

condition = "30-40_ internal rotation"
subject = 27
filename = "acc_sternum.csv"
start, end = 5090, 5091

source_path = (
    ROOT / condition / f"Subject {subject}"
    / "IMU data" / "Sternum" / filename
)

df = pd.read_csv(source_path)
cleaned = df.copy()

# Mark only the two previously investigated rows as missing.
cleaned.iloc[start:end + 1, :] = np.nan

relative_path = source_path.relative_to(ROOT)
output_path = clean_root / relative_path
output_path.parent.mkdir(parents=True, exist_ok=True)
cleaned.to_csv(output_path, index=False)

# Verify the saved copy.
saved = pd.read_csv(output_path)
affected_is_nan = saved.iloc[start:end + 1, :].isna().all().all()

original_values = df.to_numpy(dtype=float)
saved_values = saved.to_numpy(dtype=float)

unaffected = np.ones(len(df), dtype=bool)
unaffected[start:end + 1] = False

unaffected_unchanged = np.allclose(
    original_values[unaffected],
    saved_values[unaffected],
    equal_nan=True
)

row_count_preserved = len(df) == len(saved)

print(f"Original file: {source_path}")
print(f"Cleaned copy: {output_path}")
print(f"Rows marked NaN: {start}–{end}")
print(f"Affected region entirely NaN: {affected_is_nan}")
print(f"Unaffected values unchanged: {unaffected_unchanged}")
print(f"Row count preserved: {row_count_preserved}")

if not (affected_is_nan and unaffected_unchanged and row_count_preserved):
    raise RuntimeError("Sternum cleaning verification failed.")

sternum_audit = pd.DataFrame([{
    "source_file": str(source_path),
    "cleaned_file": str(output_path),
    "first_affected_row": start,
    "last_affected_row": end,
    "samples_marked_missing": end - start + 1,
    "reason": "extreme_isolated_acceleration_spike",
    "action": "flagged_as_NaN_in_cleaned_copy",
}])

sternum_audit.to_csv("imu_cleaning_changes.csv", index=False)

print("IMU change audit saved to imu_cleaning_changes.csv")
print("Original sensor file unchanged.")
print("Step 29 complete.")

print("\n" + "=" * 65)
print("STEP 30: DOCUMENT PENDING IMU ANOMALIES")

pending_imu_cases = [
    {
        "condition": "50-60_ external rotation",
        "subject": 25,
        "location": "Hand",
        "filename": "acc_hand.csv",
        "start_row": 1092,
        "end_row": 1099,
        "reason": "Short burst of extreme acceleration readings",
    },
    {
        "condition": "50-60_ external rotation",
        "subject": 25,
        "location": "Forearm",
        "filename": "acc_forearm.csv",
        "start_row": 1092,
        "end_row": 1099,
        "reason": "Extreme readings near the hand acceleration burst",
    },
    {
        "condition": "40-50_ internal rotation",
        "subject": 26,
        "location": "Hand",
        "filename": "acc_hand.csv",
        "start_row": 3568,
        "end_row": 3574,
        "reason": "Short burst of extreme acceleration readings",
    },
]

pending_records = []

for case in pending_imu_cases:
    source_path = (
        ROOT / case["condition"] / f'Subject {case["subject"]}'
        / "IMU data" / case["location"] / case["filename"]
    )

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    df = pd.read_csv(source_path)
    start = case["start_row"]
    end = case["end_row"]

    if end >= len(df):
        raise ValueError(f"Invalid row range in {source_path}")

    pending_records.append({
        "source_file": str(source_path),
        "subject": case["subject"],
        "condition": case["condition"],
        "sensor_location": case["location"],
        "first_suspicious_row": start,
        "last_suspicious_row": end,
        "suspicious_samples": end - start + 1,
        "reason": case["reason"],
        "status": "pending_review_not_modified",
    })

    print(f"\nFile: {source_path}")
    print(f"Rows flagged for review: {start}–{end}")
    print("Status: pending review; original and cleaned data unchanged")

pending_report = Path("imu_pending_review.csv")
pd.DataFrame(pending_records).to_csv(pending_report, index=False)

print(f"\nPending-review report: {pending_report.resolve()}")
print("No sensor values were changed in this step.")
print("Step 30 complete.")

print("\n" + "=" * 65)
print("STEP 31: CLEANING COVERAGE VERIFICATION")

cleaned_cases = [
    (
        "40-50_ internal rotation", 3,
        "EMG data/anterior_deltoid.csv", 39, 338
    ),
    (
        "40-50_ internal rotation", 3,
        "EMG data/infraspinatus.csv", 0, 237
    ),
    (
        "30-40_ internal rotation", 27,
        "IMU data/Sternum/acc_sternum.csv", 5090, 5091
    ),
]

verification_results = []

for condition, subject, relative_file, start, end in cleaned_cases:
    relative_file = Path(relative_file)

    source_path = (
        ROOT / condition / f"Subject {subject}" / relative_file
    )
    cleaned_path = (
        clean_root / condition / f"Subject {subject}" / relative_file
    )

    if not source_path.is_file() or not cleaned_path.is_file():
        raise FileNotFoundError(
            f"Original or cleaned file missing: {cleaned_path}"
        )

    original = pd.read_csv(source_path)
    cleaned = pd.read_csv(cleaned_path)

    row_count_ok = len(original) == len(cleaned)
    affected_nan = cleaned.iloc[start:end + 1].isna().all().all()

    original_array = original.to_numpy(dtype=float)
    cleaned_array = cleaned.to_numpy(dtype=float)

    unaffected_mask = np.ones(len(original), dtype=bool)
    unaffected_mask[start:end + 1] = False

    unaffected_ok = np.allclose(
        original_array[unaffected_mask],
        cleaned_array[unaffected_mask],
        equal_nan=True
    )

    passed = row_count_ok and affected_nan and unaffected_ok

    verification_results.append({
        "file": str(cleaned_path),
        "row_count_preserved": row_count_ok,
        "target_region_is_nan": affected_nan,
        "unaffected_values_unchanged": unaffected_ok,
        "passed": passed,
    })

    print(f"\nFile: {cleaned_path}")
    print(f"Row count preserved: {row_count_ok}")
    print(f"Target region entirely NaN: {affected_nan}")
    print(f"Unaffected values unchanged: {unaffected_ok}")
    print(f"Verification passed: {passed}")

pending_path = Path("imu_pending_review.csv")

if not pending_path.is_file():
    raise FileNotFoundError(pending_path)

pending_df = pd.read_csv(pending_path)
pending_count_ok = len(pending_df) == 3

print(f"\nPending-review report exists: {pending_path.is_file()}")
print(f"Pending cases recorded: {len(pending_df)}")
print(f"Expected three pending cases: {pending_count_ok}")

if not all(item["passed"] for item in verification_results):
    raise RuntimeError("At least one cleaned file failed verification.")

if not pending_count_ok:
    raise RuntimeError("Pending-review report does not contain three cases.")

print("\nStep 31 complete: cleaning coverage verified.")
print("Original dataset remains unchanged.")

print("\n" + "=" * 65)
print("STEP 32: CLEANED DATASET INVENTORY")

inventory_records = []

for condition, subject, relative_file, start, end in cleaned_cases:
    relative_file = Path(relative_file)

    source_path = (
        ROOT / condition / f"Subject {subject}" / relative_file
    )
    cleaned_path = (
        clean_root / condition / f"Subject {subject}" / relative_file
    )

    inventory_records.append({
        "condition": condition,
        "subject": subject,
        "relative_file": str(relative_file),
        "original_exists": source_path.is_file(),
        "cleaned_copy_exists": cleaned_path.is_file(),
        "cleaning_status": "cleaned_copy_verified",
        "first_affected_row": start,
        "last_affected_row": end,
    })

for case in pending_imu_cases:
    relative_file = (
        Path(case["condition"])
        / f'Subject {case["subject"]}'
        / "IMU data"
        / case["location"]
        / case["filename"]
    )

    source_path = ROOT / relative_file
    cleaned_path = clean_root / relative_file

    inventory_records.append({
        "condition": case["condition"],
        "subject": case["subject"],
        "relative_file": str(relative_file),
        "original_exists": source_path.is_file(),
        "cleaned_copy_exists": cleaned_path.is_file(),
        "cleaning_status": "pending_review_original_retained",
        "first_affected_row": case["start_row"],
        "last_affected_row": case["end_row"],
    })

inventory_df = pd.DataFrame(inventory_records)

inventory_path = Path("cleaned_dataset_inventory.csv")
inventory_df.to_csv(inventory_path, index=False)

print(f"Inventory records: {len(inventory_df)}")
print(f"Verified cleaned copies: {(inventory_df['cleaning_status'] == 'cleaned_copy_verified').sum()}")
print(f"Pending-review cases: {(inventory_df['cleaning_status'] == 'pending_review_original_retained').sum()}")
print(f"Missing original files: {(~inventory_df['original_exists']).sum()}")
print(f"Missing cleaned copies: {(~inventory_df['cleaned_copy_exists']).sum()}")
print(f"Inventory saved to: {inventory_path.resolve()}")


if not inventory_df["original_exists"].all():
    raise RuntimeError("An original file listed in the inventory is missing.")

verified_rows = inventory_df[
    inventory_df["cleaning_status"] == "cleaned_copy_verified"
]

pending_rows = inventory_df[
    inventory_df["cleaning_status"] == "pending_review_original_retained"
]

if not verified_rows["cleaned_copy_exists"].all():
    raise RuntimeError("A verified cleaned copy is missing.")

if pending_rows["cleaned_copy_exists"].any():
    print("Note: pending-review cases do not require cleaned copies.")

print("Step 32 complete: inventory verified.")
print("All original files exist.")
print("All verified cleaned copies exist.")
print("Pending-review cases remain unmodified.")

