# Data Model Architecture — WSD4FEDSRM

**Phase:** 4 — Software Architecture Design (no implementation)
**Status:** Design only. No loader, preprocessing, or model code is included here.
**Scope:** Object-oriented representation of the raw WSD4FEDSRM dataset that will back all downstream phases (preprocessing, feature engineering, modeling, explainability, personalization, agents, dashboard, API).

---

## 1. Grounding: What the Real Dataset Actually Looks Like

This design is derived from inspecting the real extracted files under `datasets/raw/WSD4FEDSRM/`, not just the folder names. Three findings materially shaped the class design:

### 1.1 Two distinct levels of granularity

Not all data lives at the same level. Flattening everything into "per subject" would be wrong:

| Granularity | Data | Source files |
|---|---|---|
| **Subject-level** (once per participant) | Demographics, anthropometrics, KSS sleepiness (pre/post), MVIC calibration force | `demographic.csv`, `body_composition.csv`, `breadth.csv`, `circumference.csv`, `depth.csv`, `length.csv`, `KSS_data.csv`, `MVIC_force_data.csv` |
| **Condition-level** (once per subject **per trial**) | EMG, IMU, PPG time series, Borg RPE trajectory | `EMG, IMU, and PPG data/<condition>/Subject N/...`, `Borg data/borg_data.csv` |

A subject performs **10 conditions**: 6 fatigue trials (3 intensity bands × 2 directions) + 4 MVIC calibration trials (2 directions × 2 repetitions). Borg RPE is only collected for the 6 fatigue trials (verified: `borg_data.csv` has exactly 204 rows = 34 subjects × 6 tasks, no MVIC rows) — so `Labels` is **optional** at the Condition level.

### 1.2 Condition identity is directly decodable, no cross-referencing needed

`description.txt` confirms the Borg `task_order` codes are **fixed labels**, not permuted per subject:

```
task1_35i → IR, 30–40% MVIC     task4_35e → ER, 30–40% MVIC
task2_45i → IR, 40–50% MVIC     task5_45e → ER, 40–50% MVIC
task3_55i → IR, 50–60% MVIC     task6_55e → ER, 50–60% MVIC
```

The `group` column (1/2/3) in `demographic.csv` only records the **chronological execution order** used to counterbalance fatigue carry-over across subjects — it does not change what a task code means. This means folder names (`"30-40_ external rotation"`) and Borg task codes (`task4_35e`) both resolve to the same `(direction, intensity_band)` key, so `Condition` can be identified consistently from either source.

### 1.3 Raw sensor files carry no timestamps

EMG/IMU/PPG CSVs contain only raw sample values (e.g. `EMG(mV)`, `acc_x(m/s^2),acc_y(m/s^2),acc_z(m/s^2)`) — no time column. Time must be derived from row index ÷ sampling rate. This means every sensor class needs an explicit `sampling_rate_hz`, since it cannot be inferred from the data itself.

**Confirmed sampling rates:** EMG 1000 Hz, IMU 100 Hz, PPG 200 Hz (from `description.txt`, matching the published paper).

---

## 2. Design Principles

1. **One class per real-world concept**, not per file. `IMUData` represents one sensor *location* (fusing its 3 physical files: acc/gyr/mag — they are the same rigid sensor, sampled in lockstep, and should never be handled separately).
2. **Composition over inheritance**, except where a genuine is-a relationship exists (`EMGData`, `IMUData`, `PPGData` all *are* `SensorData`).
3. **No magic strings.** Directions, intensity bands, muscles, and IMU locations are closed sets known in advance — they become `Enum`s, not free-text.
4. **Immutability by default.** Raw data objects, once loaded from disk, should not be silently mutated by downstream code (preprocessing/feature engineering produce *new* objects rather than editing these in place).
5. **The model says nothing about how it was loaded.** These classes hold data; a separate `DatasetLoader` (already scaffolded in `ml/preprocessing/dataset_loader.py`, Phase 5+) is responsible for populating them. This keeps the model reusable even if the on-disk format changes later.

---

## 3. Supporting Enums (needed for type-safe fields below)

These are small, closed-vocabulary types referenced by the main classes. They are part of the same package but are not one of the 10 requested classes — listed here for completeness since the main classes depend on them.

```
Direction        { INTERNAL_ROTATION, EXTERNAL_ROTATION }
IntensityBand    { BAND_30_40, BAND_40_50, BAND_50_60 }              # fatigue trials only
ConditionType    { FATIGUE_TASK, MVIC_CALIBRATION }
EMGMuscle        { ANTERIOR_DELTOID, POSTERIOR_DELTOID, INFRASPINATUS,
                    UPPER_TRAPEZIUS, LATISSIMUS_DORSI, PECTORALIS_MAJOR }
IMULocation      { HAND, FOREARM, UPPER_ARM, SHOULDER, STERNUM, PELVIS }
Sex              { MALE, FEMALE }
```

---

## 4. Class Specifications

### 4.1 `SensorData` (abstract base)

**Purpose:** Common contract for any raw wearable time-series recorded during one condition trial for one subject. Captures shared provenance/metadata so every sensor reading is traceable back to its exact source file and experimental context.

**Attributes:**
| Field | Type | Notes |
|---|---|---|
| `subject_id` | `str` | Back-reference, e.g. `"P01"` |
| `condition_id` | `str` | Back-reference, e.g. `"P01_ER_30-40"` |
| `sampling_rate_hz` | `float` | Fixed per modality (EMG=1000, IMU=100, PPG=200) |
| `source_file` | `Path` | Exact CSV this was loaded from, for auditability |

**Computed properties (not stored):** `n_samples`, `duration_seconds = n_samples / sampling_rate_hz`.

**Relationships:** Abstract base of `EMGData`, `IMUData`, `PPGData`. Never instantiated directly. A `SensorData` instance is **owned by** exactly one `Condition` (composition — it cannot exist meaningfully outside that context).

---

### 4.2 `EMGData(SensorData)`

**Purpose:** One muscle channel's surface EMG recording for one subject-condition pair.

**Attributes (in addition to base):**
| Field | Type | Notes |
|---|---|---|
| `muscle` | `EMGMuscle` | Which of the 6 muscles |
| `signal` | `pd.DataFrame` / `np.ndarray` | Raw millivolt samples, single column |
| `unit` | `str` | Fixed `"mV"` |

**Relationships:** A `Condition` holds up to 6 `EMGData` instances, keyed by `EMGMuscle` (`Dict[EMGMuscle, EMGData]`) — present in **all 10** condition types (EMG is recorded during MVIC too).

---

### 4.3 `IMUData(SensorData)`

**Purpose:** One IMU location's full 9-axis motion recording (accelerometer + gyroscope + magnetometer) for one subject-condition pair. Represented as **one object per location**, not three, because the three streams are physically the same sensor sampled in lockstep — splitting them would let them drift out of sync in downstream code.

**Attributes (in addition to base):**
| Field | Type | Notes |
|---|---|---|
| `location` | `IMULocation` | Which of the 6 body locations |
| `accelerometer` | `pd.DataFrame` | columns: `acc_x, acc_y, acc_z` (m/s²) |
| `gyroscope` | `pd.DataFrame` | columns: `gyr_x, gyr_y, gyr_z` (rad/s) |
| `magnetometer` | `pd.DataFrame` | columns: `mag_x, mag_y, mag_z` (a.u.) |

**Design note:** `IMUData` intentionally does **not** reuse a single generic `data` field from the base class — it composes three named sub-frames instead, since a location legitimately carries three distinct physical quantities. This is a deliberate, documented deviation from the base class's single-stream assumption, valid because `IMUData` still honors the shared contract (`sampling_rate_hz`, `source_file`, `n_samples`, `duration_seconds` — computed from any one of the three, since they are sample-aligned).

**Relationships:** A `Condition` holds up to 6 `IMUData` instances, keyed by `IMULocation` (`Dict[IMULocation, IMUData]`) — present in all 10 condition types.

---

### 4.4 `PPGData(SensorData)`

**Purpose:** Photoplethysmogram waveform from the non-dominant index finger — **only recorded during the 6 fatigue-task conditions**, not during MVIC calibration.

**Attributes (in addition to base):**
| Field | Type | Notes |
|---|---|---|
| `signal` | `pd.DataFrame` / `np.ndarray` | Raw volt samples, single column |
| `unit` | `str` | Fixed `"Volt"` |
| `sensor_site` | `str` | Fixed `"non_dominant_index_finger"` |

**Relationships:** A `Condition` holds **zero or one** `PPGData` (`Optional[PPGData]`) — `None` for the 4 MVIC conditions.

---

### 4.5 `Labels`

**Purpose:** Ground-truth fatigue outcome for one fatigue-task `Condition` — the Borg RPE trajectory collected at 10-second intervals until volitional exhaustion. This is the primary training target for fatigue estimation models.

**Attributes:**
| Field | Type | Notes |
|---|---|---|
| `task_order_code` | `str` | Raw source code, e.g. `"task4_35e"` — kept for traceability back to `borg_data.csv` |
| `rpe_before` | `int` | Borg 6–20 scale, baseline |
| `rpe_timeseries` | `Dict[int, int]` | `{seconds_elapsed: rpe_value}`, sparse (trial can end before 250s) |
| `rpe_at_end_of_trial` | `int` | Terminal RPE (protocol targets 20 = exhaustion) |
| `trial_duration_seconds` | `float` | `length_of_trial_(sec)` from source |

**Relationships:** One-to-one composition with `Condition`, but **only for `FATIGUE_TASK` conditions** — `Condition.labels` is `Optional[Labels]`, always `None` for `MVIC_CALIBRATION` conditions (confirmed: no MVIC rows exist in `borg_data.csv`).

> **KSS sleepiness** (before/after) is *not* part of `Labels` — it is measured once per subject for the whole session, not per condition. It is modeled as two fields directly on `Subject` (see 4.8), since forcing it into a condition-scoped class would misrepresent its granularity.

---

### 4.6 `DemographicData`

**Purpose:** Static, subject-level personal facts, sourced from `demographic.csv`.

**Attributes:**
| Field | Type | Notes |
|---|---|---|
| `age` | `int` | |
| `height_cm` | `float` | |
| `sex` | `Sex` | |
| `dominant_hand` | `str` (`"left"`/`"right"`) | |
| `execution_order_group` | `int` | 1/2/3 — counterbalancing group (see §1.2) |
| `exercise_type` | `str` | Free-text self-report |
| `exercise_frequency_per_week` | `int` | |
| `assessment_date` | `date` | |

**Relationships:** One-to-one composition with `Subject`.

---

### 4.7 `AnthropometricData`

**Purpose:** Physical body measurements, sourced from `body_composition.csv`, `breadth.csv`, `circumference.csv`, `depth.csv`, `length.csv`.

**Attributes:**
| Field | Type | Notes |
|---|---|---|
| `mass_kg` | `float` | from `body_composition.csv` |
| `body_fat_percent` | `float` | |
| `muscle_percent` | `float` | |
| `visceral_fat` | `float` | |
| `bmi` | `float` | |
| `breadth_cm` | `Dict[str, float]` | e.g. `{"upper_arm_mid": 7.36, "forearm_distal": 5.01, ...}` |
| `circumference_cm` | `Dict[str, float]` | same segment-name convention |
| `depth_cm` | `Dict[str, float]` | same segment-name convention |
| `length_cm` | `Dict[str, float]` | `{"upper_arm": 37, "forearm": 29, "hand": 17, "palmar": 9}` |

**Design note:** The four geometric measurement groups (breadth/circumference/depth/length) use a small `Dict[str, float]` rather than ~7 individually named fields each — they are homogeneous, repeat the same limb-segment naming pattern across files, and are consumed as a *set of anthropometric features* downstream rather than individually addressed. Body-composition scalars, by contrast, are conceptually distinct quantities and get named fields.

**Relationships:** One-to-one composition with `Subject`.

---

### 4.8 `Subject`

**Purpose:** One participant. Aggregates their fixed profile (demographics, anthropometrics), session-level outcomes (KSS, MVIC calibration force), and every `Condition` trial they performed.

**Attributes:**
| Field | Type | Notes |
|---|---|---|
| `subject_id` | `str` | Canonical, e.g. `"P01"` (normalizes raw `"subject_1"`) |
| `demographic` | `DemographicData` | |
| `anthropometric` | `AnthropometricData` | |
| `kss_before` | `int` | Session-level, from `KSS_data.csv` |
| `kss_after` | `int` | |
| `mvic_force` | `MVICForceProfile` | Supporting value object (see below) — IR/ER calibration force from `MVIC_force_data.csv` |
| `conditions` | `Dict[str, Condition]` | Keyed by `condition_id`, exactly 10 entries when complete |

**Supporting value object — `MVICForceProfile`** (not one of the 10 required classes, but necessary — these are subject-level calibration *values*, distinct from the raw MVIC sensor recordings which live inside `Condition`):
```
MVICForceProfile:
    ir_trial_1_n, ir_trial_2_n, ir_mean_n: float
    er_trial_1_n, er_trial_2_n, er_mean_n: float
```

**Relationships:** One `Subject` → many `Condition` (composition, 1:10). One `Subject` → one `DemographicData`, one `AnthropometricData`, one `MVICForceProfile` (composition, 1:1). Many `Subject` → one `Dataset` (aggregation).

---

### 4.9 `Condition`

**Purpose:** One experimental trial — the unit that bundles synchronized multimodal sensor recordings and (if applicable) fatigue labels for one subject performing one task.

**Attributes:**
| Field | Type | Notes |
|---|---|---|
| `condition_id` | `str` | Canonical, e.g. `"P01_ER_30-40"` or `"P01_MVIC_IR_1"` |
| `subject_id` | `str` | Back-reference |
| `condition_type` | `ConditionType` | `FATIGUE_TASK` or `MVIC_CALIBRATION` |
| `direction` | `Direction` | Internal/external rotation |
| `intensity_band` | `Optional[IntensityBand]` | Set for `FATIGUE_TASK`, `None` for MVIC |
| `trial_number` | `Optional[int]` | 1 or 2, set for `MVIC_CALIBRATION` only |
| `task_order_code` | `Optional[str]` | Raw Borg code (`"task4_35e"`), fatigue trials only |
| `emg` | `Dict[EMGMuscle, EMGData]` | Always populated (6 entries) |
| `imu` | `Dict[IMULocation, IMUData]` | Always populated (6 entries) |
| `ppg` | `Optional[PPGData]` | Populated only for `FATIGUE_TASK` |
| `labels` | `Optional[Labels]` | Populated only for `FATIGUE_TASK` |

**Relationships:** Owned by exactly one `Subject` (composition). Owns its `EMGData`/`IMUData`/`PPGData`/`Labels` instances (composition — none of these are shared across conditions).

---

### 4.10 `Dataset`

**Purpose:** Top-level entry point representing the entire WSD4FEDSRM corpus. The root of the object graph and the single object downstream modules (preprocessing, feature engineering, model training) will consume.

**Attributes:**
| Field | Type | Notes |
|---|---|---|
| `root_dir` | `Path` | `datasets/raw/WSD4FEDSRM/` |
| `subjects` | `Dict[str, Subject]` | Keyed by `subject_id`, up to 34 entries |
| `dataset_name` | `str` | Fixed `"WSD4FEDSRM"` |
| `license` | `str` | Fixed `"CC BY 4.0"` |
| `doi` | `str` | Fixed `"10.5281/zenodo.8415066"` |

**Relationships:** One `Dataset` → many `Subject` (composition — this is the aggregate root; nothing in the model exists independently of a `Dataset` instance in practice, even though `Subject`/`Condition` objects are technically constructible standalone for testing).

---

## 5. ASCII UML Class Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                                 Dataset                                │
│──────────────────────────────────────────────────────────────────────│
│ root_dir: Path                                                         │
│ dataset_name: str                                                      │
│ license: str                                                           │
│ doi: str                                                                │
│ subjects: Dict[str, Subject]                                           │
└───────────────────────────────────┬──────────────────────────────────┘
                                     │ 1 ── * (composition)
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                                Subject                                │
│────────────────────────────────────────────────────────────────────│
│ subject_id: str                                                       │
│ demographic: DemographicData          ◇──1──1  DemographicData        │
│ anthropometric: AnthropometricData    ◇──1──1  AnthropometricData     │
│ mvic_force: MVICForceProfile          ◇──1──1  MVICForceProfile       │
│ kss_before: int                                                       │
│ kss_after: int                                                        │
│ conditions: Dict[str, Condition]                                      │
└───────────────────────────────────┬──────────────────────────────────┘
                                     │ 1 ── * (composition, ≤10)
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                               Condition                               │
│────────────────────────────────────────────────────────────────────│
│ condition_id: str            subject_id: str                          │
│ condition_type: ConditionType     direction: Direction                │
│ intensity_band: Optional[IntensityBand]                                │
│ trial_number: Optional[int]       task_order_code: Optional[str]      │
│ emg: Dict[EMGMuscle, EMGData]          ◇──0..6── EMGData               │
│ imu: Dict[IMULocation, IMUData]        ◇──0..6── IMUData               │
│ ppg: Optional[PPGData]                 ◇──0..1── PPGData               │
│ labels: Optional[Labels]               ◇──0..1── Labels                │
└──────────────────────────────────────────────────────────────────────┘
        │                    │                    │                │
        ▼                    ▼                    ▼                ▼
┌───────────────┐  ┌──────────────────┐  ┌───────────────┐  ┌─────────────┐
│  «abstract»   │  │      Labels      │  │  (see below)  │  │(see below)  │
│  SensorData   │  │──────────────────│  └───────────────┘  └─────────────┘
│───────────────│  │ task_order_code  │
│ subject_id    │  │ rpe_before       │
│ condition_id  │  │ rpe_timeseries   │
│ sampling_rate │  │ rpe_at_end       │
│ source_file   │  │ trial_duration_s │
│───────────────│  └──────────────────┘
│ +n_samples    │
│ +duration_s   │
└───────┬───────┘
        △  (inheritance)
        │
   ┌────┼─────────────┬──────────────────┐
   │                   │                  │
┌──────────────┐ ┌─────────────────┐ ┌──────────────┐
│   EMGData    │ │     IMUData     │ │   PPGData    │
│──────────────│ │─────────────────│ │──────────────│
│ muscle       │ │ location        │ │ signal       │
│ signal       │ │ accelerometer   │ │ unit         │
│ unit         │ │ gyroscope       │ │ sensor_site  │
└──────────────┘ │ magnetometer    │ └──────────────┘
                  └─────────────────┘

┌───────────────────┐        ┌────────────────────────┐
│  DemographicData   │        │   AnthropometricData   │
│───────────────────│        │────────────────────────│
│ age                │        │ mass_kg                │
│ height_cm          │        │ body_fat_percent       │
│ sex                │        │ muscle_percent         │
│ dominant_hand      │        │ visceral_fat           │
│ execution_order_grp│        │ bmi                    │
│ exercise_type      │        │ breadth_cm: Dict       │
│ exercise_freq_wk   │        │ circumference_cm: Dict │
│ assessment_date    │        │ depth_cm: Dict         │
└───────────────────┘        │ length_cm: Dict        │
                              └────────────────────────┘

Legend:  ◇── composition (owner's lifetime bounds the part)
         △── inheritance (is-a)
         1──*  one-to-many     1──1  one-to-one     0..1 / 0..6  optional / bounded multiplicity
```

---

## 6. Recommended Folder Structure

```
ml/
    data_models/
        __init__.py
        enums.py            # Direction, IntensityBand, ConditionType, EMGMuscle, IMULocation, Sex
        sensors.py          # SensorData (base), EMGData, IMUData, PPGData
        labels.py           # Labels
        demographics.py     # DemographicData
        anthropometry.py    # AnthropometricData
        subject.py          # Subject, MVICForceProfile
        condition.py        # Condition
        dataset.py           # Dataset
```

This mirrors the requested example structure, with two small additions justified above: `enums.py` (shared closed-vocabulary types every other module imports) and `MVICForceProfile` co-located in `subject.py` since it is only ever an attribute of `Subject`, not a standalone concept.

**Import direction is strictly one-way** (`dataset → subject → condition → {sensors, labels}`, and `subject → {demographics, anthropometry}`), with `enums.py` imported by everything and depending on nothing — this keeps the model acyclic and each file unit-testable in isolation.

---

## 7. Why This Architecture Beats Nested Dictionaries

The previous phase's loader returns a plain `dict` (`{"participant_id": ..., "emg": DataFrame, ...}`). That was the right call *for a loading layer*, but the wrong call as the shape the rest of the codebase (models, SHAP, agents, dashboard, API) programs against, for concrete reasons visible in this exact dataset:

| Problem with nested dicts | Concrete example from WSD4FEDSRM | How the class model fixes it |
|---|---|---|
| **No compile-time/IDE safety** | `data["emg"]["Anterior_Deltoid"]` vs. actual key `"anterior_deltoid"` fails silently at runtime, deep in a training loop | `EMGMuscle.ANTERIOR_DELTOID` is a real enum member — a typo is a `NameError` at the call site, not a `KeyError` three functions later |
| **Ambiguous optionality** | Is `data["ppg"]` missing because this is an MVIC trial (expected) or because loading failed (bug)? A dict can't say | `PPGData: Optional[...]` is declared once on `Condition`; `None` is the documented, intentional state for MVIC trials |
| **No single source of truth for structure** | Every consumer (preprocessing, SHAP, dashboard) re-derives "what's inside an EMG record" by re-reading loader code or guessing from a sample | The dataclass **is** the schema — one definition, everywhere imported |
| **Silent shape drift** | If a future dataset version renames a column, dict-based code fails deep inside a pipeline with a confusing `KeyError` | A dataclass field rename is a one-line, greppable change; type checkers (`mypy`) catch every call site that needs updating |
| **Can't attach behavior** | Computing `duration_seconds` for every sensor reading means copy-pasting `len(df) / fs` everywhere | `SensorData.duration_seconds` is a single property, inherited by all three sensor types |
| **Hard to test in isolation** | Testing "does my feature extractor handle a subject with missing PPG" requires constructing a specific nested dict by hand and hoping it matches what the loader would produce | `Condition(ppg=None, ...)` is a first-class, discoverable constructor argument |
| **Poor discoverability** | A new contributor has to run the loader and `print()` a sample to learn the shape | `Subject`, `Condition`, etc. show up in autocomplete, `help()`, and generated docs immediately |

In short: dicts are fine as a **wire format** (which is exactly how `dataset_loader.py` currently uses them — reading raw CSVs into DataFrames). They are the wrong choice as the **domain model** that every downstream phase (feature engineering, fusion models, SHAP, agents, API responses) will import and build on for the rest of the project's life.

---

## 8. Dataclasses vs. Pydantic — Decision

**Recommendation: Python `dataclasses` (standard library), not Pydantic.**

| Consideration | Dataclasses | Pydantic | Verdict for this project |
|---|---|---|---|
| **Runtime validation of external input** | None built-in | Strong (this is Pydantic's core purpose) | Irrelevant here — data enters the system through `dataset_loader.py`, which already reads/validates raw CSVs. These objects are constructed *after* loading, from already-known-good, program-internal data, not from untrusted external JSON/HTTP payloads. |
| **Performance in tight ML loops** | No overhead beyond a normal object | Validates on every instantiation (real cost across thousands of `EMGData`/`IMUData` objects — this dataset alone has 8,364 sensor files) | Dataclasses win — this is an offline research pipeline that will construct large numbers of sensor objects in batch, not a handful of API request bodies. |
| **Dependency footprint** | Zero (stdlib) | Adds a dependency (already a mature, common one — but still one more thing pinned/upgraded/vendored in a research repo) | Dataclasses keep the core data model dependency-free. |
| **Holding `pd.DataFrame`/`np.ndarray` fields** | Trivial — just a typed field | Requires `arbitrary_types_allowed=True` and loses most of Pydantic's validation value anyway, since Pydantic doesn't meaningfully validate DataFrame *contents* | Dataclasses are the natural fit; Pydantic's main selling point degrades to nothing once most fields are DataFrames. |
| **Where Pydantic *would* earn its keep** | — | Validating config, and validating request/response bodies at the FastAPI boundary | This is exactly why the **FastAPI layer** (Section 9) should use Pydantic **at its edges** — not a reason to use it for the internal data model. |

**Conclusion:** Use `dataclasses` for `Dataset`/`Subject`/`Condition`/`SensorData`-family/`Labels`/`DemographicData`/`AnthropometricData` — the internal, trusted, DataFrame-heavy domain model. Reserve Pydantic for the future FastAPI request/response schemas, where its actual strength (validating untrusted external input at a system boundary) applies. This also matches the project's own boundary-validation principle: validate at the edges, trust internal code.

---

## 9. Extensibility — How Future Modules Build on This Model

| Future module | How it consumes this model |
|---|---|
| **Preprocessing** (`ml/preprocessing/`) | Takes a `Dataset`/`Subject`/`Condition`, returns *new* cleaned objects (or a parallel `ProcessedCondition` type) — never mutates raw instances in place, preserving raw data as the immutable ground truth. |
| **Feature Engineering** (`ml/feature_engineering/`) | Operates per-`Condition`, reading `.emg`, `.imu`, `.ppg` uniformly via the shared `SensorData` contract (`duration_seconds`, `sampling_rate_hz`) regardless of modality. |
| **Baseline Models** (`ml/models/classical/`, `ml/models/deep_learning/`) | Trains per-modality using the enum-keyed dicts (`Dict[EMGMuscle, EMGData]`) directly as feature groups — no re-parsing of file paths. |
| **Multimodal Attention Fusion** (`ml/models/proposed/`) | Fuses across `condition.emg`, `condition.imu`, `condition.ppg` — the object model makes "which modalities are present for this trial" (`ppg is None` for MVIC) an explicit, checkable condition rather than an inferred one. |
| **SHAP Explainability** (`ml/explainability/`) | Attributions map cleanly back to named, typed features (`EMGMuscle.INFRASPINATUS`, `IMULocation.SHOULDER`) instead of opaque dict keys or array indices — directly usable as human-readable SHAP feature labels. |
| **Personalization / LOSO** (`ml/personalization/`) | Iterates `Dataset.subjects`, holding out one `Subject` at a time — the aggregate root makes leave-one-subject-out splitting a one-line operation over a typed collection. |
| **AI Agents** (`backend/agents/`) | Consume `Subject`/`Condition`/`Labels` objects to reason over and explain a specific subject's fatigue trajectory, with typed access to demographics/anthropometrics for personalized narrative generation. |
| **Dashboard** (`dashboard/`) | Renders per-subject, per-condition views directly from typed objects — no dict-key guessing in UI code. |
| **FastAPI APIs** (`backend/api/`) | Internal dataclasses are converted to Pydantic response models **at the API boundary only** (thin adapter layer), giving automatic OpenAPI schema generation and request validation without coupling the core model to a web framework. |

---

## 10. Summary

- The dataset has two granularities (subject vs. condition) and a directly decodable condition-identity scheme — both are now explicit in the model rather than implicit in folder-naming conventions.
- Ten required classes are fully specified with attributes and relationships, plus a small set of justified supporting enums/value objects (`Direction`, `IntensityBand`, `ConditionType`, `EMGMuscle`, `IMULocation`, `Sex`, `MVICForceProfile`).
- `dataclasses` (stdlib) is the right tool for this internal, DataFrame-heavy, trusted-input model; Pydantic is deferred to the future FastAPI boundary only.
- The design is acyclic, one-way-importable, and every future module (preprocessing through dashboard/API) has a defined, typed consumption path.

**Next step (not part of this phase):** Phase 5 implements these dataclasses in `ml/data_models/` and updates `dataset_loader.py` to populate them from the raw CSVs.
