---
chunk_id: DATA-AUDIT-01
title: Known data quality problems in the recorded sessions
category: protocol
source: Audit of the study recordings by this project (ml/dataset_pipeline, Module 1 and 3)
date: 2026-10-24
licence: own summary, no external text
tags: [data quality, audit, dropout, corrupt recording, duration mismatch, missing channel]
---
# Known data quality problems in the recorded sessions

Every recording was checked for missing files, corrupt values, stuck sensors and timing mismatches before any model used it. The three sensor streams agree on duration to within a hundredth of a second in all 204 exercises, and no file is empty. A few recordings still have problems, and the software flags them instead of hiding them.

- **One participant has two muscle channels with corrupt values** in one exercise (45% internal rotation). Those two channels are marked unreliable for that exercise, and the other channels are used as normal.
- **One accelerometer spike** in one trunk sensor is far above anything physical. The affected samples are flagged and replaced by interpolation.
- **Recorded exercise length disagrees with the sensors** in three exercises. In one of them the written length is about 40 seconds longer than the sensors, probably a typing error.
- **One participant has movement-sensor files of different lengths** (about 11 seconds apart) in one exercise. All streams are cut to the shortest one.
- **Mains hum was not found** in any muscle recording, so no notch filter is applied unless a peak is measured.

When a channel is more than 20% unreliable in a 10-second window, the features from that channel are left out for that window and the dashboard shows a data-quality flag. An estimate built while a flag is showing is less certain than usual.
