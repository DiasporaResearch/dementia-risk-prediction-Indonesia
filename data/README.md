# data/

This directory is intentionally empty in version control (see `.gitignore`).

Patient-level data collected under this protocol will **not** be committed to this repository in raw form, for ethical and privacy reasons (see `docs/PROTOCOL.md`, Section 4.9).

Expected files once data collection begins (all git-ignored):

- `raw_data.csv` exported case report form data (input to `scripts/01_data_cleaning.py`)
- `processed_data.csv`  cleaned/encoded dataset (input to `scripts/02`–`04`)

A de-identified analysis dataset will be released separately alongside the resulting publication(s), consistent with the TRIPOD data-availability commitment in `docs/TRIPOD_checklist.md`.
