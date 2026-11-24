# Exact Deletion Audits for Temporal Walk Predictors with Structural Certificates

This repository contains the two result summaries used to build six of the paper's
tables and the Python script that turns them into LaTeX.

Download and extract `twc-manuscript-artifact.zip`. From the extracted folder, run:

```sh
python3 analysis/generate_tables.py
```

The script checks the SHA-256 hashes in `evidence/manifest.json` and writes
six tables and the headline macros under `manuscript/`. It uses Python's
standard library and reads saved measurements.
