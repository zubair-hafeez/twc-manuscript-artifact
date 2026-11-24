# Exact Deletion Audits for Temporal Walk Predictors with Structural Certificates

This repository contains the two result summaries used to build six of the paper's
tables and the Python script that turns them into LaTeX.

From the repository root, run:

```sh
python3 generate_tables.py
```

The script checks the SHA-256 hashes in `manifest.json` and writes
six tables and the headline macros under `manuscript/`. It uses Python's
standard library and reads saved measurements.

The summaries record the frozen confirmation and exploratory evaluations.
The script regenerates the tables from those summaries. Rerunning the experiments requires the original experiment code and datasets.
