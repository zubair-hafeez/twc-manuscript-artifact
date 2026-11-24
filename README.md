
## ETWC implementation

`evaluators.py` contains the eager, lazy, and compact evaluators.
`sparse.py` implements structural pruning and the independence certificate.
The tests compare the evaluators with physical deletion and independent
walk enumeration on small histories.

With Python 3.9–3.12, install the test dependencies and run:

```sh
python -m pip install -r requirements.txt
python -m pytest -q test_v3_exactness.py
```
