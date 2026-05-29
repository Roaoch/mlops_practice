#!/bin/bash

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python "data_creation.py" -q
.venv/bin/python "data_preprocessing.py" -q
.venv/bin/python "model_preparation.py" -q

score=$(.venv/bin/python "model_testing.py")

echo "F1 Score: $score"