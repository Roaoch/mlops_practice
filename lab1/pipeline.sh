python3 -m venv .venv
.venv/bin/pip install "torch>=2.10.0" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install "datasets>=4.6.1" "gensim>=4.4.0" "scikit-learn>=1.8.0" "triton>=3.6.0"

.venv/bin/python "data_creation.py" -q
.venv/bin/python "data_preprocessing.py" -q
.venv/bin/python "model_preparation.py" -q

score=$(.venv/bin/python "model_testing.py")

echo "F1 Score: $score"