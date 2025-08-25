# Viral Genome Prediction (NLP + ML)

Starter scaffold to reproduce a simple version of the paper's idea: **k-mer tokenization → bag-of-words (CountVectorizer) → conventional ML classifier (e.g., KNN)** on human DNA contigs.

## Quickstart

1. Create and activate a Python 3.11 virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # on macOS/Linux
   # .\.venv\Scripts\activate  # on Windows PowerShell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Put your labeled data in `data/data.csv` with columns:
   - `sequence` (raw DNA letters A/C/G/T/N)
   - `label` (0 for normal human DNA, 1 for contig containing viral sequence)

3. (Optional) Adjust hyperparameters in `configs/config.yaml` (e.g., `k=6`).

4. Train a baseline model:
   ```bash
   python src/train.py --config configs/config.yaml
   ```

5. The trained model and vectorizer will be saved under `models/` and basic metrics in `reports/`.

> Tip: This repo is organized as a light, hackable baseline you can extend (e.g., try different k, add more classifiers, or switch to char n-grams). Read `src/train.py` to see the whole pipeline.
