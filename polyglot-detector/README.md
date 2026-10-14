# Polyglot File Detection

ML pipeline for detecting polyglot files (files simultaneously valid under two or more formats) for cybersecurity use — data prep, model training across multiple frameworks, and inference, including a CPU-optimized and server-hosted path.

All of these started as code blocks in the "Polyglot File Detection: ML Approach" doc; this repo is that code, organized to actually run.

## Layout

```
data_prep/
  feature_extractor.py          # magic-byte / trailing-data / entropy features for one file
  batch_feature_extraction.py   # walks a labeled dataset dir -> feature_matrix.csv
  compatibility_checker.py      # flags which files are good candidates for Mitra / TruePolyglot
  generate_polyglots.py         # orchestrates Mitra + TruePolyglot as subprocesses

models/
  registry.py                   # common PolyglotModel interface + MODEL_REGISTRY
  train_all.py                  # trains any/all registered models from one CLI command
  train_lightgbm_standalone.py  # original single-model LightGBM example (pre-registry)

inference/
  infer.py                      # CLI: score a file against chosen registered models
  server.py                     # FastAPI server, loads all models once at startup
  cpu_infer.py                  # CPU-optimized path: thread tuning, ONNX export, batching
```

## Suggested dataset layout

```
dataset/
├── polyglot/
├── benign/
└── malicious_non_polyglot/
```

Folder names must match `LABEL_MAP` in `batch_feature_extraction.py` exactly. Subfolders inside each label dir are fine (it recurses).

## Typical flow

1. Generate/collect raw files into `dataset/<label>/...`.
2. `python data_prep/compatibility_checker.py dataset/` to pre-screen inputs for `generate_polyglots.py`.
3. `python data_prep/generate_polyglots.py compatibility_report.csv generated_polyglots/` to produce more positives via Mitra/TruePolyglot.
4. `python data_prep/batch_feature_extraction.py dataset/ feature_matrix.csv`.
5. `python models/train_all.py --models lightgbm xgboost torch_cnn keras_cnn --feature-csv feature_matrix.csv`.
6. `python inference/infer.py some_file.pdf --models lightgbm torch_cnn` for one-off scoring, or run `inference/server.py` via `uvicorn` to keep models resident and scan over HTTP.

## Notes

- `registry.py` is the single place to add a new model type — one adapter class, one line in `MODEL_REGISTRY`; `train_all.py` and `infer.py` don't need to change.
- `cpu_infer.py` assumes you've exported trained CNNs to ONNX (`export_torch_to_onnx`) after training.
- `server.py`'s `/scan` endpoint accepts raw file uploads — put it behind your normal network/auth controls before exposing it.
