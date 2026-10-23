# XGBoost polyglot detector

Flags files that are valid in more than one format (the kind `mitra.py` generates).

```
python3 detect.py [--threshold 0.5] FILE...        # uses polyglot_xgb.json
python3 build_dataset.py data/polyglots out.npz    # features for polyglots + negatives
python3 train.py out.npz model.json                # grouped CV report + final model
```

- `features.py` - format-agnostic features: signatures at canonical offsets, signatures embedded
  past offset 0 (each hit structurally verified), and "logical end vs physical end" checks
  (PNG/RIFF/ZIP/PDF/JPEG/GIF). File size and byte statistics are deliberately excluded:
  the polyglots are small, null-heavy PoCs, so those features let the model separate
  "Mitra test file" from "real-world file" instead of detecting polyglots.
- Any file size works: head/tail 2 MB windows plus seeks, and files over 4 MB are streamed
  end to end in parallel (about 10 s per GiB on 4 cores; `extract_file(path, deep=False)` skips it).
- Data: `data/` (989 Mitra polyglots, 55 clean inputs, ~6.8k ordinary files as features only).

## Evaluation (5-fold CV; polyglots held out by format pair, negatives split per file)

ROC-AUC 0.986, PR-AUC 0.959; at threshold 0.5 precision 0.95, recall 0.90, false-positive rate 0.66%.

## Limitations

- Trained only on Mitra output. Real-world polyglots with other layouts are untested.
- Weak spots: ARJ pairs (ELF/EXE/FLAC/GIF/TIFF/WAV) and DICOM/ELF are missed entirely.
- Most remaining false positives are the clean Mitra inputs themselves (tiny minimal files).
- The large-file scan is not covered by training data; its accuracy on big files is unmeasured.
- The model file is trained on all of the data, so scoring the training polyglots overstates accuracy; use the CV numbers.
