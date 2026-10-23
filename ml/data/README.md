# Polyglot detector dataset (snapshot)

- `polyglots/` - 989 positives, generated with `mitra.py -r` from pairs of the files in `../../input/`
  (the generator was still running when this snapshot was taken, so only part of the 3003 pairs is covered).
  File names follow Mitra's `<layout>-<types>.<hash>.<ext1>.<ext2>` scheme.
- `dataset.npz` - feature matrix `X` (`features.py`), labels `y`, CV `groups`, `names`, `feature_names`.
  Negatives: 55 clean Mitra inputs (`input/...`) and ~6.8k ordinary files sampled from the build machine
  (`sys/<ext>/<id>`; features only - the files themselves are not included).

Rebuild with `python3 build_dataset.py <polyglot dir> out.npz` and train with `python3 train.py out.npz model.json`.
