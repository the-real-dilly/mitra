"""Train + evaluate the XGBoost polyglot detector.

usage: train.py DATASET.npz OUT_MODEL.json
Evaluation is grouped: all samples of a given format pair (e.g. png|zip) land in
the same fold, so the reported scores are for format pairs the model has not seen.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

# Byte-statistics features separate "tiny Mitra test file" from "real-world file"
# (both polyglots and clean Mitra inputs are null-heavy PoCs), so they are dropped.
SHORTCUT = ["entropy", "entropy_head", "entropy_tail", "ent_win_std", "ent_win_min", "ent_win_max",
			"ent_win_jumps", "frac_null", "frac_print", "frac_high", "frac_ff", "max_null_run",
			"n_null_runs16", "print_head", "print_tail", "tail_after_zeros"]
CLEAN_WEIGHT = 15.0  # clean Mitra inputs are the only in-distribution negatives

PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
			  min_child_weight=2, reg_lambda=2.0, tree_method="hist", eval_metric="logloss", n_jobs=4)


def main(path, out):
	D = np.load(path, allow_pickle=True)
	X, y, g, names, fn = D["X"], D["y"], D["groups"], D["names"], list(D["feature_names"])
	keep = [i for i, k in enumerate(fn) if k not in SHORTCUT]
	X, fn = X[:, keep], [fn[i] for i in keep]
	w = np.where(np.array([str(n).split("/")[-2] == "input" for n in names]), CLEAN_WEIGHT, 1.0)
	spw = (y == 0).sum() / max((y == 1).sum(), 1)
	print("samples %d  pos %d  neg %d  features %d  neg/pos %.2f" % (len(y), y.sum(), (y == 0).sum(), X.shape[1], spw))

	oof = np.zeros(len(y))
	cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
	for k, (tr, te) in enumerate(cv.split(X, y, g)):
		m = xgb.XGBClassifier(scale_pos_weight=spw, random_state=k, **PARAMS).fit(X[tr], y[tr], sample_weight=w[tr])
		oof[te] = m.predict_proba(X[te])[:, 1]
	pred = oof >= 0.5
	p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary")
	print("\n== grouped 5-fold CV (unseen format pairs) ==")
	print("ROC-AUC %.4f  PR-AUC %.4f" % (roc_auc_score(y, oof), average_precision_score(y, oof)))
	print("@0.5  precision %.4f  recall %.4f  F1 %.4f" % (p, r, f))
	tn, fp, fn_, tp = confusion_matrix(y, pred).ravel()
	print("TN %d FP %d FN %d TP %d   FPR %.4f" % (tn, fp, fn_, tp, fp / (fp + tn)))

	# recall per polyglot format pair, and false positives per negative class
	rec = defaultdict(list)
	fpc = defaultdict(list)
	for gi, yi, pi in zip(g, y, pred):
		(rec if yi else fpc)[gi].append(pi)
	worst = sorted(((np.mean(v), len(v), k) for k, v in rec.items() if len(v) >= 3))[:12]
	print("\nlowest-recall polyglot pairs (min 3 samples):")
	for r_, n_, k in worst:
		print("  %-24s recall %.2f  n=%d" % (k, r_, n_))
	bad = sorted(((np.mean(v), len(v), k) for k, v in fpc.items() if np.mean(v) > 0), reverse=True)[:12]
	print("\nnegative classes with false positives:")
	for r_, n_, k in bad:
		print("  %-24s FPR %.2f  n=%d" % (k, r_, n_))

	# final model on everything
	m = xgb.XGBClassifier(scale_pos_weight=spw, random_state=0, **PARAMS).fit(X, y, sample_weight=w)
	imp = sorted(zip(m.feature_importances_, fn), reverse=True)[:15]
	print("\ntop features:")
	for v, k in imp:
		print("  %-22s %.3f" % (k, v))
	m.save_model(out)
	with open(out.replace(".json", ".features.json"), "w") as fh:
		json.dump(fn, fh)
	print("\nsaved", out)


if __name__ == "__main__":
	main(*sys.argv[1:3])
