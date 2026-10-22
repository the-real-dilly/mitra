"""Score files with the trained polyglot detector.

usage: detect.py [--model model.json] [--threshold 0.5] FILE [FILE ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "polyglot_xgb.json"))
	ap.add_argument("--threshold", type=float, default=0.5)
	ap.add_argument("files", nargs="+")
	a = ap.parse_args()

	m = xgb.XGBClassifier()
	m.load_model(a.model)
	names = json.load(open(a.model.replace(".json", ".features.json")))
	for p in a.files:
		with open(p, "rb") as fh:
			f = features.extract(fh.read())
		x = np.array([[f.get(k, 0.0) for k in names]], dtype=np.float32)
		s = float(m.predict_proba(x)[0, 1])
		print("%.4f  %s  %s" % (s, "POLYGLOT" if s >= a.threshold else "clean", p))


if __name__ == "__main__":
	main()
