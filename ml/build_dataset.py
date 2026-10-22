"""Build the polyglot-detection dataset.

positives: polyglots written by `mitra.py` (see ml/README.md)
negatives: the clean Mitra input files + a stratified sample of ordinary
           files found on this machine (many are legitimate containers such as
           jars/wheels/PDFs/PNGs, i.e. hard negatives).

usage: build_dataset.py POLY_DIR OUT.npz [--neg-per-ext N] [--seed S]
"""
import argparse
import collections
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(HERE, "..", "input")

MIN_SIZE, MAX_SIZE = 100, 300 * 1024
SKIP_PREFIXES = ("/proc", "/sys", "/dev", "/tmp", "/home/user/mitra", "/root/.cache", "/root/.ccr")
# text / source extensions are plentiful; keep few of each
TEXTY = {"py", "go", "h", "pyi", "rb", "yaml", "js", "txt", "json", "xml", "rst", "vim", "rbs", "md",
		 "pl", "ts", "c", "pm", "cmake", "list", "map", "s", "gemspec", "ronn", "yml", "toml", "cfg"}


def poly_group(fn):
	"""group = unordered pair of the two extensions, e.g. 'png|zip'"""
	parts = fn.split("..")
	exts = sorted(p.lower() for p in parts[-2:]) if len(parts) >= 3 else [fn.rsplit(".", 1)[-1]]
	if len(parts) == 2:  # name.hash.ext1.ext2
		exts = sorted(fn.rsplit(".", 2)[-2:])
	return "|".join(exts)


def system_negatives(per_ext, seed):
	rnd = random.Random(seed)
	by_ext = collections.defaultdict(list)
	for root, dirs, files in os.walk("/", followlinks=False):
		if root.startswith(SKIP_PREFIXES):
			dirs[:] = []
			continue
		for fn in files:
			p = os.path.join(root, fn)
			try:
				if os.path.islink(p):
					continue
				s = os.path.getsize(p)
			except OSError:
				continue
			if MIN_SIZE <= s <= MAX_SIZE:
				ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else "noext"
				if len(ext) > 8:
					ext = "noext"
				# reservoir-ish: keep up to 20x the cap, sample later
				lst = by_ext[ext]
				if len(lst) < per_ext * 20:
					lst.append(p)
				else:
					lst[rnd.randrange(len(lst))] = p
	out = []
	for ext, lst in by_ext.items():
		cap = per_ext // 3 if ext in TEXTY else per_ext
		rnd.shuffle(lst)
		out += lst[:cap]
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("poly_dir")
	ap.add_argument("out")
	ap.add_argument("--neg-per-ext", type=int, default=120)
	ap.add_argument("--seed", type=int, default=0)
	a = ap.parse_args()

	X, y, groups, names = [], [], [], []

	def add(path, label, group):
		try:
			with open(path, "rb") as fh:
				d = fh.read()
		except OSError:
			return
		if not (MIN_SIZE <= len(d) <= 5 * MAX_SIZE):
			return
		X.append(features.extract(d))
		y.append(label)
		groups.append(group)
		names.append(path)

	for fn in sorted(os.listdir(a.poly_dir)):
		add(os.path.join(a.poly_dir, fn), 1, poly_group(fn))
	npos = len(y)
	print("positives:", npos)

	for fn in sorted(os.listdir(INPUT_DIR)):
		if fn != "README.md":
			add(os.path.join(INPUT_DIR, fn), 0, "clean|" + fn.rsplit(".", 1)[-1])
	print("clean inputs:", len(y) - npos)

	for p in system_negatives(a.neg_per_ext, a.seed):
		add(p, 0, "sys|" + (p.rsplit(".", 1)[-1] if "." in os.path.basename(p) else "noext"))
	print("negatives total:", len(y) - npos)

	M = features.vectorize(X)
	np.savez_compressed(a.out, X=M, y=np.array(y), groups=np.array(groups), names=np.array(names),
						feature_names=np.array(features.FEATURE_NAMES))
	print("saved", a.out, M.shape)


if __name__ == "__main__":
	main()
