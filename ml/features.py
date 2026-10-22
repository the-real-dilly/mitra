"""Format-agnostic feature extraction for polyglot detection.

Every feature is computed from the raw bytes only. File size is deliberately
NOT a feature: the generated polyglots are small, so size would be a shortcut.
"""
import math
import os
import re
import struct

import numpy as np

# name -> (offset, magic): signatures anchored at a canonical offset
ANCHORED = {
	"jpg": (0, b"\xff\xd8\xff"),
	"png": (0, b"\x89PNG\r\n\x1a\n"),
	"gif": (0, b"GIF8"),
	"bmp": (0, b"BM"),
	"zip": (0, b"PK\x03\x04"),
	"zip_empty": (0, b"PK\x05\x06"),
	"7z": (0, b"7z\xbc\xaf\x27\x1c"),
	"rar": (0, b"Rar!\x1a\x07"),
	"gzip": (0, b"\x1f\x8b\x08"),
	"bzip2": (0, b"BZh"),
	"xz": (0, b"\xfd7zXZ\x00"),
	"zstd": (0, b"\x28\xb5\x2f\xfd"),
	"lz4": (0, b"\x04\x22\x4d\x18"),
	"pdf": (0, b"%PDF-"),
	"ps": (0, b"%!PS"),
	"elf": (0, b"\x7fELF"),
	"pe": (0, b"MZ"),
	"java": (0, b"\xca\xfe\xba\xbe"),
	"wasm": (0, b"\x00asm"),
	"riff": (0, b"RIFF"),
	"rifx": (0, b"RIFX"),
	"ogg": (0, b"OggS"),
	"flac": (0, b"fLaC"),
	"id3": (0, b"ID3"),
	"flv": (0, b"FLV\x01"),
	"ebml": (0, b"\x1a\x45\xdf\xa3"),
	"tiff_le": (0, b"II*\x00"),
	"tiff_be": (0, b"MM\x00*"),
	"psd": (0, b"8BPS"),
	"ico": (0, b"\x00\x00\x01\x00"),
	"cab": (0, b"MSCF"),
	"ar": (0, b"!<arch>\n"),
	"cpio": (0, b"070707"),
	"cpio_new": (0, b"070701"),
	"pcap": (0, b"\xd4\xc3\xb2\xa1"),
	"pcap_be": (0, b"\xa1\xb2\xc3\xd4"),
	"pcapng": (0, b"\x0a\x0d\x0d\x0a"),
	"woff": (0, b"wOFF"),
	"woff2": (0, b"wOF2"),
	"otf": (0, b"OTTO"),
	"ttf": (0, b"\x00\x01\x00\x00\x00"),
	"rtf": (0, b"{\\rtf"),
	"svg": (0, b"<svg"),
	"xml": (0, b"<?xml"),
	"html": (0, b"<html"),
	"php": (0, b"<?php"),
	"icc": (36, b"acsp"),
	"mp4": (4, b"ftyp"),
	"jp2": (4, b"jP  "),
	"arj": (0, b"\x60\xea"),
	"dicom": (128, b"DICM"),
	"tar": (257, b"ustar"),
	"iso": (0x8001, b"CD001"),
	"wad": (0, b"IWAD"),
	"nes": (0, b"NES\x1a"),
	"ilda": (0, b"ILDA"),
	"bpg": (0, b"BPG\xfb"),
	"bplist": (0, b"bplist00"),
}

# strong (>= 4 bytes) magics searched anywhere in the file, past offset 0
EMBEDDED = {
	"zip": b"PK\x03\x04",
	"zip_cd": b"PK\x01\x02",
	"zip_eocd": b"PK\x05\x06",
	"pdf": b"%PDF-",
	"pdf_eof": b"%%EOF",
	"png": b"\x89PNG\r\n\x1a\n",
	"png_iend": b"IEND\xaeB`\x82",
	"gif": b"GIF89a",
	"gif87": b"GIF87a",
	"jpg": b"\xff\xd8\xff",
	"7z": b"7z\xbc\xaf\x27\x1c",
	"rar": b"Rar!\x1a\x07",
	"gzip": b"\x1f\x8b\x08",
	"bzip2": b"BZh",
	"xz": b"\xfd7zXZ\x00",
	"elf": b"\x7fELF",
	"pe": b"PE\x00\x00",
	"riff": b"RIFF",
	"ogg": b"OggS",
	"flac": b"fLaC",
	"id3": b"ID3",
	"ebml": b"\x1a\x45\xdf\xa3",
	"mp4": b"ftyp",
	"psd": b"8BPS",
	"tiff_le": b"II*\x00",
	"tiff_be": b"MM\x00*",
	"cab": b"MSCF",
	"ar": b"!<arch>\n",
	"woff": b"wOFF",
	"ps": b"%!PS",
	"rtf": b"{\\rtf",
	"dicom": b"DICM",
	"tar": b"ustar",
	"iso": b"CD001",
	"java": b"\xca\xfe\xba\xbe",
	"wasm": b"\x00asm",
	"html": b"<html",
	"svg": b"<svg",
	"xml": b"<?xml",
	"script": b"<script",
	"php": b"<?php",
	"pcap": b"\xd4\xc3\xb2\xa1",
	"flv": b"FLV\x01",
	"bplist": b"bplist00",
}
_EMB_RE = {k: re.compile(re.escape(v)) for k, v in EMBEDDED.items()}


def entropy(b):
	if len(b) == 0:
		return 0.0
	c = np.bincount(np.frombuffer(b, np.uint8), minlength=256)
	p = c[c > 0] / len(b)
	return float(-(p * np.log2(p)).sum())


class Src:
	"""Random-access view of a file. Only `head`/`tail` windows are held in memory,
	so a multi-GB file costs a few MB. Files <= 2*WINDOW are held whole."""
	WINDOW = 2 * 1024 * 1024

	def __init__(self, path=None, data=None):
		self.path = path
		if data is not None:
			self.n, self.head, self.tail, self.whole = len(data), data, data, True
			self._data = data
			return
		self.n = os.path.getsize(path)
		self.whole = self.n <= 2 * self.WINDOW
		if self.whole:
			self.head = self.tail = self._data = self.read(0, self.n)
		else:
			self.head = self.read(0, self.WINDOW)
			self.tail = self.read(self.n - self.WINDOW, self.WINDOW)
			self._data = None

	def read(self, off, size):
		if self._data_ready():
			return self._data[off:off + size]
		if off < 0 or off >= self.n:
			return b""
		with open(self.path, "rb") as fh:
			fh.seek(off)
			return fh.read(size)

	def _data_ready(self):
		return getattr(self, "_data", None) is not None

	def sample(self):
		"""bytes used for window-wise statistics and signature scans"""
		return self.head if self.whole else self.head + self.tail


def _png_end(src):
	"""Offset right after IEND if the chunk walk is sane, else -1."""
	if not src.head.startswith(b"\x89PNG\r\n\x1a\n"):
		return -1
	o = 8
	for _ in range(100000):
		if o + 12 > src.n:
			break
		h = src.read(o, 8)
		if len(h) < 8:
			break
		(n,) = struct.unpack(">I", h[:4])
		o += 12 + n
		if h[4:8] == b"IEND":
			return o
	return -1


def _riff_end(d):
	if d[:4] in (b"RIFF", b"RIFX") and len(d) >= 8:
		e = ">" if d[:4] == b"RIFX" else "<"
		return struct.unpack(e + "I", d[4:8])[0] + 8
	return -1


def _zip_info(src):
	"""(eocd_end, cd_offset_ok) -> trailing & offset sanity (EOCD found in the tail)."""
	t = src.tail
	i = t.rfind(b"PK\x05\x06")
	if i < 0 or i + 22 > len(t):
		return -1, -1
	cd_size, cd_off = struct.unpack("<II", t[i + 12:i + 20])
	(cl,) = struct.unpack("<H", t[i + 20:i + 22])
	end = src.n - len(t) + i + 22 + cl
	# central directory must sit exactly where the EOCD says (shifted by any prefix)
	ok = 1 if src.read(cd_off, 4) == b"PK\x01\x02" else 0
	return end, ok


def _pdf_eof_end(src):
	i = src.tail.rfind(b"%%EOF")
	return -1 if i < 0 else src.n - len(src.tail) + i + 5


FEATURE_NAMES = None


def extract_file(path):
	"""path -> features, in O(1) memory for any file size (head/tail windows + seeks)."""
	return extract(Src(path=path))


def extract(src):
	"""bytes | Src -> dict[str, float]"""
	if isinstance(src, (bytes, bytearray)):
		src = Src(data=bytes(src))
	f = {}
	n = src.n
	d = src.sample()          # what statistics / signature scans look at
	head = src.head[:64]

	# --- global statistics (no raw size) ---
	arr = np.frombuffer(d, np.uint8) if n else np.zeros(1, np.uint8)
	f["entropy"] = entropy(d)
	f["entropy_head"] = entropy(d[:1024])
	f["entropy_tail"] = entropy(src.tail[-1024:])
	win = 512
	ents = [entropy(d[i:i + win]) for i in range(0, max(n - win, 1), win)] or [0.0]
	ents = np.array(ents)
	f["ent_win_std"] = float(ents.std())
	f["ent_win_min"] = float(ents.min())
	f["ent_win_max"] = float(ents.max())
	f["ent_win_jumps"] = float((np.abs(np.diff(ents)) > 2.0).mean()) if len(ents) > 1 else 0.0
	f["frac_null"] = float((arr == 0).mean())
	f["frac_print"] = float(((arr >= 32) & (arr < 127)).mean())
	f["frac_high"] = float((arr >= 128).mean())
	f["frac_ff"] = float((arr == 255).mean())
	# longest run of zeros (cavities), relative to size
	z = (arr == 0).astype(np.int8)
	if z.any():
		dz = np.diff(np.concatenate(([0], z, [0])))
		runs = np.where(dz == -1)[0] - np.where(dz == 1)[0]
		f["max_null_run"] = float(runs.max()) / max(n, 1)
		f["n_null_runs16"] = float((runs >= 16).sum())
	else:
		f["max_null_run"] = 0.0
		f["n_null_runs16"] = 0.0
	# long-run printable text vs binary mix
	f["print_head"] = float(((arr[:256] >= 32) & (arr[:256] < 127)).mean())
	f["print_tail"] = float(((arr[-256:] >= 32) & (arr[-256:] < 127)).mean())

	# --- anchored signatures ---
	anchored_hits = []
	for k, (off, m) in ANCHORED.items():
		hit = d[off:off + len(m)] == m
		f["a_" + k] = float(hit)
		if hit and k not in ("bmp", "pe", "ico", "id3", "xml", "ttf", "zip_empty"):
			anchored_hits.append(k)
	f["n_anchored"] = float(len(anchored_hits))
	# anchored magics at non-zero offsets (dicom/tar/iso/icc/mp4) next to a magic at 0
	f["n_anchored_far"] = float(sum(1 for k in anchored_hits if ANCHORED[k][0] > 0))
	f["anchored_mix"] = float(len(anchored_hits) >= 2)

	# --- embedded signatures (past offset 0) ---
	n_embedded_kinds = 0
	first_offs = []
	total_emb = 0
	for k, rx in _EMB_RE.items():
		hits = [m.start() for m in rx.finditer(d)]
		hits_nz = [h for h in hits if h > 0]
		f["e_" + k] = float(min(len(hits_nz), 20))
		if hits_nz and k not in ("zip_cd", "zip_eocd", "pdf_eof", "png_iend", "script",
								 "gzip", "bzip2", "pe", "riff", "id3", "html", "svg", "xml"):
			n_embedded_kinds += 1
			first_offs.append(hits_nz[0])
			total_emb += len(hits_nz)
	f["n_embedded_kinds"] = float(n_embedded_kinds)
	f["embedded_total"] = float(min(total_emb, 100))
	f["embedded_first_off_log"] = math.log2(min(first_offs) + 1) if first_offs else -1.0
	f["embedded_density"] = total_emb / max(n, 1) * 1000.0

	# --- structural consistency: logical end vs physical end ---
	png = _png_end(src)
	f["png_trailing"] = float(n - png) if png >= 0 else -1.0
	f["png_trailing_log"] = math.log2(n - png + 1) if png >= 0 else -1.0
	riff = _riff_end(src.head)
	f["riff_trailing"] = float(n - riff) if riff >= 0 else -9e9
	f["riff_mismatch"] = float(n != riff) if riff >= 0 else -1.0
	zend, zok = _zip_info(src)
	f["zip_trailing"] = float(n - zend) if zend >= 0 else -1.0
	f["zip_cd_ok"] = float(zok)
	# zip with something before the first local header
	zi = src.head.find(b"PK\x03\x04")
	f["zip_prefix_log"] = math.log2(zi + 1) if zi >= 0 else -1.0
	f["zip_prefix_nz"] = float(zi > 0)
	pe = _pdf_eof_end(src)
	f["pdf_trailing"] = float(n - pe) if pe >= 0 else -1.0
	pi = src.head.find(b"%PDF-")
	f["pdf_prefix_nz"] = float(pi > 0)
	f["pdf_prefix_log"] = math.log2(pi + 1) if pi >= 0 else -1.0
	# jpeg: EOI near end?
	f["jpg_eoi_end"] = float(src.tail.rstrip(b"\0").endswith(b"\xff\xd9")) if head[:2] == b"\xff\xd8" else -1.0
	f["gif_trailer_end"] = float(src.tail.rstrip(b"\0").endswith(b";")) if head[:3] == b"GIF" else -1.0
	# id3v1 footer, 7z/rar-ish
	f["id3v1_footer"] = float(n >= 128 and src.tail[-128:-125] == b"TAG")
	# does the bulk of the file after the first 4 KB sit behind a big gap of identical bytes
	f["tail_after_zeros"] = float(bool(re.search(rb"\x00{32}[^\x00]", d)))

	return f


def vectorize(feats_list):
	global FEATURE_NAMES
	if FEATURE_NAMES is None:
		FEATURE_NAMES = list(feats_list[0].keys())
	return np.array([[f.get(k, 0.0) for k in FEATURE_NAMES] for f in feats_list], dtype=np.float32)
