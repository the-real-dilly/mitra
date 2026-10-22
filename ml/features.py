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



# --------------------------------------------------------------------------
# Whole-file streaming scan (files larger than the head/tail windows).
# 3-5 byte signatures hit by chance in compressed/random data (~0.2 expected
# hits per GiB per signature), so every candidate hit in the middle of a file is
# confirmed with a cheap structural check on the bytes that follow it.
# --------------------------------------------------------------------------
def _u16(b, o): return b[o] | (b[o + 1] << 8)
def _u32be(b, o): return int.from_bytes(b[o:o + 4], "big")

_VERIFY = {
	"zip": lambda b, p: _u16(b, p + 4) <= 63 and 1 <= _u16(b, p + 26) <= 512 and b[p + 30] >= 32,
	"zip_cd": lambda b, p: _u16(b, p + 4) <= 99 and 1 <= _u16(b, p + 28) <= 512 and b[p + 46] >= 32,
	"zip_eocd": lambda b, p: _u16(b, p + 4) <= 1 and _u16(b, p + 20) < 4096,
	"pdf": lambda b, p: re.match(rb"[12]\.\d", b[p + 5:p + 8]) is not None,
	"gif": lambda b, p: _u16(b, p + 6) > 0 and _u16(b, p + 8) > 0,
	"gif87": lambda b, p: _u16(b, p + 6) > 0 and _u16(b, p + 8) > 0,
	"jpg": lambda b, p: b[p + 6:p + 10] in (b"JFIF", b"Exif"),
	"7z": lambda b, p: b[p + 6] == 0,
	"gzip": lambda b, p: b[p + 3] & 0xE0 == 0 and b[p + 8] in (0, 2, 4) and b[p + 9] <= 13,
	"bzip2": lambda b, p: 0x31 <= b[p + 3] <= 0x39 and b[p + 4:p + 10] == b"1AY&SY",
	"elf": lambda b, p: b[p + 4] in (1, 2) and b[p + 5] in (1, 2) and b[p + 6] == 1,
	"pe": lambda b, p: _u16(b, p + 4) in (0x14c, 0x8664, 0x1c0, 0xaa64),
	"riff": lambda b, p: re.fullmatch(rb"[A-Za-z0-9 ]{4}", b[p + 8:p + 12]) is not None,
	"ogg": lambda b, p: b[p + 4] == 0 and b[p + 5] < 8,
	"flac": lambda b, p: b[p + 4] & 0x7F == 0 and b[p + 5:p + 8] == b"\x00\x00\x22",
	"id3": lambda b, p: 2 <= b[p + 3] <= 4 and b[p + 4] == 0 and all(c < 0x80 for c in b[p + 6:p + 10]),
	"ebml": lambda b, p: b[p + 4] & 0x80 != 0,
	"mp4": lambda b, p: p >= 4 and 12 <= _u32be(b, p - 4) <= 256,
	"psd": lambda b, p: b[p + 4:p + 6] == b"\x00\x01",
	"tiff_le": lambda b, p: 8 <= int.from_bytes(b[p + 4:p + 8], "little") < (1 << 20),
	"tiff_be": lambda b, p: 8 <= _u32be(b, p + 4) < (1 << 20),
	"cab": lambda b, p: b[p + 4:p + 8] == b"\0\0\0\0",
	"woff": lambda b, p: b[p + 4:p + 8] in (b"\x00\x01\x00\x00", b"OTTO", b"true"),
	"ps": lambda b, p: b[p + 4:p + 5] == b"-",
	"rtf": lambda b, p: b[p + 5:p + 6] == b"1",
	"dicom": lambda b, p: b[p + 4:p + 6] == b"\x02\x00",
	"tar": lambda b, p: b[p + 5:p + 6] in (b"\0", b" ", b"0"),
	"iso": lambda b, p: b[p + 5] == 1,
	"java": lambda b, p: b[p + 4:p + 6] == b"\0\0" and 45 <= b[p + 7] <= 75,
	"wasm": lambda b, p: b[p + 4:p + 8] == b"\x01\0\0\0",
	"pcap": lambda b, p: b[p + 4:p + 6] == b"\x02\x00",
	"flv": lambda b, p: b[p + 4] in (1, 4, 5) and b[p + 5:p + 9] == b"\0\0\0\x09",
}
# kinds scanned in the middle of big files: verified kinds + long literals that need no check
_MID_KINDS = [k for k in EMBEDDED if k in _VERIFY or k in ("png", "png_iend", "pdf_eof", "rar", "xz", "ar", "bplist")]
_SCAN_CHUNK = 16 * 1024 * 1024
_SCAN_PAD = 80
_KIND_CAP = 20


def _confirm(kind, buf, pos):
	"""structural check of a signature hit; unknown kinds / truncated tails are accepted"""
	v = _VERIFY.get(kind)
	if v is None:
		return True
	try:
		return bool(v(buf, pos))
	except IndexError:
		return True


def _scan_region(args):
	"""Scan file bytes [lo, hi) -> {kind: (count, first_offset)} of verified hits."""
	path, lo, hi = args
	out = {}
	with open(path, "rb") as fh:
		pos = lo
		while pos < hi:
			start = max(pos - 8, 0)                      # context before (mp4 box size)
			fh.seek(start)
			buf = fh.read(min(pos + _SCAN_CHUNK, hi) - start + _SCAN_PAD)
			if not buf:
				break
			for k in _MID_KINDS:
				sig = EMBEDDED[k]
				c, first = out.get(k, (0, -1))
				if c >= _KIND_CAP:
					continue
				i = buf.find(sig)
				while i >= 0:
					abs_ = start + i
					if pos <= abs_ < hi:
						try:
							ok = _VERIFY[k](buf, i) if k in _VERIFY else True
						except IndexError:
							ok = False
						if ok:
							c += 1
							first = abs_ if first < 0 else first
							if c >= _KIND_CAP:
								break
					i = buf.find(sig, i + 1)
				out[k] = (c, first)
			pos += _SCAN_CHUNK
	return out


def scan_middle(path, lo, hi, workers=None):
	"""Parallel scan of [lo, hi); regions are split across processes."""
	import concurrent.futures as cf
	workers = workers or int(os.environ.get("POLYGLOT_SCAN_WORKERS", os.cpu_count() or 1))
	span = hi - lo
	if span <= 0:
		return {}
	parts = max(1, min(workers * 2, span // _SCAN_CHUNK or 1))
	step = -(-span // parts)
	jobs = [(path, lo + i * step, min(lo + (i + 1) * step, hi)) for i in range(parts)]
	merged = {}
	if parts == 1 or workers == 1:
		results = map(_scan_region, jobs)
	else:
		ex = cf.ProcessPoolExecutor(max_workers=workers)
		results = ex.map(_scan_region, jobs)
	for r in results:
		for k, (c, first) in r.items():
			c0, f0 = merged.get(k, (0, -1))
			merged[k] = (c0 + c, f0 if f0 >= 0 else first)
	if parts > 1 and workers > 1:
		ex.shutdown()
	return merged


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


def extract_file(path, deep=True):
	"""path -> features, in O(1) memory for any file size.

	Head/tail windows + seeks give the structural features. With deep=True (default)
	files larger than the windows are additionally streamed end to end (in parallel)
	so embedded payloads in the middle of the file are counted too.
	"""
	return extract(Src(path=path), deep=deep)


def extract(src, deep=True):
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
	# segments: (bytes, absolute offset of its first byte)
	segs = [(src.head, 0)] if src.whole else [(src.head, 0), (src.tail, n - len(src.tail))]
	mid = {}
	if deep and not src.whole and src.path:
		mid = scan_middle(src.path, len(src.head), n - len(src.tail))
	n_embedded_kinds = 0
	first_offs = []
	total_emb = 0
	for k, rx in _EMB_RE.items():
		cnt, first = 0, -1
		for buf, base in segs:
			for m in rx.finditer(buf):
				a = base + m.start()
				if a > 0 and _confirm(k, buf, m.start()):
					cnt += 1
					first = a if first < 0 else first
		if k in mid:
			c, fo = mid[k]
			cnt += c
			first = fo if first < 0 else first
		f["e_" + k] = float(min(cnt, 20))
		if cnt and k not in ("zip_cd", "zip_eocd", "pdf_eof", "png_iend", "script",
							 "gzip", "bzip2", "pe", "riff", "id3", "html", "svg", "xml"):
			n_embedded_kinds += 1
			first_offs.append(first)
			total_emb += cnt
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
