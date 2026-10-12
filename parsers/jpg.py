#!/usr/bin/env python3

from parsers import FType
from helpers import *


class parser(FType):
	DESC = "JFIF / JPEG File Interchange Format"
	TYPE = "JPG"
	MAGIC = b"\xFF\xD8"

	# max payload per marker segment: the length field is 16-bit and
	# includes itself (2 bytes), so 0xFFFF - 2 is the largest chunk
	# a single COM/APPn segment can carry
	MAXSEG = 0xFFFF - 2

	def __init__(self, data=""):
		FType.__init__(self, data)
		self.data = data

		self.bParasite = True

		self.parasite_o = 6
		# no real cap: bigger payloads are chained across several
		# consecutive marker segments (see wrap())
		self.parasite_s = 0xFFFFFFFF

		self.cut = 2
		self.prewrap = 1+1+2 # per-segment header: FF, marker, length:2


	def identify(self):
		d = self.data
		if not d.startswith(self.MAGIC):
			return False
		# right after SOI there must be another marker (FF xx, xx != 00);
		# rules out random data that merely happens to start with FFD8
		if len(d) < 4 or d[2] != 0xFF or d[3] == 0x00:
			return False
		return True


	def getPrewrap(self, parasite_s):
		# total header overhead once `parasite_s` bytes are chained
		# across as many segments as wrap() will emit for them
		if parasite_s <= 0:
			n_segments = 1
		else:
			n_segments = -(-parasite_s // self.MAXSEG) # ceil division
		return n_segments * self.prewrap


	def wrap(self, parasite, marker=b"\xFE"):
		# chain the parasite across as many back-to-back marker segments
		# as needed - JPEG decoders simply walk from one marker segment
		# to the next, so this is transparent to them
		segments = []
		o = 0
		n = len(parasite)
		while True:
			chunk = parasite[o:o+self.MAXSEG]
			segments.append(b"".join([
				b"\xFF",
				marker,
				int2b(len(chunk)+2),
				chunk,
				]))
			o += self.MAXSEG
			if o >= n:
				break
		return b"".join(segments)
