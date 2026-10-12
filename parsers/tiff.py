#!/usr/bin/env python3

from parsers import FType
from helpers import *


# Strategy:
# 1. move parasite right after File Header
# 2. adjust every IFD pointer (main chain + sub-IFDs)
# 3. adjust every tag value that is itself a file offset:
#    - out-of-line tag values (count*typesize > 4) are pointers to a data
#      block that moved by delta
#    - offset-array tags (StripOffsets/TileOffsets/FreeOffsets) also need
#      each element of the array shifted, since those elements are
#      themselves offsets into the (now shifted) pixel data
#    - IFD-pointer tags (SubIFDs/ExifIFD/GPSIFD/InteropIFD) point to other
#      IFDs, which are walked recursively
#
# Not supported: BigTIFF (different magic, 8-byte offsets) - identify()
# won't match it, so it's just rejected rather than mishandled.


BYTE =			1
ASCII =			2
SHORT =			3
LONG = 			4
RATIONAL = 	5
SBYTE =			6
UNDEFINED =	7
SSHORT =		8
SLONG =			9
SRATIONAL =	10
FLOAT =			11
DOUBLE =		12


Lengths = {
	BYTE			:1,
	ASCII			:1,
	SHORT			:2,
	LONG			:4,
	RATIONAL	:8,
	SBYTE			:1,
	UNDEFINED	:1,
	SSHORT		:2,
	SLONG			:4,
	SRATIONAL	:8,
	FLOAT			:4,
	DOUBLE		:8,
}

# tags whose value is an array of offsets pointing into the image data
OFFSET_ARRAY_TAGS = {
	273,  # StripOffsets
	288,  # FreeOffsets
	324,  # TileOffsets
}

# tags whose value is an offset to another IFD
SUBIFD_PTR_TAGS = {
	330,    # SubIFDs
	34665,  # ExifIFD
	34853,  # GPSIFD
	40965,  # InteropIFD
}

# tag whose (single) value is a raw data offset, not an IFD
JPEG_OFFSET_TAG = 513  # JPEGInterchangeFormat (old-style JPEG-in-TIFF)


class parser(FType):
	DESC = "TIFF / Tagged Image File Format"
	TYPE = "TIFF"
	MAGICb = b"MM\0\x2a"
	MAGICl = b"II\x2a\0"

	# pointer to the first IFD
	ifdptr_o = 4
	ifdptr_s = 4

	# where the first image data can start
	data_o = 8


	def __init__(self, data=""):
		FType.__init__(self, data)
		self.data = data

		self.bParasite = True
		self.parasite_o = self.data_o
		self.parasite_s = 0xFFFFFFFF

		self.cut = self.parasite_o
		self.prewrap = 0


	def identify(self):
		d = self.data

		if d.startswith(self.MAGICb):
			self.endianness = BIG
		elif d.startswith(self.MAGICl):
			self.endianness = LITTLE
		else:
			return False

		return True


	def fixparasite(self, parasite):
		# word alignment
		if len(parasite) % 2 == 1:
			parasite += b"\0"
		return parasite


	def fixformat(self, d, delta):
		e = self.endianness

		# Many real-world (and especially minimal/PoC) TIFFs have offsets
		# or array elements that don't actually resolve to valid in-file
		# locations (truncated test fixtures, placeholder/zero values...).
		# Treat every offset fix as best-effort: skip silently rather than
		# raising if the target field would fall outside the buffer.
		def safe_inc(o, size):
			if o < 0 or o + size > len(d):
				return False
			return True

		# A shifted value must still fit its field: placeholder values
		# (0xFFFFFFFF...) or 16-bit SHORT offsets overflow once a large
		# delta is added, which makes struct.pack raise. Skip those.
		def do_inc4(o):
			nonlocal d
			if safe_inc(o, 4) and get4(d, o, e) + delta <= 0xFFFFFFFF:
				d = inc4(d, o, delta, e)
				return True
			return False

		def do_inc2(o):
			nonlocal d
			if safe_inc(o, 2) and get2(d, o, e) + delta <= 0xFFFF:
				d = inc2(d, o, delta, e)
				return True
			return False

		# a worklist of field offsets whose 4-byte value is itself an
		# offset to an IFD that needs to be shifted by delta and walked.
		# starts with the pointer to the first IFD; grows with "next IFD"
		# pointers and with sub-IFD pointer tags as they're discovered.
		worklist = [self.ifdptr_o]
		seen = set()

		while worklist:
			ptr_o = worklist.pop(0)
			if ptr_o in seen:
				continue
			seen.add(ptr_o)

			if not do_inc4(ptr_o):
				continue
			ifd_o = get4(d, ptr_o, e)
			if ifd_o == 0 or ifd_o + 2 > len(d):
				continue

			o = ifd_o
			c = get2(d, o, e)  # entries count
			o += 2

			for i in range(c):
				if o + 12 > len(d):
					# truncated/bogus IFD - stop walking this table
					break

				tag = get2(d, o, e)
				type_ = get2(d, o + 2, e)
				count = get4(d, o + 4, e)
				val_o = o + 8

				typesize = Lengths.get(type_, 1)
				out_of_line = typesize * count > 4

				if out_of_line:
					# the value field is a pointer to a data block
					# that moved along with everything else
					if do_inc4(val_o):
						dataoff = get4(d, val_o, e)

						if tag in OFFSET_ARRAY_TAGS:
							if type_ == LONG:
								for j in range(count):
									do_inc4(dataoff + 4 * j)
							elif type_ == SHORT:
								for j in range(count):
									do_inc2(dataoff + 2 * j)
							# else: unsupported offset-array type - leave untouched

						elif tag in SUBIFD_PTR_TAGS and type_ == LONG:
							for j in range(count):
								worklist.append(dataoff + 4 * j)

				else:
					if tag in OFFSET_ARRAY_TAGS:
						if type_ == LONG and count == 1:
							do_inc4(val_o)
						elif type_ == SHORT and count in (1, 2):
							for j in range(count):
								do_inc2(val_o + 2 * j)

					elif tag in SUBIFD_PTR_TAGS:
						# single inline IFD offset (count == 1, LONG, fits in 4 bytes)
						worklist.append(val_o)

					elif tag == JPEG_OFFSET_TAG:
						do_inc4(val_o)

				o += 12  # tag:2 type:2 count:4 value:4

			# "next IFD" pointer field, right after the last entry
			if o + 4 <= len(d):
				worklist.append(o)

		return d
