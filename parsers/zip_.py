#!/usr/bin/env python3

import binascii
import struct
from parsers import FType


def _dos_datetime(dt):
	year, month, day, hour, minute, second = dt
	dosdate = ((year - 1980) << 9) | (month << 5) | day
	dostime = (hour << 11) | (minute << 5) | (second // 2)
	return dostime, dosdate


class parser(FType):
	DESC = "Zip"
	TYPE = "Zip"
	MAGIC = b"PK\3\4"

	EOCD = b"PK\x05\x06"
	ZIP64_EOCD = b"PK\x06\x06"
	ZIP64_LOCATOR = b"PK\x06\x07"

	def __init__(self, data=""):
		FType.__init__(self, data)
		self.data = data

		self.bParasite = True
		# the parasite is stored as the very first bytes of the file
		# (local header + raw data), so it's always at offset 0
		self.parasite_o = 0
		# 32-bit stored/uncompressed size fields; 0xFFFFFFFF is reserved
		# as the ZIP64 escape value, so the real cap is one less
		self.parasite_s = 0xFFFFFFFE

		self.start_o = 4*1024*1024 # no actual downward limit

# Doesn't apply the usual way
#		self.cut = 0
#		self.prewrap = 0x25


	def identify(self):
		if not self.data.startswith(self.MAGIC):
			return False
		# a well-formed zip has an End Of Central Directory record,
		# within the last 64KB + 22 bytes (max comment length)
		return self.data.rfind(self.EOCD) != -1


	def parasitize(self, fparasite):
		# Strategy: store the parasite as a new, raw-STORED first entry,
		# prepended before the host's own data, then patch the central
		# directory in place rather than decompressing/recompressing
		# every existing member. This preserves the host's original
		# compression, flags, extra fields, timestamps and comment
		# byte-for-byte, and works even on members this tool can't
		# itself decompress (encrypted, exotic methods...).

		original = self.data
		parasite_data = fparasite.data

		if len(parasite_data) > self.parasite_s:
			return None, []

		eocd_o = original.rfind(self.EOCD)
		if eocd_o == -1 or eocd_o + 22 > len(original):
			return None, []

		# ZIP64 isn't handled by this raw-copy path. Its locator sits
		# right before the EOCD (20 bytes); searching the whole file
		# would false-positive on compressed data.
		if original[eocd_o - 20:eocd_o - 16] == self.ZIP64_LOCATOR:
			return None, []

		# refuse rather than silently wrap a counter/offset that is full
		entries = struct.unpack_from("<H", original, eocd_o + 10)[0]
		cd_size = struct.unpack_from("<I", original, eocd_o + 12)[0]
		cd_offset = struct.unpack_from("<I", original, eocd_o + 16)[0]
		if entries >= 0xFFFF \
			or cd_size + 46 >= 0xFFFFFFFF \
			or cd_offset + len(parasite_data) + 30 + 46 >= 0xFFFFFFFF:
			return None, []

		crc = binascii.crc32(parasite_data) & 0xffffffff
		size = len(parasite_data)
		dostime, dosdate = _dos_datetime((1981, 2, 3, 4, 5, 6)) # invariant for easier testing

		local_header = struct.pack("<4sHHHHHIIIHH",
			b"PK\x03\x04",   # local file header signature
			20,              # version needed to extract
			0,               # flags
			0,               # compression method: stored
			dostime, dosdate,
			crc,
			size, size,      # compressed size, uncompressed size
			0, 0)            # filename length, extra length (no filename)

		parasite_entry = local_header + parasite_data
		delta = len(parasite_entry)

		new_data = bytearray(parasite_entry)
		new_data += original

		# everything that was in the original file moved by `delta`
		eocd_o += delta

		cd_record = struct.pack("<4sHHHHHHIIIHHHHHII",
			b"PK\x01\x02",   # central directory file header signature
			20, 20,          # version made by, version needed to extract
			0,               # flags
			0,               # compression method: stored
			dostime, dosdate,
			crc,
			size, size,      # compressed size, uncompressed size
			0, 0, 0,         # filename, extra, comment lengths
			0,               # disk number start
			0,               # internal attributes
			0,               # external attributes
			0)               # relative offset of local header: first bytes of the file

		new_data = new_data[:eocd_o] + cd_record + new_data[eocd_o:]
		eocd_o += len(cd_record)

		def patch16(o, inc):
			v = (struct.unpack_from("<H", new_data, o)[0] + inc) & 0xffff
			struct.pack_into("<H", new_data, o, v)

		def patch32(o, inc):
			v = (struct.unpack_from("<I", new_data, o)[0] + inc) & 0xffffffff
			struct.pack_into("<I", new_data, o, v)

		# EOCD layout: sig:4 disk:2 cddisk:2 entries_here:2 entries_total:2
		#              cd_size:4 cd_offset:4 comment_len:2
		patch16(eocd_o + 8, 1)                  # entries on this disk
		patch16(eocd_o + 10, 1)                 # total entries
		patch32(eocd_o + 12, len(cd_record))    # central directory size
		patch32(eocd_o + 16, delta)             # central directory offset

		# the parasite's bytes sit right after its local header
		swaps = [len(local_header), len(local_header) + size]
		return bytes(new_data), swaps
