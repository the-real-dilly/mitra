import math
from collections import Counter

MAGIC_BYTES = {
    b"\xFF\xD8\xFF": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"GIF87a": "GIF",
    b"GIF89a": "GIF",
    b"PK\x03\x04": "ZIP",
    b"%PDF-": "PDF",
    b"MZ": "EXE",
}

EOF_MARKERS = {
    "JPEG": b"\xFF\xD9",
    "GIF": b"\x3B",
    "PDF": b"%%EOF",
}


def find_all_magic_offsets(data: bytes) -> dict:
    """Find every known magic-byte signature and where it occurs."""
    hits = {}
    for sig, fmt in MAGIC_BYTES.items():
        offset = data.find(sig)
        if offset != -1:
            hits.setdefault(fmt, []).append(offset)
    return hits


def trailing_bytes_after_eof(data: bytes, fmt: str) -> int:
    """Bytes remaining after the expected end-of-file marker for a format."""
    marker = EOF_MARKERS.get(fmt)
    if not marker:
        return 0
    idx = data.rfind(marker)
    if idx == -1:
        return 0
    return len(data) - (idx + len(marker))


def chunk_entropy(data: bytes, chunk_size: int = 256) -> list:
    """Shannon entropy per chunk — spikes often mark embedded/compressed content."""
    entropies = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if not chunk:
            continue
        counts = Counter(chunk)
        length = len(chunk)
        entropy = -sum((c / length) * math.log2(c / length) for c in counts.values())
        entropies.append(entropy)
    return entropies


def extract_features(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()

    magic_hits = find_all_magic_offsets(data)
    entropies = chunk_entropy(data)

    features = {
        "file_size": len(data),
        "num_distinct_formats_detected": len(magic_hits),
        "has_magic_at_offset_zero": any(0 in offs for offs in magic_hits.values()),
        "has_magic_mid_file": any(
            any(o > 0 for o in offs) for offs in magic_hits.values()
        ),
        "max_trailing_bytes_after_eof": max(
            (trailing_bytes_after_eof(data, fmt) for fmt in magic_hits), default=0
        ),
        "mean_chunk_entropy": sum(entropies) / len(entropies) if entropies else 0,
        "max_chunk_entropy": max(entropies) if entropies else 0,
        "entropy_variance": (
            sum((e - sum(entropies) / len(entropies)) ** 2 for e in entropies) / len(entropies)
            if entropies else 0
        ),
    }
    return features