import os
import zipfile
from pathlib import Path

MAGIC_SIGNATURES = {
    b"\xFF\xD8\xFF": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"GIF87a": "GIF",
    b"GIF89a": "GIF",
    b"PK\x03\x04": "ZIP",
    b"%PDF-": "PDF",
    b"MZ": "EXE",
    b"\x25\x21PS": "PS",
    b"DICM": "DICOM",  # note: offset 128, checked separately below
}

# Formats Mitra handles well as a base layer (tolerate appended/injected bytes)
MITRA_FRIENDLY_FORMATS = {"PNG", "GIF", "JPEG", "ZIP", "PDF", "DICOM", "EXE"}


def detect_format(data: bytes) -> str:
    if data[128:132] == b"DICM":
        return "DICOM"
    for sig, fmt in MAGIC_SIGNATURES.items():
        if data.startswith(sig):
            return fmt
    return "UNKNOWN"


def is_valid_zip(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def is_valid_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF-") and b"%%EOF" in data


def check_mitra_compatible(data: bytes, fmt: str) -> tuple:
    """Mitra needs a format that tolerates extra bytes (trailing, cavity, or comment fields)."""
    if fmt not in MITRA_FRIENDLY_FORMATS:
        return False, f"format {fmt} not in Mitra's well-supported list"
    if len(data) < 64:
        return False, "file too small to contain a reliable format header/footer"
    return True, "ok"


def check_truepolyglot_compatible(path: str, data: bytes, fmt: str) -> tuple:
    """TruePolyglot specifically needs valid PDF+ZIP pairs."""
    if fmt == "PDF":
        return is_valid_pdf(data), "valid PDF" if is_valid_pdf(data) else "malformed PDF (no %%EOF)"
    if fmt == "ZIP":
        ok = is_valid_zip(path)
        return ok, "valid ZIP" if ok else "corrupt or unreadable ZIP"
    return False, f"format {fmt} not usable by TruePolyglot (PDF/ZIP only)"


def scan_dataset(directory: str) -> list:
    results = []
    for path in Path(directory).rglob("*"):
        if not path.is_file():
            continue
        try:
            with open(path, "rb") as f:
                head = f.read(4096)
        except Exception as e:
            results.append({"file": str(path), "error": str(e)})
            continue

        fmt = detect_format(head)
        with open(path, "rb") as f:
            full_data = f.read() if fmt == "PDF" else head  # PDF check needs the EOF marker near the tail

        mitra_ok, mitra_reason = check_mitra_compatible(head, fmt)
        tp_ok, tp_reason = check_truepolyglot_compatible(str(path), full_data, fmt)

        results.append({
            "file": str(path),
            "format": fmt,
            "size_bytes": path.stat().st_size,
            "mitra_compatible": mitra_ok,
            "mitra_reason": mitra_reason,
            "truepolyglot_compatible": tp_ok,
            "truepolyglot_reason": tp_reason,
        })
    return results


if __name__ == "__main__":
    import sys
    import csv

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = scan_dataset(dataset_dir)

    with open("compatibility_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    mitra_count = sum(1 for r in rows if r.get("mitra_compatible"))
    tp_count = sum(1 for r in rows if r.get("truepolyglot_compatible"))
    print(f"Scanned {len(rows)} files")
    print(f"Mitra-compatible: {mitra_count}")
    print(f"TruePolyglot-compatible: {tp_count}")
    print("Full report written to compatibility_report.csv")