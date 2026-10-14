import csv
import itertools
import subprocess
from pathlib import Path

MITRA_PATH = "mitra.py"              # or full path, e.g. "/opt/mitra/mitra.py"
TRUEPOLYGLOT_PATH = "truepolyglot"   # or full path to the built binary/script


def load_compatibility_report(csv_path: str) -> list:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def run_mitra(file_a: str, file_b: str, out_dir: str, timeout: int = 60) -> dict:
    """Invoke Mitra on a pair of files. Mitra writes its own output filename(s) into out_dir."""
    try:
        result = subprocess.run(
            ["python3", MITRA_PATH, "-o", out_dir, file_a, file_b],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "pair": (file_a, file_b),
            "tool": "mitra",
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"pair": (file_a, file_b), "tool": "mitra", "success": False, "stderr": "timeout"}
    except FileNotFoundError:
        return {"pair": (file_a, file_b), "tool": "mitra", "success": False, "stderr": "mitra.py not found on PATH"}


def run_truepolyglot(pdf_path: str, zip_path: str, out_path: str, timeout: int = 60) -> dict:
    """Invoke TruePolyglot on a PDF+ZIP pair. Adjust the arg order/flags to match your build."""
    try:
        result = subprocess.run(
            [TRUEPOLYGLOT_PATH, pdf_path, zip_path, out_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "pair": (pdf_path, zip_path),
            "tool": "truepolyglot",
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"pair": (pdf_path, zip_path), "tool": "truepolyglot", "success": False, "stderr": "timeout"}
    except FileNotFoundError:
        return {"pair": (pdf_path, zip_path), "tool": "truepolyglot", "success": False, "stderr": "truepolyglot not found on PATH"}


def generate_from_report(report_csv: str, out_dir: str, max_pairs: int = 500) -> list:
    rows = load_compatibility_report(report_csv)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log = []

    mitra_candidates = [r["file"] for r in rows if r.get("mitra_compatible") == "True"]
    pdf_candidates = [r["file"] for r in rows if r.get("format") == "PDF" and r.get("truepolyglot_compatible") == "True"]
    zip_candidates = [r["file"] for r in rows if r.get("format") == "ZIP" and r.get("truepolyglot_compatible") == "True"]

    # Mitra: pair up distinct-format candidates, capped at max_pairs
    mitra_pairs = list(itertools.combinations(mitra_candidates, 2))[:max_pairs]
    print(f"Running Mitra on {len(mitra_pairs)} pairs...")
    for file_a, file_b in mitra_pairs:
        log.append(run_mitra(file_a, file_b, out_dir))

    # TruePolyglot: every PDF x every ZIP, capped at max_pairs
    tp_pairs = list(itertools.product(pdf_candidates, zip_candidates))[:max_pairs]
    print(f"Running TruePolyglot on {len(tp_pairs)} pairs...")
    for i, (pdf_path, zip_path) in enumerate(tp_pairs):
        out_path = str(Path(out_dir) / f"tp_{i:04d}.pdf")
        log.append(run_truepolyglot(pdf_path, zip_path, out_path))

    successes = sum(1 for r in log if r["success"])
    print(f"\n{successes}/{len(log)} generations succeeded")
    return log


if __name__ == "__main__":
    import sys

    report_csv = sys.argv[1] if len(sys.argv) > 1 else "compatibility_report.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "generated_polyglots"

    log = generate_from_report(report_csv, out_dir)

    failures = [r for r in log if not r["success"]]
    if failures:
        print(f"\n{len(failures)} failures, first few:")
        for f in failures[:10]:
            print(f"  {f['tool']} on {f['pair']}: {f['stderr']}")