import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_prep.feature_extractor import extract_features

LABEL_MAP = {
    "polyglot": 1,
    "benign": 0,
    "malicious_non_polyglot": 0,
}


def extract_features_safe(path: Path) -> dict:
    """Wraps extract_features with error handling so one bad file doesn't kill the batch run."""
    try:
        return extract_features(str(path))
    except Exception as e:
        return {"error": str(e)}


def build_feature_matrix(dataset_dir: str, output_csv: str = "feature_matrix.csv") -> None:
    root = Path(dataset_dir)
    rows = []
    skipped = []

    for label_folder, label_value in LABEL_MAP.items():
        folder = root / label_folder
        if not folder.is_dir():
            print(f"Warning: expected subfolder '{label_folder}' not found under {dataset_dir}, skipping")
            continue

        files = [p for p in folder.rglob("*") if p.is_file()]
        print(f"Processing {len(files)} files in {label_folder}...")

        for i, path in enumerate(files, 1):
            features = extract_features_safe(path)
            if "error" in features:
                skipped.append({"file": str(path), "error": features["error"]})
                continue

            features["label"] = label_value
            features["source_file"] = str(path)
            rows.append(features)

            if i % 500 == 0:
                print(f"  ...{i}/{len(files)}")

    if not rows:
        print("No files processed — check dataset_dir and subfolder names.")
        return

    fieldnames = [k for k in rows[0].keys()]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_csv}")
    print(f"Label distribution: {sum(1 for r in rows if r['label'] == 1)} polyglot, "
          f"{sum(1 for r in rows if r['label'] == 0)} non-polyglot")

    if skipped:
        print(f"\n{len(skipped)} files skipped due to errors:")
        for s in skipped[:10]:
            print(f"  {s['file']}: {s['error']}")
        if len(skipped) > 10:
            print(f"  ...and {len(skipped) - 10} more")


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "feature_matrix.csv"
    build_feature_matrix(dataset_dir, output_csv)
# Usage once feature_matrix.csv is written (feeds directly into model training):
#
#   df = pd.read_csv("feature_matrix.csv")
#   y = df["label"]
#   X = df.drop(columns=["label", "source_file"])
