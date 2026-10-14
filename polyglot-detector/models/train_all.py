# train_all.py
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.registry import MODEL_REGISTRY


def load_tabular(feature_csv: str):
    df = pd.read_csv(feature_csv)
    y = df["label"]
    X = df.drop(columns=[c for c in ["label", "source_file"] if c in df.columns])
    return X, y


def load_raw_bytes(manifest_csv: str, max_len: int):
    """manifest_csv: columns 'source_file', 'label' — reuse feature_matrix.csv's source_file/label columns."""
    df = pd.read_csv(manifest_csv)
    X = []
    for path in df["source_file"]:
        with open(path, "rb") as f:
            data = f.read(max_len)
        padded = list(data) + [0] * (max_len - len(data))
        X.append(padded[:max_len])
    return X, df["label"].tolist()


def train_one(model_name: str, feature_csv: str, max_len: int, out_dir: str) -> dict:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Registered: {list(MODEL_REGISTRY)}")

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls()

    if model.input_kind == "tabular":
        X, y = load_tabular(feature_csv)
    else:
        X, y = load_raw_bytes(feature_csv, max_len)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\n=== Training {model_name} ({model.input_kind}) ===")
    model.fit(X_train, y_train, X_test, y_test)

    probs = model.predict_proba(X_test)
    preds = [1 if p > 0.5 else 0 for p in probs]

    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))
    print(f"AUC: {auc:.4f}")

    out_path = Path(out_dir) / model_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))

    return {"model": model_name, "auc": auc, "report": report, "saved_to": str(out_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                         help="Model names from the registry to train, e.g. --models lightgbm torch_cnn")
    parser.add_argument("--feature-csv", default="feature_matrix.csv")
    parser.add_argument("--max-len", type=int, default=65536, help="Byte-truncation length for raw-byte models")
    parser.add_argument("--out-dir", default="trained_models")
    args = parser.parse_args()

    results = []
    for model_name in args.models:
        try:
            results.append(train_one(model_name, args.feature_csv, args.max_len, args.out_dir))
        except Exception as e:
            print(f"!! {model_name} failed: {e}")
            results.append({"model": model_name, "error": str(e)})

    summary_path = Path(args.out_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSummary written to {summary_path}")
    for r in results:
        if "auc" in r:
            print(f"  {r['model']}: AUC {r['auc']:.4f}")
        else:
            print(f"  {r['model']}: FAILED ({r['error']})")


if __name__ == "__main__":
    main()