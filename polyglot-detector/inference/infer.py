# infer.py
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.registry import MODEL_REGISTRY
from data_prep.feature_extractor import extract_features


def load_model(model_name: str, model_dir: str):
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls()
    model.load(str(Path(model_dir) / model_name))
    return model


def featurize_for_model(path: str, model, max_len: int = 65536):
    if model.input_kind == "tabular":
        feats = extract_features(path)
        feats.pop("label", None)
        feats.pop("source_file", None)
        return [list(feats.values())]  # single row, same column order as training
    else:
        with open(path, "rb") as f:
            data = f.read(max_len)
        padded = list(data) + [0] * (max_len - len(data))
        return [padded[:max_len]]


def score_file(path: str, model_names: list, model_dir: str, max_len: int = 65536) -> dict:
    scores = {}
    for name in model_names:
        model = load_model(name, model_dir)
        X = featurize_for_model(path, model, max_len)
        prob = model.predict_proba(X)[0]
        scores[name] = float(prob)

    scores["ensemble_mean"] = sum(scores.values()) / len(scores)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="File(s) to score")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--model-dir", default="trained_models")
    parser.add_argument("--max-len", type=int, default=65536)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    results = {}
    for file_path in args.files:
        scores = score_file(file_path, args.models, args.model_dir, args.max_len)
        scores["flagged"] = scores["ensemble_mean"] > args.threshold
        results[file_path] = scores
        flag = "POLYGLOT" if scores["flagged"] else "clean"
        print(f"{file_path}: {flag} (ensemble={scores['ensemble_mean']:.3f})")
        for name in args.models:
            print(f"    {name}: {scores[name]:.3f}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()