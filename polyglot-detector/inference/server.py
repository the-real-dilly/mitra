# server.py
import sys
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.registry import MODEL_REGISTRY
from inference.infer import featurize_for_model  # reused from the inference script above

MODEL_DIR = "trained_models"
MODEL_NAMES = list(MODEL_REGISTRY.keys())  # or a fixed subset you want served
MAX_LEN = 65536

loaded_models = {}  # populated once at startup, kept for the life of the process


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    for name in MODEL_NAMES:
        model_cls = MODEL_REGISTRY[name]
        model = model_cls()
        model.load(str(Path(MODEL_DIR) / name))
        loaded_models[name] = model
        print(f"  loaded {name}")
    print("All models ready.")
    yield
    loaded_models.clear()


app = FastAPI(lifespan=lifespan)


class ScoreResponse(BaseModel):
    scores: dict
    ensemble_mean: float
    flagged: bool


def score_path(path: str, threshold: float = 0.5) -> ScoreResponse:
    scores = {}
    for name, model in loaded_models.items():
        X = featurize_for_model(path, model, MAX_LEN)
        scores[name] = float(model.predict_proba(X)[0])

    ensemble_mean = sum(scores.values()) / len(scores)
    return ScoreResponse(scores=scores, ensemble_mean=ensemble_mean, flagged=ensemble_mean > threshold)


@app.post("/scan", response_model=ScoreResponse)
async def scan_upload(file: UploadFile = File(...), threshold: float = 0.5):
    """Scan an uploaded file directly — no need to already have it on the server's disk."""
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        return score_path(tmp.name, threshold)


class PathRequest(BaseModel):
    path: str
    threshold: float = 0.5


@app.post("/scan-path", response_model=ScoreResponse)
async def scan_path(req: PathRequest):
    """Scan a file already present on the server's filesystem, by path."""
    return score_path(req.path, req.threshold)


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(loaded_models.keys())}