# cpu_infer.py
import os
import numpy as np

# Set before importing lightgbm/xgboost/torch/onnxruntime — controls BLAS/OpenMP thread pools.
# Match to physical cores, not hyperthreads, to avoid oversubscription.
NUM_CORES = os.cpu_count() // 2 or 1
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)

import onnxruntime as ort


def export_torch_to_onnx(torch_model, sample_input_len: int, out_path: str):
    """One-time conversion: run after training, not at serve time."""
    import torch
    torch_model.model.eval()
    dummy = torch.randint(0, 256, (1, sample_input_len), dtype=torch.long)
    torch.onnx.export(
        torch_model.model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {out_path}")


class ONNXCPUModel:
    """Drop-in replacement for a CNN adapter at serve time — same predict_proba shape, CPU-tuned session."""

    def __init__(self, onnx_path: str, num_threads: int = NUM_CORES):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_proba(self, X) -> list:
        X_arr = np.asarray(X, dtype=np.int64)
        outputs = self.session.run(None, {self.input_name: X_arr})
        return outputs[0].flatten().tolist()


def batch_score_tabular(model, feature_rows: list) -> list:
    """LightGBM/XGBoost already vectorize internally — one batched call beats N single-row calls."""
    X_batch = np.array([list(row.values()) for row in feature_rows])
    return model.predict_proba(X_batch)


def batch_score_raw_bytes(onnx_model: ONNXCPUModel, file_paths: list, max_len: int = 65536) -> list:
    batch = []
    for path in file_paths:
        with open(path, "rb") as f:
            data = f.read(max_len)
        padded = list(data) + [0] * (max_len - len(data))
        batch.append(padded[:max_len])
    return onnx_model.predict_proba(batch)