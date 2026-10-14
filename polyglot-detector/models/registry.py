# registry.py
import pickle
from abc import ABC, abstractmethod
from pathlib import Path


class PolyglotModel(ABC):
    """Common interface every model adapter must implement."""

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        ...

    @abstractmethod
    def predict_proba(self, X) -> list:
        """Return probability of class 1 (polyglot) per row."""
        ...

    @abstractmethod
    def save(self, path: str):
        ...

    @abstractmethod
    def load(self, path: str):
        ...


class LightGBMModel(PolyglotModel):
    input_kind = "tabular"  # expects the feature_matrix.csv columns

    def __init__(self, params: dict = None):
        import lightgbm as lgb
        self.lgb = lgb
        self.params = params or {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.05, "num_leaves": 31, "verbose": -1,
        }
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        train_set = self.lgb.Dataset(X_train, label=y_train)
        valid_sets = []
        if X_val is not None:
            valid_sets = [self.lgb.Dataset(X_val, label=y_val, reference=train_set)]
        self.model = self.lgb.train(
            self.params, train_set, num_boost_round=500,
            valid_sets=valid_sets or None,
            callbacks=[self.lgb.early_stopping(30)] if valid_sets else None,
        )

    def predict_proba(self, X) -> list:
        return list(self.model.predict(X))

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model = self.lgb.Booster(model_file=path)


class XGBoostModel(PolyglotModel):
    input_kind = "tabular"

    def __init__(self, params: dict = None):
        import xgboost as xgb
        self.xgb = xgb
        self.params = params or {"objective": "binary:logistic", "eval_metric": "auc"}
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        dtrain = self.xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]
        if X_val is not None:
            dval = self.xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))
        self.model = self.xgb.train(
            self.params, dtrain, num_boost_round=500, evals=evals,
            early_stopping_rounds=30 if X_val is not None else None,
            verbose_eval=False,
        )

    def predict_proba(self, X) -> list:
        return list(self.model.predict(self.xgb.DMatrix(X)))

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model = self.xgb.Booster()
        self.model.load_model(path)


class TorchByteCNNModel(PolyglotModel):
    """CNN over raw byte sequences, PyTorch. Expects X as a list/array of byte tensors, not tabular features."""
    input_kind = "raw_bytes"

    def __init__(self, max_len: int = 65536, device: str = None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Embedding(256, 16),
            nn.Conv1d(16, 32, kernel_size=9, stride=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        # Note: Embedding before Conv1d needs a permute in a real forward pass;
        # wrap in a custom nn.Module for production use rather than pure Sequential.

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs: int = 10, lr: float = 1e-3):
        import torch.nn as nn
        opt = self.torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        X_t = self.torch.tensor(X_train, dtype=self.torch.long).to(self.device)
        y_t = self.torch.tensor(y_train, dtype=self.torch.float32).to(self.device)
        for epoch in range(epochs):
            opt.zero_grad()
            out = self.model(X_t).squeeze(-1)
            loss = loss_fn(out, y_t)
            loss.backward()
            opt.step()
            print(f"epoch {epoch+1}/{epochs} loss={loss.item():.4f}")

    def predict_proba(self, X) -> list:
        X_t = self.torch.tensor(X, dtype=self.torch.long).to(self.device)
        with self.torch.no_grad():
            return self.model(X_t).squeeze(-1).cpu().tolist()

    def save(self, path: str):
        self.torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(self.torch.load(path, map_location=self.device))


class KerasByteCNNModel(PolyglotModel):
    """Same idea as TorchByteCNNModel, TensorFlow/Keras version."""
    input_kind = "raw_bytes"

    def __init__(self, max_len: int = 65536):
        import tensorflow as tf
        self.tf = tf
        self.max_len = max_len
        self.model = self._build_model()

    def _build_model(self):
        tf = self.tf
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(256, 16, input_length=self.max_len),
            tf.keras.layers.Conv1D(32, 9, strides=2, activation="relu"),
            tf.keras.layers.Conv1D(64, 9, strides=2, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs: int = 10):
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
        val_data = (X_val, y_val) if X_val is not None else None
        self.model.fit(X_train, y_train, validation_data=val_data, epochs=epochs, verbose=1)

    def predict_proba(self, X) -> list:
        return self.model.predict(X).flatten().tolist()

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = self.tf.keras.models.load_model(path)


# The registry: add an entry here whenever you build a new model adapter.
MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "torch_cnn": TorchByteCNNModel,
    "keras_cnn": KerasByteCNNModel,
}