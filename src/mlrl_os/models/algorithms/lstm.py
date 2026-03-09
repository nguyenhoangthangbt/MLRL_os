"""LSTM sequence model algorithm wrapper for ML/RL OS.

Uses PyTorch LSTM under the hood. PyTorch is lazy-imported inside train()
to avoid import errors when the package is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlrl_os.models.algorithms.protocol import TrainedModel

logger = logging.getLogger(__name__)


class _LSTMModelBundle:
    """Container for all artifacts needed to run inference with a trained LSTM.

    Stored inside TrainedModel.model so that predict() and predict_proba()
    have access to the network, scaler, label mapping, and task metadata.
    """

    def __init__(
        self,
        network: Any,
        input_size: int,
        n_classes: int | None,
        task: str,
        label_map: dict[int, int] | None,
        inverse_label_map: dict[int, int] | None,
        y_mean: float,
        y_std: float,
    ) -> None:
        self.network = network
        self.input_size = input_size
        self.n_classes = n_classes
        self.task = task
        self.label_map = label_map
        self.inverse_label_map = inverse_label_map
        self.y_mean = y_mean
        self.y_std = y_std


class LSTMAlgorithm:
    """LSTM sequence model algorithm.

    Supports both regression (MSE loss) and classification (CrossEntropy).
    PyTorch is imported lazily inside ``train()`` so that the dependency is
    only required when this algorithm is actually used.

    The input feature matrix X is 2D (samples, features). Each sample is
    treated as a single-timestep sequence of shape (1, features) for the LSTM.
    """

    @property
    def name(self) -> str:
        return "lstm"

    @property
    def supports_regression(self) -> bool:
        return True

    @property
    def supports_classification(self) -> bool:
        return True

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        seed: int,
        **kwargs: Any,
    ) -> TrainedModel:
        """Train an LSTM model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            task: "regression" or "classification".
            seed: Random seed for reproducibility.
            **kwargs: Overrides for epochs, hidden_size, num_layers,
                      dropout, patience, learning_rate, batch_size.

        Returns:
            TrainedModel wrapping an ``_LSTMModelBundle``.

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: If the task is not supported.
        """
        import torch  # noqa: F811 -- lazy import
        import torch.nn as nn

        if task not in ("regression", "classification"):
            msg = f"Unsupported task: {task!r}. Expected 'regression' or 'classification'."
            raise ValueError(msg)

        # --- hyperparameters ---
        feature_names = kwargs.pop("feature_names", [f"f{i}" for i in range(X.shape[1])])
        epochs: int = kwargs.pop("epochs", 50)
        hidden_size: int = kwargs.pop("hidden_size", 64)
        num_layers: int = kwargs.pop("num_layers", 2)
        dropout: float = kwargs.pop("dropout", 0.1)
        patience: int = kwargs.pop("patience", 5)
        learning_rate: float = kwargs.pop("learning_rate", 1e-3)
        batch_size: int = kwargs.pop("batch_size", 64)

        # --- reproducibility ---
        torch.manual_seed(seed)
        np.random.seed(seed)  # noqa: NPY002 -- needed for data split

        input_size = X.shape[1]

        # --- prepare data ---
        X_float = X.astype(np.float32)

        # Normalise regression targets to zero mean / unit variance for
        # more stable LSTM training. Classification targets are left as-is.
        y_mean = 0.0
        y_std = 1.0
        label_map: dict[int, int] | None = None
        inverse_label_map: dict[int, int] | None = None
        n_classes: int | None = None

        if task == "regression":
            y_float = y.astype(np.float32)
            y_mean = float(np.mean(y_float))
            y_std = float(np.std(y_float))
            if y_std < 1e-8:
                y_std = 1.0
            y_norm = (y_float - y_mean) / y_std
        else:
            # Map labels to contiguous 0..K-1
            unique_labels = sorted(set(int(v) for v in y))
            label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
            inverse_label_map = {idx: orig for orig, idx in label_map.items()}
            n_classes = len(unique_labels)
            y_norm = np.array([label_map[int(v)] for v in y], dtype=np.int64)

        # Train / validation split (80/20)
        n_samples = len(X_float)
        n_val = max(1, int(n_samples * 0.2))
        n_train = n_samples - n_val
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train = torch.from_numpy(X_float[train_idx]).unsqueeze(1)  # (N, 1, F)
        X_val = torch.from_numpy(X_float[val_idx]).unsqueeze(1)

        if task == "regression":
            y_train = torch.from_numpy(y_norm[train_idx])
            y_val = torch.from_numpy(y_norm[val_idx])
        else:
            y_train = torch.from_numpy(y_norm[train_idx])
            y_val = torch.from_numpy(y_norm[val_idx])

        # --- build model ---
        network = _build_lstm_network(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            task=task,
            n_classes=n_classes,
        )

        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        if task == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # --- training loop with early stopping ---
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            network.train()

            # Mini-batch training
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                batch_idx = perm[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                optimizer.zero_grad()
                output = network(X_batch)

                if task == "regression":
                    loss = criterion(output.squeeze(-1), y_batch)
                else:
                    loss = criterion(output, y_batch)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            network.eval()
            with torch.no_grad():
                val_output = network(X_val)
                if task == "regression":
                    val_loss = criterion(val_output.squeeze(-1), y_val).item()
                else:
                    val_loss = criterion(val_output, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in network.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.debug(
                    "Early stopping at epoch %d (patience=%d)", epoch + 1, patience
                )
                break

        # Restore best weights
        if best_state is not None:
            network.load_state_dict(best_state)

        network.eval()

        bundle = _LSTMModelBundle(
            network=network,
            input_size=input_size,
            n_classes=n_classes,
            task=task,
            label_map=label_map,
            inverse_label_map=inverse_label_map,
            y_mean=y_mean,
            y_std=y_std,
        )

        return TrainedModel(
            model=bundle,
            algorithm_name=self.name,
            task=task,
            feature_names=feature_names,
        )

    def predict(self, model: TrainedModel, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the trained LSTM model."""
        import torch  # noqa: F811 -- lazy import

        bundle: _LSTMModelBundle = model.model
        network = bundle.network
        network.eval()

        X_tensor = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)

        with torch.no_grad():
            output = network(X_tensor)

        if bundle.task == "regression":
            raw = output.squeeze(-1).numpy()
            return (raw * bundle.y_std + bundle.y_mean).astype(np.float64)

        # Classification: argmax then map back to original labels
        pred_indices = output.argmax(dim=-1).numpy()
        if bundle.inverse_label_map is not None:
            return np.array(
                [bundle.inverse_label_map[int(idx)] for idx in pred_indices]
            )
        return pred_indices

    def predict_proba(
        self, model: TrainedModel, X: np.ndarray
    ) -> np.ndarray | None:
        """Return class probabilities for classification, None for regression."""
        if model.task == "regression":
            return None

        import torch  # noqa: F811 -- lazy import
        import torch.nn.functional as F  # noqa: N812

        bundle: _LSTMModelBundle = model.model
        network = bundle.network
        network.eval()

        X_tensor = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)

        with torch.no_grad():
            output = network(X_tensor)
            proba = F.softmax(output, dim=-1)

        return proba.numpy()

    def feature_importance(
        self, model: TrainedModel
    ) -> dict[str, float] | None:
        """LSTM does not provide built-in feature importance."""
        return None


# ---------------------------------------------------------------------------
# Internal network builder
# ---------------------------------------------------------------------------


def _build_lstm_network(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    task: str,
    n_classes: int | None,
) -> Any:
    """Build a PyTorch LSTM network.

    Returns an ``nn.Module`` with a forward() that accepts input of shape
    (batch, seq_len, features) and returns output of shape (batch, out_size).
    """
    import torch.nn as nn

    output_size = 1 if task == "regression" else (n_classes or 1)

    class _LSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: Any) -> Any:
            # x shape: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # Take the last timestep's output
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden)

    return _LSTMNet()
