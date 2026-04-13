from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class ONNXQModel:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, state_batch: np.ndarray) -> np.ndarray:
        state_batch = np.asarray(state_batch, dtype=np.float32)
        if state_batch.ndim == 3:
            state_batch = np.expand_dims(state_batch, axis=0)
        return self.session.run(None, {self.input_name: state_batch})[0]
