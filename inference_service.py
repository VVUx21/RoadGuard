import base64
import io
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 100
DEFAULT_BINARY_MODEL_PATH = "binary_pothole_cnn.keras"
DEFAULT_SEVERITY_MODEL_PATH = "severity_pothole_cnn.keras"
LABEL_NAMES = ["minor", "moderate", "severe"]


@dataclass
class ModelArtifacts:
    binary_model: Optional[tf.keras.Model]
    severity_model: Optional[tf.keras.Model]
    binary_model_path: str
    severity_model_path: str


class InferenceService:
    def __init__(
        self,
        binary_model_path: str = DEFAULT_BINARY_MODEL_PATH,
        severity_model_path: str = DEFAULT_SEVERITY_MODEL_PATH,
    ) -> None:
        self.artifacts = ModelArtifacts(
            binary_model=None,
            severity_model=None,
            binary_model_path=binary_model_path,
            severity_model_path=severity_model_path,
        )
        self.try_load_models()

    def try_load_models(self) -> None:
        self.artifacts.binary_model = self._safe_load_model(self.artifacts.binary_model_path)
        self.artifacts.severity_model = self._safe_load_model(self.artifacts.severity_model_path)

    def reload_models(
        self,
        binary_model_path: Optional[str] = None,
        severity_model_path: Optional[str] = None,
    ) -> None:
        if binary_model_path:
            self.artifacts.binary_model_path = binary_model_path
        if severity_model_path:
            self.artifacts.severity_model_path = severity_model_path
        self.try_load_models()

    def model_status(self) -> Dict[str, object]:
        binary_exists = os.path.exists(self.artifacts.binary_model_path)
        severity_exists = os.path.exists(self.artifacts.severity_model_path)
        return {
            "binary_model_path": self.artifacts.binary_model_path,
            "severity_model_path": self.artifacts.severity_model_path,
            "binary_model_file_exists": binary_exists,
            "severity_model_file_exists": severity_exists,
            "binary_model_loaded": self.artifacts.binary_model is not None,
            "severity_model_loaded": self.artifacts.severity_model is not None,
        }

    def predict_from_bytes(self, file_bytes: bytes) -> Dict[str, object]:
        if self.artifacts.binary_model is None:
            raise RuntimeError(
                "Binary model is not loaded. Train the model and ensure binary_pothole_cnn.keras is present."
            )

        tensor = self._preprocess_image_bytes(file_bytes)

        bin_probs = self.artifacts.binary_model.predict(tensor, verbose=0)[0]
        class_idx = int(np.argmax(bin_probs))
        binary_confidence = float(bin_probs[class_idx])

        if class_idx == 0:
            return {
                "detection": "normal",
                "severity": "N/A",
                "binary_confidence": binary_confidence,
                "severity_confidence": None,
                "all_binary_probabilities": {
                    "normal": float(bin_probs[0]),
                    "pothole": float(bin_probs[1]),
                },
            }

        if self.artifacts.severity_model is None:
            raise RuntimeError(
                "Severity model is not loaded. Train the model and ensure severity_pothole_cnn.keras is present."
            )

        sev_probs = self.artifacts.severity_model.predict(tensor, verbose=0)[0]
        sev_idx = int(np.argmax(sev_probs))
        severity_confidence = float(sev_probs[sev_idx])

        return {
            "detection": "pothole",
            "severity": LABEL_NAMES[sev_idx],
            "binary_confidence": binary_confidence,
            "severity_confidence": severity_confidence,
            "all_binary_probabilities": {
                "normal": float(bin_probs[0]),
                "pothole": float(bin_probs[1]),
            },
            "all_severity_probabilities": {
                LABEL_NAMES[i]: float(sev_probs[i]) for i in range(len(LABEL_NAMES))
            },
        }

    def predict_from_base64(self, image_base64: str) -> Dict[str, object]:
        payload = image_base64.split(",", 1)[-1]
        file_bytes = base64.b64decode(payload)
        return self.predict_from_bytes(file_bytes)

    @staticmethod
    def _safe_load_model(model_path: str) -> Optional[tf.keras.Model]:
        if not os.path.exists(model_path):
            return None
        return tf.keras.models.load_model(model_path)

    @staticmethod
    def _preprocess_image_bytes(file_bytes: bytes) -> np.ndarray:
        image_array = np.frombuffer(file_bytes, dtype=np.uint8)
        gray = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Unable to decode image. Send a valid JPG or PNG file.")

        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        tensor = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
        return tensor
