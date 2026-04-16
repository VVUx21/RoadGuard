import os
from http import HTTPStatus

from flask import Flask, jsonify, request

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover
    CORS = None

from inference_service import InferenceService


app = Flask(__name__)
if CORS is not None:
    CORS(app)

service = InferenceService(
    binary_model_path=os.getenv("BINARY_MODEL_PATH", "binary_pothole_cnn.keras"),
    severity_model_path=os.getenv("SEVERITY_MODEL_PATH", "severity_pothole_cnn.keras"),
)

@app.get("/")
def root():
    return jsonify(
        {
            "service": "RoadGuard Pothole API",
            "endpoints": {
                "GET /health": "Service and model status",
                "POST /predict": "Predict from multipart image file or base64 payload",
                "POST /reload-models": "Reload model files from disk",
            },
        }
    )


@app.get("/health")
def health():
    status = service.model_status()
    loaded = status["binary_model_loaded"] and status["severity_model_loaded"]
    response_code = HTTPStatus.OK if loaded else HTTPStatus.PARTIAL_CONTENT
    return jsonify({"ok": loaded, "models": status}), response_code


@app.post("/reload-models")
def reload_models():
    payload = request.get_json(silent=True) or {}
    binary_model_path = payload.get("binary_model_path")
    severity_model_path = payload.get("severity_model_path")

    service.reload_models(
        binary_model_path=binary_model_path,
        severity_model_path=severity_model_path,
    )

    return jsonify(
        {
            "message": "Models reloaded.",
            "models": service.model_status(),
        }
    )


@app.post("/predict")
def predict():
    try:
        if "image" in request.files:
            file_storage = request.files["image"]
            if not file_storage or file_storage.filename == "":
                return (
                    jsonify({"error": "Image file is empty."}),
                    HTTPStatus.BAD_REQUEST,
                )
            prediction = service.predict_from_bytes(file_storage.read())
            return jsonify(prediction)

        payload = request.get_json(silent=True) or {}
        image_base64 = payload.get("image_base64")
        if image_base64:
            prediction = service.predict_from_base64(image_base64)
            return jsonify(prediction)

        return (
            jsonify(
                {
                    "error": (
                        "Provide image as multipart/form-data with key 'image' "
                        "or JSON with key 'image_base64'."
                    )
                }
            ),
            HTTPStatus.BAD_REQUEST,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.SERVICE_UNAVAILABLE
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Unexpected error: {exc}"}), HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
