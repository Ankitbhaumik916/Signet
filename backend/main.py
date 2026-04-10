"""
FastAPI Backend for Signature Authentication System
Implements a Siamese CNN network for signature verification
"""

import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import torch
import gc
import cv2
import numpy as np
import json
from datetime import datetime, timezone
from time import perf_counter
from contextvars import ContextVar
from threading import Lock

from model.siamese_model import SiameseModel
from utils.embedding_cache import EmbeddingLRUCache
from utils.preprocess import preprocess_signature
from utils.similarity import (
    blend_similarity_scores,
    compute_classical_similarity,
    compute_heatmap,
    compute_similarity,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1")))

# Initialize FastAPI app
app = FastAPI(
    title="Signature Authentication API",
    description="AI-powered signature verification using Siamese CNN",
    version="1.0.0"
)

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request, call_next):
    started_at = perf_counter()
    request_context.set({})

    response = await call_next(request)

    if request.url.path == "/verify":
        latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
        context = request_context.get({})
        _append_request_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "inference_mode": context.get("inference_mode"),
                "neural_score": context.get("neural_score"),
                "classical_score": context.get("classical_score"),
                "verdict": context.get("verdict"),
                "latency_ms": latency_ms,
            }
        )

    return response

# Global model instance
model = None
embedding_cache = EmbeddingLRUCache(max_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "256")))
request_context: ContextVar[dict] = ContextVar("request_context", default={})
REQUEST_LOG_PATH = os.getenv("REQUEST_LOG_PATH", "/tmp/verify_requests.jsonl")
VERIFICATION_COUNTER_PATH = Path(
    os.getenv(
        "VERIFICATION_COUNTER_PATH",
        str(Path(__file__).resolve().parent / "verification_counter.json"),
    )
)
verification_counter_lock = Lock()
verification_counter = 0


def _load_verification_counter() -> int:
    try:
        if VERIFICATION_COUNTER_PATH.exists():
            with open(VERIFICATION_COUNTER_PATH, "r", encoding="utf-8") as counter_file:
                payload = json.load(counter_file)
                return int(payload.get("verification_count", 0))
    except Exception as exc:
        logger.warning("Failed to load verification counter: %s", exc)

    return 0


def _persist_verification_counter(count: int) -> None:
    try:
        with open(VERIFICATION_COUNTER_PATH, "w", encoding="utf-8") as counter_file:
            json.dump({"verification_count": count}, counter_file)
    except Exception as exc:
        logger.warning("Failed to persist verification counter: %s", exc)


def _increment_verification_counter() -> int:
    global verification_counter
    with verification_counter_lock:
        verification_counter += 1
        _persist_verification_counter(verification_counter)
        return verification_counter


def _load_weights_from_candidates(model_wrapper: SiameseModel) -> str | None:
    """Try common local weight paths for HF Spaces startup and return loaded path."""
    configured_path = os.getenv("MODEL_WEIGHTS_PATH", "").strip()
    backend_dir = Path(__file__).resolve().parent
    project_dir = backend_dir.parent
    workspace_dir = project_dir.parent
    candidates = [
        configured_path,
        str(backend_dir / "model_weights" / "siamese_model.pth"),
        str(backend_dir / "model_weights" / "siamese_model.pt"),
        str(project_dir / "model_weights" / "siamese_model.pth"),
        str(project_dir / "model_weights" / "siamese_model.pt"),
        str(project_dir / "siamese_model.pth"),
        str(project_dir / "siamese_model.pt"),
        str(workspace_dir / "siamese_model.pth"),
        str(workspace_dir / "siamese_model.pt"),
        str(workspace_dir / "Signet_bck" / "model_weights" / "siamese_model.pth"),
        str(workspace_dir / "Signet_bck" / "model_weights" / "siamese_model.pt"),
        "/app/model_weights/siamese_model.pth",
        "/app/model_weights/siamese_model.pt",
        "/tmp/signature_model_weights/siamese_model.pth",
        "/tmp/signature_model_weights/siamese_model.pt",
    ]

    for candidate in candidates:
        if not candidate:
            continue

        weight_path = Path(candidate)
        if not weight_path.exists():
            continue

        try:
            logger.info("Attempting to load startup weights from %s", weight_path)
            checkpoint = torch.load(weight_path, map_location=model_wrapper.device)

            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )

            model_wrapper.model.load_state_dict(state_dict, strict=True)
            model_wrapper.model.eval()
            model_wrapper.encoder.eval()
            model_wrapper.has_pretrained_weights = True
            logger.info("Loaded pretrained weights from %s", weight_path)
            return str(weight_path)
        except Exception as exc:
            logger.warning("Failed to load weights from %s: %s", weight_path, exc)

    return None


def _json_error(status_code: int, error_code: str, message: str, details: dict | None = None) -> JSONResponse:
    content = {
        "error": message,
        "error_code": error_code,
    }
    if details:
        content["details"] = details
    return JSONResponse(status_code=status_code, content=content)


def _is_supported_mime(upload: UploadFile) -> bool:
    allowed_mimes = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/webp",
    }
    return (upload.content_type or "").lower() in allowed_mimes


def _decode_image(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None

    nparr = np.frombuffer(image_bytes, np.uint8)
    if nparr.size == 0:
        return None

    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _is_near_empty_image(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_ratio = float((binary < 250).mean())
    return ink_ratio < 0.002


def _set_request_result_context(
    inference_mode: str | None,
    neural_score: float | None,
    classical_score: float | None,
    verdict: str | None,
) -> None:
    request_context.set(
        {
            "inference_mode": inference_mode,
            "neural_score": neural_score,
            "classical_score": classical_score,
            "verdict": verdict,
        }
    )


def _append_request_log(record: dict) -> None:
    try:
        with open(REQUEST_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as exc:
        logger.warning("Failed to write request log: %s", exc)


def load_model():
    """Load or initialize the Siamese model"""
    global model
    try:
        logger.info("Loading Siamese model...")
        model = SiameseModel()
        model.load_pretrained_weights()

        if not getattr(model, "has_pretrained_weights", False):
            loaded_path = _load_weights_from_candidates(model)
            if loaded_path:
                logger.info("Startup weights active: %s", loaded_path)

        model.model.eval()
        model.encoder.eval()
        logger.info(
            "Model ready. has_pretrained_weights=%s",
            getattr(model, "has_pretrained_weights", False),
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        model = None

@app.on_event("startup")
async def startup_event():
    """Startup hook that eagerly initializes model so mode is known before first request."""
    logger.info("Running eager model initialization at startup")
    global verification_counter
    verification_counter = _load_verification_counter()
    logger.info("Loaded verification counter=%s", verification_counter)
    load_model()


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    current_mode = "classical-fallback"
    if model is not None and getattr(model, "has_pretrained_weights", False):
        current_mode = "hybrid-neural"

    return {
        "status": "ok",
        "service": "Signature Authentication API",
        "cuda_available": torch.cuda.is_available(),
        "inference_mode": current_mode,
        "verification_count": verification_counter,
    }


@app.post("/verify", tags=["Verification"])
async def verify_signature(
    genuine: UploadFile = File(...),
    test: UploadFile = File(...)
):
    """
    Verify if a test signature matches a genuine signature
    
    Args:
        genuine: Reference/genuine signature image
        test: Test signature image to verify
    
    Returns:
        JSON response with verification results including:
        - is_authentic: bool
        - similarity_score: float (0.0-1.0)
        - confidence: str (High/Medium/Low)
        - difference_heatmap: base64 encoded image
        - verdict: str (Genuine/Forged/Suspicious)
    """
    try:
        # Validate file extension types
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
        genuine_ext = os.path.splitext(genuine.filename)[1].lower()
        test_ext = os.path.splitext(test.filename)[1].lower()
        
        if genuine_ext not in valid_extensions or test_ext not in valid_extensions:
            return _json_error(
                status_code=400,
                error_code="INVALID_FILE_EXTENSION",
                message=f"Invalid file type. Allowed: {', '.join(valid_extensions)}",
                details={
                    "genuine_extension": genuine_ext,
                    "test_extension": test_ext,
                },
            )

        # Validate MIME types
        if not _is_supported_mime(genuine) or not _is_supported_mime(test):
            return _json_error(
                status_code=400,
                error_code="INVALID_MIME_TYPE",
                message="Only JPEG, PNG, GIF, BMP, or WEBP images are allowed.",
                details={
                    "genuine_content_type": genuine.content_type,
                    "test_content_type": test.content_type,
                },
            )
        
        # Read and preprocess images
        logger.info(f"Processing genuine signature: {genuine.filename}")
        logger.info(f"Processing test signature: {test.filename}")
        
        genuine_bytes = await genuine.read()
        test_bytes = await test.read()

        # Validate image byte payloads and decode integrity
        genuine_image = _decode_image(genuine_bytes)
        test_image = _decode_image(test_bytes)
        if genuine_image is None or test_image is None:
            return _json_error(
                status_code=400,
                error_code="CORRUPT_IMAGE",
                message="One or both uploaded images are corrupt or unreadable.",
            )

        # Early near-empty gate before model execution
        if _is_near_empty_image(genuine_image) or _is_near_empty_image(test_image):
            return _json_error(
                status_code=400,
                error_code="LOW_SIGNAL_IMAGE",
                message="One or both images appear empty or contain too little signature signal.",
            )
        
        # Lazy model load for memory-friendly startup.
        global model
        if model is None:
            load_model()

        target_device = "cpu"
        if model is not None:
            target_device = str(getattr(model, "device", "cpu"))

        genuine_tensor = preprocess_signature(genuine_bytes, device=target_device)
        test_tensor = preprocess_signature(test_bytes, device=target_device)
        
        if genuine_tensor is None or test_tensor is None:
            return _json_error(
                status_code=400,
                error_code="PREPROCESS_FAILED",
                message="Failed to preprocess images. Ensure images are valid signatures.",
            )

        neural_score = None
        classical_score, _ = compute_classical_similarity(genuine_tensor, test_tensor)

        has_neural = model is not None and getattr(model, "has_pretrained_weights", False)
        if has_neural:
            reference_key = embedding_cache.key_from_bytes(genuine_bytes)
            cached_reference_embedding = embedding_cache.get(reference_key, device=target_device)

            with torch.inference_mode():
                if cached_reference_embedding is None:
                    genuine_embedding = model.encoder(genuine_tensor)
                    embedding_cache.put(reference_key, genuine_embedding)
                else:
                    genuine_embedding = cached_reference_embedding

                test_embedding = model.encoder(test_tensor)

            neural_score, _ = compute_similarity(genuine_embedding, test_embedding)
            similarity_score = blend_similarity_scores(neural_score, classical_score)
            inference_mode = "hybrid-neural"
        else:
            similarity_score = classical_score
            inference_mode = "classical-fallback"

        if inference_mode == "hybrid-neural":
            if similarity_score >= 0.90:
                verdict = "Genuine"
                is_authentic = True
                conf_level = "High"
            elif similarity_score >= 0.80:
                verdict = "Suspicious"
                is_authentic = False
                conf_level = "Medium"
            else:
                verdict = "Forged"
                is_authentic = False
                conf_level = "High"
        else:
            if similarity_score >= 0.93:
                verdict = "Genuine"
                is_authentic = True
                conf_level = "High"
            elif similarity_score >= 0.82:
                verdict = "Suspicious"
                is_authentic = False
                conf_level = "Medium"
            else:
                verdict = "Forged"
                is_authentic = False
                conf_level = "High"
        
        # Generate difference heatmap
        heatmap_b64 = compute_heatmap(genuine_tensor, test_tensor)
        
        logger.info(f"Verification complete - Verdict: {verdict}, Score: {similarity_score:.4f}")
        
        response_payload = JSONResponse(
            status_code=200,
            content={
                "is_authentic": is_authentic,
                "similarity_score": float(similarity_score),
                "confidence": conf_level,
                "difference_heatmap": heatmap_b64,
                "verdict": verdict,
                "inference_mode": inference_mode,
                "neural_score": neural_score,
                "classical_score": float(classical_score),
                "verification_count": _increment_verification_counter(),
            }
        )

        _set_request_result_context(
            inference_mode=inference_mode,
            neural_score=neural_score,
            classical_score=float(classical_score),
            verdict=verdict,
        )

        return response_payload
    
    except HTTPException:
        _set_request_result_context(None, None, None, None)
        raise
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}", exc_info=True)
        _set_request_result_context(None, None, None, None)
        return _json_error(
            status_code=500,
            error_code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
            details={"reason": str(e)},
        )
    finally:
        gc.collect()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Signature Authentication API",
        "docs": "/docs",
        "health": "/health",
        "verification_count": verification_counter,
        "endpoints": {
            "verify": "POST /verify"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
