"""
FastAPI Backend for Signature Authentication System
Implements a Siamese CNN network for signature verification
"""

import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import torch
import gc

from model.siamese_model import SiameseModel
from utils.preprocess import preprocess_signature
from utils.similarity import compute_similarity, compute_heatmap

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

# Global model instance
model = None


def load_model():
    """Load or initialize the Siamese model"""
    global model
    try:
        logger.info("Loading Siamese model...")
        model = SiameseModel()
        model.load_pretrained_weights()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("Model will be loaded on first request")
        # Don't raise - allow app to start


@app.on_event("startup")
async def startup_event():
    """Startup hook for lightweight boot on low-memory platforms"""
    logger.info("Skipping eager model load to reduce startup memory usage")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Signature Authentication API",
        "cuda_available": torch.cuda.is_available()
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
        # Lazy load model if not already loaded
        if model is None:
            logger.info("Model not loaded yet, loading now...")
            load_model()
            if model is None:
                raise HTTPException(status_code=500, detail="Model failed to load")
        
        # Validate file types
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
        genuine_ext = os.path.splitext(genuine.filename)[1].lower()
        test_ext = os.path.splitext(test.filename)[1].lower()
        
        if genuine_ext not in valid_extensions or test_ext not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(valid_extensions)}"
            )
        
        # Read and preprocess images
        logger.info(f"Processing genuine signature: {genuine.filename}")
        logger.info(f"Processing test signature: {test.filename}")
        
        genuine_bytes = await genuine.read()
        test_bytes = await test.read()
        
        genuine_tensor = preprocess_signature(genuine_bytes)
        test_tensor = preprocess_signature(test_bytes)
        
        if genuine_tensor is None or test_tensor is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to preprocess images. Ensure images are valid signatures."
            )
        
        # Generate embeddings using the model
        with torch.no_grad():
            genuine_embedding = model.encoder(genuine_tensor)
            test_embedding = model.encoder(test_tensor)
        
        # Compute similarity score
        similarity_score, confidence = compute_similarity(
            genuine_embedding,
            test_embedding
        )
        
        # Determine verdict based on similarity thresholds
        if similarity_score >= 0.85:
            verdict = "Genuine"
            is_authentic = True
            conf_level = "High"
        elif similarity_score >= 0.70:
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
        
        return JSONResponse(
            status_code=200,
            content={
                "is_authentic": is_authentic,
                "similarity_score": float(similarity_score),
                "confidence": conf_level,
                "difference_heatmap": heatmap_b64,
                "verdict": verdict
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
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
