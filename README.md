# Signature Authentication System (Signet)

Production-ready full-stack application for signature verification using a Siamese CNN pipeline.

- Frontend: Next.js 14 + TypeScript + Tailwind (deployed on Vercel)
- Backend: FastAPI + PyTorch + OpenCV
- Inference: pairwise comparison of genuine/test signatures with heatmap output

---

## Table of Contents

1. [Overview](#overview)
2. [Live Deployment](#live-deployment)
3. [Architecture](#architecture)
4. [ML Model Details](#ml-model-details)
5. [Repository Structure](#repository-structure)
6. [API Contract](#api-contract)
7. [Environment Variables](#environment-variables)
8. [Local Development](#local-development)
9. [Deployment Guide (Detailed)](#deployment-guide-detailed)
10. [Troubleshooting](#troubleshooting)
11. [Operational Notes](#operational-notes)

---

## Overview

Signet compares two signature images:

1. `genuine` (reference signature)
2. `test` (signature to verify)

The backend preprocesses both images, computes neural and classical similarity signals, and returns:

- `is_authentic`
- `similarity_score`
- `confidence`
- `difference_heatmap` (base64 image)
- `verdict`
- `inference_mode`
- `neural_score` (when neural weights are available)
- `classical_score`

The frontend calls a Next.js API route (`/api/verify`) that proxies to the backend URL from `PYTHON_BACKEND_URL`.

---

## Live Deployment

- Frontend (Vercel): https://signet-rose.vercel.app/
- Backend (Vercel): https://signet-backend.vercel.app
- Backend health: https://signet-backend.vercel.app/health

Expected health response:

```json
{
  "status": "ok",
  "service": "Signature Authentication API",
  "cuda_available": false,
  "inference_mode": "hybrid-neural"
}
```

---

## Architecture

### Request Flow

1. User uploads two images in frontend UI
2. Frontend sends `multipart/form-data` to `POST /api/verify`
3. Next.js route forwards payload to `${PYTHON_BACKEND_URL}/verify`
4. FastAPI preprocesses images and runs model inference
5. Backend returns verdict + score + heatmap JSON
6. Frontend renders result card and visualization

### Runtime Design Choices

- Backend uses a Siamese CNN when pretrained weights are available
- Classical similarity stays enabled as a robust fallback
- Hybrid score blending improves robustness on noisy signatures
- NumPy pinned to `<2` for Torch/OpenCV compatibility
- Memory footprint reduced for low-memory containers:
  - reduced input size (default `128x128`)
  - lazy model loading at first verification request

---

## ML Model Details

This section describes the active prediction path in the current codebase.

### 1) Input and preprocessing pipeline

For each uploaded image (`genuine`, `test`), backend preprocessing performs:

1. Decode image bytes using OpenCV
2. Convert to grayscale
3. Apply Gaussian blur
4. Apply Otsu thresholding
5. Find signature region via contour bounding box
6. Crop and center signature region
7. Resize to `128x128`
8. Normalize pixel values to `[0, 1]`
9. Convert to tensor shape `(1, 1, H, W)`

Key implementation: [backend/utils/preprocess.py](backend/utils/preprocess.py)

### 2) Siamese encoder architecture

Current encoder is lightweight for serverless memory constraints:

- Conv(1→32) + ReLU + MaxPool
- Conv(32→64) + ReLU + MaxPool
- Conv(64→128) + ReLU + MaxPool
- Conv(128→256) + ReLU + MaxPool
- AdaptiveAvgPool to `(1,1)`
- Dense `256 → 256` + Dropout(0.2)
- Dense `256 → 128` + Dropout(0.2)
- L2 normalization of final embedding

The Siamese network shares this encoder for both inputs and compares embedding similarity.

Key implementation: [backend/model/siamese_model.py](backend/model/siamese_model.py)

### 3) Dual inference modes (important)

The backend supports two runtime modes and automatically chooses based on model availability:

#### A) `hybrid-neural` mode

Used when pretrained Siamese weights are available.

- Computes neural similarity from Siamese embeddings
- Computes classical similarity from SSIM + pixel agreement
- Blends both scores for robustness (`0.85 * neural + 0.15 * classical`)
- Thresholds:
  - `>= 0.90` -> `Genuine`
  - `0.80 - 0.8999` -> `Suspicious`
  - `< 0.80` -> `Forged`

#### B) `classical-fallback` mode

Used when pretrained neural weights are unavailable or fail to load.

- Computes SSIM (structural similarity)
- Computes pixel agreement (`1 - mean absolute difference`)
- Blended score: `0.7 * SSIM + 0.3 * pixel_agreement`
- Stricter thresholds:
  - `>= 0.93` → `Genuine`
  - `0.82 - 0.9299` → `Suspicious`
  - `< 0.82` → `Forged`

The API response includes `inference_mode`, `neural_score`, and `classical_score` for transparency.

Key implementation:

- [backend/main.py](backend/main.py)
- [backend/utils/similarity.py](backend/utils/similarity.py)

### 4) Heatmap generation

The system generates a visual difference map:

- Pixel-wise absolute difference
- Colormap mapping (OpenCV)
- Optional overlay on the test signature
- Returned as base64 PNG in `difference_heatmap`

This is an explainability aid, not a calibrated forgery detector by itself.

### 5) Why predictions may still be imperfect

Even with safeguards, signature verification quality depends on:

- Whether true trained domain weights are available
- Consistency of image capture (angle, pen color, lighting)
- Background noise and cropping quality
- Genuine intra-writer variation

The hybrid mode improves robustness, but final quality still depends on model weights and calibration quality.

### 6) Recommended model-improvement roadmap

For better accuracy in real-world use:

1. Train Siamese model on a labeled signature dataset (genuine/forged pairs)
2. Export and ship fixed versioned weights with integrity checks
3. Add threshold calibration from validation ROC/EER analysis
4. Add signature quality gate (reject non-signature/random natural images)
5. Track metrics in production (false accept / false reject rates)

### 7) Safety and interpretation guidance

- Treat current output as decision support, not legal proof.
- For high-stakes verification, combine with human review and additional checks.
- Log `inference_mode`, scores, and input quality metadata for audits.

---

## Repository Structure

```text
signature-auth/
├─ frontend/
│  ├─ app/
│  │  ├─ page.tsx
│  │  ├─ layout.tsx
│  │  └─ api/verify/route.ts
│  ├─ components/
│  ├─ public/
│  ├─ package.json
│  └─ vercel.json
├─ backend/
│  ├─ main.py
│  ├─ Dockerfile
│  ├─ requirements.txt
│  ├─ pyproject.toml
│  ├─ model/siamese_model.py
│  └─ utils/
│     ├─ preprocess.py
│     └─ similarity.py
├─ backend/vercel.json
├─ DEPLOYMENT.md
├─ QUICKSTART.md
└─ README.md
```

---

## API Contract

### `GET /`

Basic metadata and endpoint info.

### `GET /health`

Health endpoint for uptime checks.

### `POST /verify`

Consumes `multipart/form-data` with fields:

- `genuine`: image file (`jpg`, `jpeg`, `png`, `gif`, `bmp`)
- `test`: image file (`jpg`, `jpeg`, `png`, `gif`, `bmp`)

Success response example:

```json
{
  "is_authentic": true,
  "similarity_score": 0.91,
  "confidence": "High",
  "difference_heatmap": "data:image/png;base64,...",
  "verdict": "Genuine",
  "inference_mode": "hybrid-neural",
  "neural_score": 0.92,
  "classical_score": 0.87
}
```

Error responses:

- `400`: invalid/missing files or preprocess failure
- `500`: model/inference/runtime errors

---

## Environment Variables

### Frontend (Vercel)

- `PYTHON_BACKEND_URL`
  - Example: `https://signet-backend.vercel.app`
  - Required in Production, Preview, Development

### Backend (Vercel)

- `ALLOWED_ORIGINS`
  - Example: `https://signet-rose.vercel.app`
  - Comma-separated list supported by app logic
- `PORT`
  - Vercel sets this automatically for the function runtime
- `TORCH_NUM_THREADS` (optional)
  - Default: `1`
- `TORCH_NUM_INTEROP_THREADS` (optional)
  - Default: `1`

---

## Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm

### 1) Clone

```bash
git clone https://github.com/Ankitbhaumik916/Signet.git
cd Signet/signature-auth
```

### 2) Backend setup

```bash
cd backend
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run backend:

```bash
python main.py
```

Backend local URL: `http://localhost:8000`

### 3) Frontend setup

```bash
cd ../frontend
npm install
```

Create `.env.local`:

```env
PYTHON_BACKEND_URL=http://localhost:8000
```

Run frontend:

```bash
npm run dev
```

Frontend local URL: `http://localhost:3000`

---

## Deployment Guide (Detailed)

### A) Backend on Vercel

1. Import the repository into Vercel.
2. Set the Root Directory to `backend`.
3. Confirm Vercel uses `backend/vercel.json` and `backend/api/index.py`.
4. Add backend variables:
  - `ALLOWED_ORIGINS=https://<your-frontend-domain>`
5. Deploy and wait for success.
6. Validate:

```bash
curl https://<your-vercel-backend-domain>/health
```

### B) Frontend on Vercel

1. Import same repository in Vercel.
2. Set Root Directory to `frontend`.
3. Add env variable:
  - `PYTHON_BACKEND_URL=https://<your-vercel-backend-domain>`
4. Deploy.
5. If env vars are added after first deploy, redeploy once.

### C) Link verification checklist

- Backend health returns `200`
- Vercel env has correct backend URL
- Backend `ALLOWED_ORIGINS` contains frontend domain
- `/api/verify` in browser network tab returns `200`

---

## Troubleshooting

### `502 Bad Gateway` on frontend verify

Usually caused by missing/wrong `PYTHON_BACKEND_URL` in Vercel.

Fix:

1. Set `PYTHON_BACKEND_URL` correctly
2. Redeploy Vercel
3. Re-test

### Vercel error: secret reference not found

If `vercel.json` or dashboard env references a non-existing secret, replace with plain value in dashboard env vars.

### Vercel Out of Memory

Symptoms: backend crashes during `/verify`, intermittent 5xx.

Mitigations already applied in code:

- lightweight encoder
- lower input resolution
- lazy load model
- reduced torch threads

Operationally:

- use latest backend deploy commit
- avoid very large input images
- scale plan/resources if traffic grows

### CORS errors in browser

Set `ALLOWED_ORIGINS` on the backend deployment to exact frontend domain(s), for example:

`https://signet-rose.vercel.app`

Then redeploy backend.

### NumPy/Torch ABI errors (`_ARRAY_API` / `multiarray failed to import`)

Keep NumPy constrained to `<2` in backend dependencies.

---

## Operational Notes

- Health endpoint checks service availability only.
- First `/verify` request can be slightly slower after cold start due to lazy model load.
- Logging is enabled via Python logger in backend service.

### Production validation checklist

- [ ] Backend Vercel deploy = success
- [ ] Vercel latest deploy = ready
- [ ] Backend `/health` = 200
- [ ] Frontend verify request = 200
- [ ] Genuine pair and forged pair tested

---

## License

This project is for educational and demonstration use unless otherwise specified by repository owner.
