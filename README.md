# Signature Authentication System (Signet)

Production-ready full-stack application for signature verification using a Siamese CNN pipeline.

- Frontend: Next.js 14 + TypeScript + Tailwind (deployed on Vercel)
- Backend: FastAPI + PyTorch + OpenCV (deployed on Railway)
- Inference: pairwise comparison of genuine/test signatures with heatmap output

---

## Table of Contents

1. [Overview](#overview)
2. [Live Deployment](#live-deployment)
3. [Architecture](#architecture)
4. [Repository Structure](#repository-structure)
5. [API Contract](#api-contract)
6. [Environment Variables](#environment-variables)
7. [Local Development](#local-development)
8. [Deployment Guide (Detailed)](#deployment-guide-detailed)
9. [Troubleshooting](#troubleshooting)
10. [Operational Notes](#operational-notes)

---

## Overview

Signet compares two signature images:

1. `genuine` (reference signature)
2. `test` (signature to verify)

The backend preprocesses both images, generates embeddings with a Siamese encoder, computes similarity, and returns:

- `is_authentic`
- `similarity_score`
- `confidence`
- `difference_heatmap` (base64 image)
- `verdict`

The frontend calls a Next.js API route (`/api/verify`) that proxies to the backend URL from `PYTHON_BACKEND_URL`.

---

## Live Deployment

- Frontend (Vercel): https://signet-rose.vercel.app/
- Backend (Railway): https://signet-production-a520.up.railway.app
- Backend health: https://signet-production-a520.up.railway.app/health

Expected health response:

```json
{
  "status": "ok",
  "service": "Signature Authentication API",
  "cuda_available": false
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

- Backend runs CPU-only PyTorch on Railway
- NumPy pinned to `<2` for PyTorch/OpenCV ABI compatibility
- Model loading is lazy (not eager at startup)
- Memory footprint reduced for low-memory containers:
  - lighter CNN layers
  - adaptive global pooling
  - reduced input size (default `128x128`)
  - limited Torch threads via env

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
├─ railway.toml
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
  "verdict": "Genuine"
}
```

Error responses:

- `400`: invalid/missing files or preprocess failure
- `500`: model/inference/runtime errors

---

## Environment Variables

### Frontend (Vercel)

- `PYTHON_BACKEND_URL`
  - Example: `https://signet-production-a520.up.railway.app`
  - Required in Production, Preview, Development

### Backend (Railway)

- `ALLOWED_ORIGINS`
  - Example: `https://signet-rose.vercel.app`
  - Comma-separated list supported by app logic
- `PORT`
  - Railway sets this automatically
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

### A) Backend on Railway

1. Create Railway project from GitHub repo.
2. Ensure Docker build uses root `railway.toml` and `backend/Dockerfile`.
3. Add backend variables:
   - `ALLOWED_ORIGINS=https://<your-frontend-domain>`
4. Deploy and wait for success.
5. Validate:

```bash
curl https://<your-railway-domain>/health
```

### B) Frontend on Vercel

1. Import same repository in Vercel.
2. Set Root Directory to `frontend`.
3. Add env variable:
   - `PYTHON_BACKEND_URL=https://<your-railway-domain>`
4. Deploy.
5. If env vars are added after first deploy, redeploy once.

### C) Link verification checklist

- Backend health returns `200`
- Vercel env has correct backend URL
- Railway `ALLOWED_ORIGINS` contains frontend domain
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

### Railway Out of Memory

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

Set `ALLOWED_ORIGINS` on Railway to exact frontend domain(s), for example:

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

- [ ] Railway latest deploy = success
- [ ] Vercel latest deploy = ready
- [ ] Backend `/health` = 200
- [ ] Frontend verify request = 200
- [ ] Genuine pair and forged pair tested

---

## License

This project is for educational and demonstration use unless otherwise specified by repository owner.
