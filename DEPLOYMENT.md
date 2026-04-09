# Deployment Guide - Signature Authentication System

This guide covers two deployment paths for this repository:

- Frontend on Vercel (recommended)
- Backend on either Vercel (limited) or Hugging Face Spaces Docker (recommended for hybrid-neural Siamese mode)

---

## Recommended Architecture

For best model quality:

1. Deploy backend to **Hugging Face Spaces (Docker)**
2. Deploy frontend to **Vercel**
3. Set frontend `PYTHON_BACKEND_URL` to your Spaces URL

Reason: the restored backend uses PyTorch + Siamese CNN with hybrid scoring, which is more reliable on container-style hosting than strict serverless limits.

---

## Part 1: Backend Deployment on Hugging Face Spaces (Docker)

### What is already prepared in this repo

- Backend Docker image: `backend/Dockerfile`
- FastAPI app entrypoint: `backend/main.py`
- Requirements (including torch): `backend/requirements.txt`
- Small image context optimization: `backend/.dockerignore`

### Space settings you need

Create a new Space at <https://huggingface.co/new-space> with:

- SDK: `Docker`
- Visibility: `Public` or `Private` (your choice)
- Hardware: `CPU Basic` (free)
- App Port: `7860`

### Environment variables in Space Settings

Add these under **Settings -> Variables and secrets**:

- `ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app`
- `TORCH_NUM_THREADS=1`
- `TORCH_NUM_INTEROP_THREADS=1`

### Step-by-step deploy

1. Create the Docker Space.
2. Clone the new Space repository locally.
3. Copy all files from your local `signature-auth/backend` folder into the Space repo root.
4. Commit and push to the Space repo.
5. Wait for build + startup to complete in the Space logs.
6. Test health:

```bash
curl https://<your-space-subdomain>.hf.space/health
```

Expected JSON includes:

```json
{
  "status": "ok",
  "service": "Signature Authentication API",
  "cuda_available": false,
  "inference_mode": "hybrid-neural"
}
```

If pretrained weights are unavailable, `inference_mode` will be `classical-fallback`.

---

## Part 2: Frontend Deployment on Vercel

1. Import this GitHub repository in Vercel.
2. Set root directory to `frontend`.
3. Add environment variable:

```env
PYTHON_BACKEND_URL=https://<your-space-subdomain>.hf.space
```

4. Deploy frontend.
5. Re-test from UI with at least one genuine pair and one forged pair.

---

## Optional: Backend on Vercel (Not Preferred for Neural)

You can still deploy backend to Vercel from `backend/`, but serverless limits can cause cold-start or runtime failures with heavy ML dependencies.

If you choose this path:

- Keep `backend/vercel.json` as provided
- Set `ALLOWED_ORIGINS` in backend project
- Test `/health` and `/verify` after every deploy

---

## Debugging Checklist

If backend returns `FUNCTION_INVOCATION_FAILED` or `500`:

1. Check deployment logs first.
2. Confirm `torch` is installed and importable in runtime.
3. Confirm `ALLOWED_ORIGINS` is set.
4. Confirm frontend points to correct backend URL.
5. Confirm `/health` is `200` before testing `/verify`.

If model quality seems low:

1. Check `inference_mode` in API response.
2. If `classical-fallback`, ensure pretrained weights are accessible.
3. Inspect `neural_score` and `classical_score` values from `/verify` response.

---

## Final Verification

Backend checks:

- `GET /health` returns `200`
- `POST /verify` returns `200` with score + verdict

Frontend checks:

- Results page shows `Inference Mode`
- Results page shows score details correctly
- Heatmap toggle works

---

Deployment status target:

- Backend (Spaces): healthy
- Frontend (Vercel): connected to backend
- End-to-end verification: successful
