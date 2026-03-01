# Deployment Guide - Signature Authentication System

This guide covers deploying the Signature Authentication System to Vercel (both backend and frontend).

## Prerequisites

- GitHub account with this repository
- Vercel account (https://vercel.com)
- Personal access token from GitHub (for pulling during builds)

---

## Part 1: Backend Deployment (Python FastAPI)

### Option A: Deploy to Vercel (Recommended)

1. **Connect GitHub Repository to Vercel**
   - Go to https://vercel.com/new
   - Import the repository `Ankitbhaumik916/Signet`
   - Select the **`backend`** directory as the root

2. **Configure Build Settings**
   - Build Command: `pip install -r requirements.txt` (automatic for Python)
   - Output Directory: (leave default)
   - Framework: None (Python)

3. **Set Environment Variables**
   - Go to Settings → Environment Variables
   - Add:
     ```
     ALLOWED_ORIGINS=*,https://your-frontend-domain.vercel.app
     MODEL_CACHE_DIR=/tmp/signature_model_weights
     ```

4. **Deploy**
   - Click "Deploy"
   - After ~2-3 minutes, you'll get a live backend URL (e.g., `https://my-backend.vercel.app`)
   - **Save this URL** for frontend configuration

### Option B: Deploy to Heroku/Railway/Render

If you prefer other platforms:
- All use standard Python deployment with `requirements.txt`
- Ensure `uvicorn` is included (it is, in `requirements.txt`)
- Set the start command to: `python main.py`

---

## Part 2: Frontend Deployment (Next.js 14)

### Deploy to Vercel (Recommended)

1. **Connect GitHub Repository to Vercel**
   - Go to https://vercel.com/new
   - Import the **same** repository
   - Select the **`frontend`** directory as the root

2. **Configure Build Settings**
   - Build Command: `npm run build` (automatic for Next.js)
   - Output Directory: `.next` (automatic)
   - Framework: Next.js

3. **Set Environment Variables**
   - Go to Settings → Environment Variables
   - Add:
     ```
     PYTHON_BACKEND_URL=https://your-backend-url.vercel.app
     ```
   - Replace `your-backend-url.vercel.app` with the actual backend URL from Part 1

4. **Deploy**
   - Click "Deploy"
   - After build completes (~1-2 minutes), you'll get a live frontend URL

---

## Part 3: Verify Deployment

### Test Backend Health
```bash
curl https://your-backend-url.vercel.app/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "Signature Authentication API",
  "cuda_available": false
}
```

### Test Frontend
- Open `https://your-frontend-domain.vercel.app`
- Upload two signature images
- Click "Verify Signature"
- Confirm results display correctly

### Test API Integration
- Use the frontend UI to verify signatures
- Check browser DevTools Network tab to confirm requests go to your backend

---

## Troubleshooting

### Backend: Model Loading Timeout
- **Issue**: Deployment takes >10 minutes or times out
- **Solution**: Model weights are lazy-loaded on first request. First `/verify` call may take 30-60 seconds to download weights (~500MB)

### Frontend: "Failed to connect to verification service"
- **Issue**: Frontend can't reach backend
- **Check**:
  1. `PYTHON_BACKEND_URL` env var is set correctly in Vercel
  2. Backend is deployed and healthy (check health endpoint)
  3. CORS is enabled on backend (it is by default with `ALLOWED_ORIGINS=*`)

### Backend: PyTorch Dependency Issues
- **Issue**: Build fails with torch wheel errors
- **Solutions**:
  1. Increase Lambda memory limit to 1GB in `vercel.json`
  2. Use `pytorch` instead of `torch` if wheels unavailable
  3. Deploy to Railway/Render instead (better for Python)

---

## Environment Variables Checklist

### Backend (`backend/vercel.json`)
- ✅ `ALLOWED_ORIGINS` - Set to frontend domain
- ✅ `MODEL_CACHE_DIR` - Default `/tmp/signature_model_weights` is fine

### Frontend (`frontend/vercel.json`)
- ✅ `PYTHON_BACKEND_URL` - Set to deployed backend URL

---

## Monitoring & Logs

### View Backend Logs
1. Go to Vercel Dashboard
2. Select backend project
3. Click "Deployments"
4. Select latest deployment
5. Click "Logs"

### View Frontend Logs
1. Same steps as backend
2. Frontend logs show Next.js build and runtime errors

---

## Redeployment

To redeploy after making changes:

1. **Commit & Push to GitHub**
   ```bash
   cd d:\Signet\signature-auth
   git add .
   git commit -m "Your changes"
   git push origin main
   ```

2. **Vercel Auto-Redeploys**
   - Both frontend and backend automatically redeploy on `git push`
   - Watch deployment progress in Vercel Dashboard

---

## Performance Optimization

### Backend
- CPU inference takes ~1-2 seconds per verification
- First request to a fresh instance takes 30-60s (model download)
- Consider using GPU instances for faster inference (paid Vercel feature)

### Frontend
- Build size: ~143 KB (excellent)
- Deploy to Vercel edge network for global CDN
- All static pages pre-rendered

---

## Security Notes

1. **CORS Configuration**
   - Change `ALLOWED_ORIGINS=*` to specific domains in production
   - Example: `ALLOWED_ORIGINS=https://myapp.vercel.app`

2. **API Authentication**
   - Currently no auth required
   - For production, add API key validation in `backend/main.py`

3. **Model Weights**
   - Weights are downloaded to `/tmp` on first request
   - Vercel serverless instances are ephemeral
   - Next instance will download again (this is by design)

---

## Local Development

To run locally before deployment:

### Backend
```bash
cd backend
python main.py
# Runs on http://localhost:8000
```

### Frontend
```bash
cd frontend
npm run dev
# Runs on http://localhost:3000
# Set PYTHON_BACKEND_URL=http://localhost:8000 in .env.local
```

---

## Support & Documentation

- **README.md** - Complete project overview
- **QUICKSTART.md** - 5-minute quick start
- **Backend API Docs** - Available at `https://your-backend-url.vercel.app/docs` (Swagger UI)

---

**Deployment Status**: ✅ Ready for Production
**Last Updated**: March 1, 2026
