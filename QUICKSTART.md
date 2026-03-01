# Quick Start Guide

Get the Signature Authentication System up and running in 5 minutes!

## Prerequisites
- Python 3.10+ (with pip)
- Node.js 18+ (with npm)
- Git

## 1️⃣ Backend Setup (Python FastAPI + PyTorch)

### Terminal 1:
```bash
# Navigate to backend
cd signature-auth/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (takes 5-10 minutes)
pip install -r requirements.txt

# Copy environment file (optional)
cp .env.example .env

# Start the server
python main.py
```

✅ Backend running at: `http://localhost:8000`

Visit API docs: `http://localhost:8000/docs`

## 2️⃣ Frontend Setup (Next.js 14)

### Terminal 2:
```bash
# Navigate to frontend
cd signature-auth/frontend

# Install dependencies
npm install

# Copy environment file
cp .env.local.example .env.local

# Start development server
npm run dev
```

✅ Frontend running at: `http://localhost:3000`

## 3️⃣ Test the System

1. Open browser to `http://localhost:3000`
2. Upload a genuine signature image (left side)
3. Upload a test signature image (right side)
4. Click "Verify Signature"
5. View results with similarity score and heatmap

## 📊 Example Test Images

You can use any signature images:
- Use the same signature twice for "Genuine" verdict
- Use different signatures for "Forged" verdict
- Slightly modified signature for "Suspicious" verdict

## 🚀 Production Deployment

### Deploy to Vercel (Easiest)

1. **Backend:**
   ```bash
   cd backend
   npm install -g vercel
   vercel
   ```

2. **Frontend:**
   ```bash
   cd frontend
   vercel
   ```

Set `PYTHON_BACKEND_URL` environment variable to your backend URL.

## 📚 Full Documentation

See [README.md](./README.md) for:
- Detailed setup instructions
- Model training guide
- API documentation
- Troubleshooting
- Advanced deployment options

## ⚡ Useful Commands

**Backend:**
```bash
# Run tests
python -m pytest

# Train model on your data
python model/train.py --data-dir ./data --epochs 50

# Check Python environment
pip list
```

**Frontend:**
```bash
# Type checking
npm run type-check

# Linting
npm run lint

# Production build
npm run build
npm start
```

## 🆘 Common Issues

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.10+

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Try different port
python main.py --port 8001
```

### Frontend connection error
- Ensure backend is running at `http://localhost:8000`
- Check `PYTHON_BACKEND_URL` in `.env.local`
- Verify CORS settings in backend `.env`

## 📖 Next Steps

1. ✅ Get it running locally
2. 📚 Read [README.md](./README.md) for full documentation
3. 🤖 Train model on your own signature dataset
4. 🚀 Deploy to Vercel
5. 📊 Monitor and optimize

---

Happy authenticating! 🎉

For detailed information, see the main [README.md](./README.md)
