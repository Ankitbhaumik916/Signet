import sys
from pathlib import Path

# Add parent directory to Python path to import main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

# FastAPI app is exported for Vercel

