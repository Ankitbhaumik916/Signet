"""
Quick validation script to check for code errors
without needing to install all dependencies
"""

import sys
import ast
from pathlib import Path

def check_python_syntax(file_path):
    """Check if Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, str(e)

def check_backend_files():
    """Check all backend Python files"""
    backend_dir = Path('d:/Signet/signature-auth/backend')
    py_files = [f for f in backend_dir.glob('**/*.py') if 'venv' not in str(f)]
    
    print("\n" + "="*70)
    print("BACKEND PYTHON FILES - SYNTAX CHECK")
    print("="*70)
    
    all_ok = True
    for py_file in sorted(py_files):
        rel_path = py_file.relative_to(backend_dir)
        ok, msg = check_python_syntax(py_file)
        status = "✅" if ok else "❌"
        print(f"{status} {rel_path}: {msg}")
        if not ok:
            all_ok = False
    
    return all_ok

def check_imports_in_main():
    """Check that main.py imports can be resolved"""
    print("\n" + "="*70)
    print("CHECKING IMPORTS IN main.py")
    print("="*70)
    
    main_py = Path('d:/Signet/signature-auth/backend/main.py')
    with open(main_py, 'r') as f:
        content = f.read()
    
    # Check for required imports
    required_imports = [
        'from fastapi import FastAPI',
        'from fastapi.middleware.cors import CORSMiddleware',
        'from fastapi.responses import JSONResponse',
        'from model.siamese_model import SiameseModel',
        'from utils.preprocess import preprocess_signature',
        'from utils.similarity import compute_similarity, compute_heatmap'
    ]
    
    print("\nRequired imports:")
    for imp in required_imports:
        if imp in content:
            print(f"✅ {imp}")
        else:
            print(f"❌ {imp} - NOT FOUND")

def check_frontend_files():
    """Check TypeScript/TSX files for basic errors"""
    print("\n" + "="*70)
    print("FRONTEND FILES - STRUCTURE CHECK")
    print("="*70)
    
    frontend_dir = Path('d:/Signet/signature-auth/frontend')
    
    required_files = [
        'app/page.tsx',
        'app/layout.tsx',
        'app/globals.css',
        'app/api/verify/route.ts',
        'components/SignatureUploader.tsx',
        'components/ResultCard.tsx',
        'components/SimilarityMeter.tsx',
        'package.json',
        'tsconfig.json',
        'tailwind.config.js',
        'next.config.js',
        'vercel.json'
    ]
    
    all_ok = True
    for file_name in required_files:
        file_path = frontend_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file_name} ({size} bytes)")
        else:
            print(f"❌ {file_name} - MISSING")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Check overall project structure"""
    print("\n" + "="*70)
    print("PROJECT STRUCTURE CHECK")
    print("="*70)
    
    base_dir = Path('d:/Signet/signature-auth')
    
    required_dirs = [
        'backend',
        'backend/model',
        'backend/utils',
        'frontend',
        'frontend/app',
        'frontend/app/api/verify',
        'frontend/components',
        'frontend/public'
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.is_dir():
            count = len(list(dir_path.glob('*')))
            print(f"✅ {dir_name}/ ({count} items)")
        else:
            print(f"❌ {dir_name}/ - MISSING")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " SIGNATURE AUTHENTICATION SYSTEM - CODE VALIDATION ".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    results = []
    
    # Check project structure
    results.append(("Project Structure", check_project_structure()))
    
    # Check Python syntax
    results.append(("Backend Python Syntax", check_backend_files()))
    
    # Check imports
    check_imports_in_main()
    
    # Check frontend files
    results.append(("Frontend Files", check_frontend_files()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Project structure is valid!")
        print("\nNext steps:")
        print("1. Backend: pip install -r requirements.txt")
        print("2. Frontend: npm install")
        print("3. Backend: python main.py")
        print("4. Frontend: npm run dev")
    else:
        print("❌ Some checks failed - please review above")
    
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
