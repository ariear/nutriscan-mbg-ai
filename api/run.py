"""
Script untuk menjalankan server FastAPI dengan uvicorn.
Jalankan dari direktori api/:
    python run.py
atau langsung:
    uvicorn app.main:app --reload --port 8000
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host    = "0.0.0.0",
        port    = 6767,
        reload  = True,
        workers = 1,  # 1 worker agar model hanya di-load sekali
    )
