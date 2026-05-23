import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host    = "0.0.0.0",
        port    = 6767,
        reload  = True,
        workers = 1,
    )
