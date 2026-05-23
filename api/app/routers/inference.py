from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/predict", tags=["inference"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10


@router.post(
    "",
    summary="Analisis nutrisi ompreng MBG",
    description=(
        "Upload gambar ompreng (JPG/PNG/WebP). "
        "API akan melakukan segmentasi dengan U-Net, "
        "menghitung proporsi nutrisi, dan mengklasifikasikan "
        "status gizi menggunakan Fuzzy Logic."
    ),
)
async def predict(request: Request, file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Format file tidak didukung: {content_type}. "
                   f"Gunakan: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()

    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Ukuran file terlalu besar: {size_mb:.1f} MB. Maksimum {MAX_FILE_SIZE_MB} MB.",
        )

    inference_service = request.app.state.inference_service

    try:
        result = inference_service.analyze(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference gagal: {exc}") from exc

    return JSONResponse(content=result)
