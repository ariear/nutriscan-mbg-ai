import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.model_loader import load_saved_model
from app.routers import inference
from app.services.fuzzy_service import FuzzyNutritionClassifier
from app.services.inference_service import InferenceService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Memuat model dan fuzzy classifier...")
    app.state.model             = load_saved_model()
    app.state.fuzzy_clf         = FuzzyNutritionClassifier()
    app.state.inference_service = InferenceService(
        model     = app.state.model,
        fuzzy_clf = app.state.fuzzy_clf,
    )
    logger.info("Server siap menerima request.")
    yield
    logger.info("Server shutdown.")


app = FastAPI(
    title       = "MBG Nutrition Inference API",
    description = (
        "REST API untuk analisis gizi ompreng MBG (Makan Bergizi Gratis). "
        "Menggunakan model segmentasi U-Net + MobileNetV2 dan Fuzzy Logic "
        "untuk mendeteksi komposisi makanan dan status gizi."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router)


@app.get("/", tags=["health"])
async def root():
    return {
        "status" : "ok",
        "service": "MBG Nutrition Inference API",
        "version": "1.0.0",
        "docs"   : "/docs",
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}
