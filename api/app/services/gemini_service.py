import logging
import requests

from app.core.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

def generate_recommendation(nutrisi_prop: dict, status: str, detail: str, healthy_score: float) -> str:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set. Falling back to default recommendation.")
        return "Konsultasikan dengan ahli gizi untuk saran lebih lanjut. (API Key tidak dikonfigurasi)"

    karbo = nutrisi_prop.get("karbo", 0)
    protein = nutrisi_prop.get("protein", 0)
    serat = nutrisi_prop.get("serat", 0)
    susu = nutrisi_prop.get("susu", 0)

    prompt = f"""Anda adalah seorang ahli gizi profesional yang bekerja untuk memberikan rekomendasi atau saran dari makan bergizi gratis (MBG). Berdasarkan data analisis makanan berikut:
- Proporsi Nutrisi: Karbohidrat {karbo}%, Protein {protein}%, Serat {serat}%, Susu {susu}%
- Skor Kesehatan: {healthy_score}/100
- Status Gizi: {status}
- Detail Masalah: {detail}

Berikan rekomendasi singkat dan praktis (maksimal 2-3 kalimat) untuk memperbaiki atau mempertahankan komposisi gizi ini agar lebih seimbang. Gunakan bahasa Indonesia yang ramah dan mudah dipahami. Jangan menyapa user, jangan memberi salam, langsung berikan rekomendasi."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        recommendation = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        return recommendation
    except requests.exceptions.HTTPError as e:
        error_msg = e.response.text
        logger.error(f"Failed to generate recommendation from Gemini: {e}. Response: {error_msg}")
        return f"Terjadi kesalahan saat mengambil rekomendasi dari AI (Error {e.response.status_code}). Konsultasikan dengan ahli gizi."
    except Exception as e:
        logger.error(f"Failed to generate recommendation from Gemini: {e}")
        return "Terjadi kesalahan saat mengambil rekomendasi dari AI. Konsultasikan dengan ahli gizi."
