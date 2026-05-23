# Nutriscan MBG AI
Bagian AI dari aplikasi Nutriscan MBG

## Panduan Setup
1. Download model di sini https://drive.google.com/file/d/1FIB96vQQdrkQxddK9bluWFlIezLeK5jS/view?usp=sharing
2. Ekstrak file model dan taruh di dalam folder api
3. Masuk ke folder api dan install library yang ada di requirements.txt
4. Copy file .env.example menjadi .env . Lalu isi GEMINI_API_KEY yang bisa didapat dari https://aistudio.google.com/api-keys
5. Jalankan python run.py atau python3 run.py

## Penggunaan API

| Endpoint | Method | Payload |
|---|---|---|
| http://localhost:6767/predict | POST | file:file |

Contoh : 
<img width="1467" height="551" alt="image" src="https://github.com/user-attachments/assets/0ec9c229-aed2-42b0-88f9-37f4cbbc1b7d" />

