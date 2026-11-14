"""
OLMABEK AI — Aqlli Browser versiyasi (Telegram YO'Q)

Ishga tushirish:
    pip install -r requirements.txt
    python main.py
Brauzer:
    http://localhost:10000

.env faylida bo'lishi kerak:
    CF_ACCOUNT_ID=...
    CF_API_TOKEN=...
"""

import os
import json
import base64
import asyncio
import logging
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import aiohttp
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# =========================================================
# CONFIG
# =========================================================
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN  = os.environ.get("CF_API_TOKEN", "").strip()
PORT          = int(os.environ.get("PORT", 10000))

FAST_MODEL   = "@cf/meta/llama-3.1-8b-instruct-fast"
THINK_MODEL  = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
IMAGE_MODEL  = "@cf/black-forest-labs/flux-1-schnell"
VISION_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"

# AQLLI VA TABIIY MULOQOT UCHUN YANGILANGAN PROMPT
SYSTEM_PROMPT = """
Sen — dasturchi @xamrakulovshokirbek tomonidan yaratilgan aqlli, farosatli va samimiy sun'iy intellekt — Olmabek AI san.

MULOQOT VA XULQ-ATVOR QOIDALARI:

1. **Tabiiylik va oddiylik (eng muhim):**
   Foydalanuvchi "Salom", "Assalomu alaykum", "Qalaysan?", "Qalaysiz?", "Nima gap?", "Nima gaplar?", "Yaxshimisiz?" kabi oddiy salomlashish yoki qisqa gap yozsa — hech qanday punkt, ro'yxat, uzoq tushuntirishsiz, xuddi yaqin do'stdek qisqa, samimiy va jonli javob ber.
   Misollar:
   - "Salom" → "Salom! Yaxshimisiz? Nima yordam kerak?"
   - "Qalaysan?" → "Yaxshi, rahmat! O'zingiz qalaysiz? Bugun nima qilaylik?"
   - "Nima gap?" → "Hammasi joyida, o'zingizda nima gap?"
   Oddiy salomga hech qachon chuqur tahlil, maslahat yoki uzoq matn yozma.

2. **Pashshadan fil yasama:**
   Oddiy savollardan chuqur ma'no qidirma, keraksiz narsalarni to'qib chiqarma. Savol qisqa va oddiy bo'lsa — qisqa javob. Faqat murakkab (dasturlash, tahlil, ta'lim, kod yozish) bo'lsa batafsil va tartibli javob ber.

3. **Suhbat tarixini o'qib xulosa qilish (Kontekst — juda muhim):**
   Har bir javob berishdan OLDIN yuqoridagi barcha xabarlarni (suhbat tarixini) diqqat bilan o'qi.
   - Foydalanuvchi avval nima deganini, ismini, muammosini, mavzuni eslab qol.
   - Yangi savolni alohida quruq savol deb emas, suhbatning davomi deb tushun.
   - Agar avvalgi xabarlarda biror mavzu, ism, muammo yoki vazifa bo'lsa — shuni xulosa qilib, mantiqiy davom ettir.
   - "Yana nima deb yozgan edi?", "Oldin aytgan edim-ku" degan holatlarda tarixga tayanib javob ber.
   - Har safar yangi odamdek gapirma. Suhbatni unutma.

4. **Til:**
   Faqat sof, ravon va tabiiy o'zbek tilida gaplash. Keraksiz ruscha yoki inglizcha aralashtirma.

SHAXSIYAT VA MUALLIFLIK:
- "Sen kimsan?", "Seni kim yaratgan?", "Kimning AI san?" deb so'ralsa: "Meni dasturchi @xamrakulovshokirbek yaratgan. Men uning loyihasi asosida sizga yordam berish uchun ishlab chiqilgan aqlli sun'iy intellekt — Olmabek AI man!" deb javob ber.
- Hech qachon o'zingni Meta, OpenAI, Google, Llama yoki Cloudflare mahsuloti deb tanishtirma.
- O'zingni doimo "Olmabek AI" deb bil.
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# CLOUDFLARE AI
# =========================================================
def cf_url(model):
    return f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"

def cf_headers():
    return {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }

def cf_call_sync(model, payload, timeout=60):
    async def _run():
        async with aiohttp.ClientSession() as s:
            async with s.post(cf_url(model), headers=cf_headers(), json=payload, timeout=timeout) as r:
                if r.status != 200:
                    err = await r.text()
                    raise Exception(f"CF {r.status}: {err[:300]}")
                data = await r.json()
                return (data.get("result", {}).get("response") or "").strip()
    return asyncio.run(_run())

def cf_image_sync(prompt, steps=4):
    async def _run():
        async with aiohttp.ClientSession() as s:
            async with s.post(
                cf_url(IMAGE_MODEL),
                headers=cf_headers(),
                json={"prompt": prompt, "steps": steps},
                timeout=90,
            ) as r:
                if r.status != 200:
                    err = await r.text()
                    raise Exception(f"CF {r.status}: {err[:300]}")
                data = await r.json()
                b64 = data.get("result", {}).get("image")
                if not b64:
                    raise Exception("Rasm bo'sh qaytdi")
                return base64.b64decode(b64)
    return asyncio.run(_run())

# =========================================================
# HTTP HANDLER
# =========================================================
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(f"{self.address_string()} — {fmt % args}")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status, data):
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        try:
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        except BrokenPipeError:
            pass

    def _file(self, filename, content_type):
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            return self._json(404, {"error": f"{filename} topilmadi"})
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    # ============ GET ============
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._file("index.html", "text/html; charset=utf-8")
        if path == "/style.css":
            return self._file("style.css", "text/css; charset=utf-8")
        if path == "/script.js":
            return self._file("script.js", "application/javascript; charset=utf-8")
        if path in ("/health", "/api/health"):
            return self._json(200, {"status": "ok", "service": "Olmabek AI"})
        return self._json(404, {"error": "Topilmadi"})

    # ============ OPTIONS (CORS preflight) ============
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ============ POST ============
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception as e:
            return self._json(400, {"error": f"JSON xato: {e}"})

        if path == "/api/chat":
            return self._chat(body)
        if path == "/api/image":
            return self._image(body)
        if path == "/api/vision":
            return self._vision(body)
        return self._json(404, {"error": "Endpoint topilmadi"})

    # ============ CHAT ============
    def _chat(self, body):
        message = (body.get("message") or "").strip()
        if not message:
            return self._json(400, {"error": "message bo'sh"})
        if not CF_ACCOUNT_ID or not CF_API_TOKEN:
            return self._json(500, {"error": ".env da CF_ACCOUNT_ID yoki CF_API_TOKEN yo'q"})

        mode    = body.get("mode", "fast")
        history = body.get("history") or []

        # Oddiy salomlashishlarni tez, murakkab savollarni aqlli modelga yo'naltirish
        model = THINK_MODEL if mode == "think" else FAST_MODEL

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Oxirgi 30 ta xabarni kontekst sifatida saqlash (suhbatni unutmasligi uchun)
        for h in history[-30:]:
            role    = h.get("role")
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        try:
            # temperature va repetition_penalty javoblarni tabiiy va takrorlanmas qiladi
            payload = {
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
            reply = cf_call_sync(model, payload)
            return self._json(200, {"reply": reply or "Javob bo'sh qaytdi.", "mode": mode})
        except Exception as e:
            log.exception("Chat xatosi")
            return self._json(500, {"error": str(e)[:300]})

    # ============ IMAGE ============
    def _image(self, body):
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return self._json(400, {"error": "prompt bo'sh"})
        if not CF_ACCOUNT_ID or not CF_API_TOKEN:
            return self._json(500, {"error": "CF keys yo'q"})
        try:
            en_prompt = prompt
            try:
                t_payload = {
                    "messages": [
                        {"role": "system", "content": "Translate to detailed English prompt for FLUX. Add 'photorealistic, 8k, sharp focus'. Output ONLY the prompt."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 120,
                }
                en_prompt = cf_call_sync(FAST_MODEL, t_payload, timeout=30) or prompt
            except Exception:
                pass

            img = cf_image_sync(en_prompt)
            return self._json(200, {
                "image": base64.b64encode(img).decode("ascii"),
                "prompt": prompt,
                "prompt_english": en_prompt,
            })
        except Exception as e:
            log.exception("Rasm xatosi")
            return self._json(500, {"error": str(e)[:300]})

    # ============ VISION ============
    def _vision(self, body):
        prompt = (body.get("prompt") or body.get("message") or "Bu rasmda nima bor?").strip()
        image_b64 = (body.get("image") or "").strip()

        if not image_b64:
            return self._json(400, {"error": "image (base64) yuborilmagan"})
        if not CF_ACCOUNT_ID or not CF_API_TOKEN:
            return self._json(500, {"error": "CF keys yo'q"})

        # data:image/...;base64, qismini olib tashlash
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        try:
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }
                ],
                "max_tokens": 1024,
            }
            reply = cf_call_sync(VISION_MODEL, payload, timeout=90)
            return self._json(200, {"reply": reply or "Javob bo'sh qaytdi."})
        except Exception as e:
            log.exception("Vision xatosi")
            return self._json(500, {"error": str(e)[:300]})


# =========================================================
# SERVER
# =========================================================
if __name__ == "__main__":
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        log.warning("⚠  CF_ACCOUNT_ID yoki CF_API_TOKEN .env da yo'q!")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    log.info(f"Olmabek AI ishga tushdi → http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Server to'xtatildi")
        server.server_close()