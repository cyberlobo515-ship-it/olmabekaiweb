"""
OLMABEK AI — backend (FastAPI + Cloudflare Workers AI + Google Gemini)

Arxitektura:
    • Chat, rasm, veb-sayt → Cloudflare Workers AI (bitta CF_ACCOUNT_ID + CF_API_TOKEN)
    • Kod (/api/code) va Codex (/api/codex) → Gemini

Ishga tushirish:
    pip install -r requirements.txt
    cp .env.example .env    # 3 ta kalitni yozing
    python main.py

Kerakli env:
    CF_ACCOUNT_ID, CF_API_TOKEN, GEMINI_API_KEY
"""
import base64
import json
import os
import re
import tempfile
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
import httpx  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# ============================ SOZLAMALAR ============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
CODE_MODEL = os.getenv("GEMINI_CODE_MODEL", "gemini-2.5-pro")
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "imagen-3.0-generate-002")
FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.0-flash").split(",") if m.strip()]
IMAGE_FALLBACKS = [m.strip() for m in os.getenv("GEMINI_IMAGE_FALLBACK", "gemini-2.5-flash-image").split(",") if m.strip()]

CF_ACCOUNT_ID = (os.getenv("CF_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip()
CF_API_TOKEN = (os.getenv("CF_API_TOKEN") or os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()
CF_READY = bool(CF_ACCOUNT_ID and CF_API_TOKEN)

CF_CHAT_MODEL = os.getenv("CF_CHAT_MODEL", "@cf/google/gemma-3-12b-it")
CF_CHAT_MODEL_PRO = os.getenv("CF_CHAT_MODEL_PRO", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
CF_CHAT_FALLBACKS = [m.strip() for m in os.getenv("CF_CHAT_FALLBACKS", "@cf/meta/llama-3.1-8b-instruct").split(",") if m.strip()]
CF_IMAGE_MODEL = os.getenv("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
CF_IMAGE_STEPS = {"standard": 4, "pro": 6, "ultra": 8}

IMAGE_TRANSLATE = os.getenv("IMAGE_TRANSLATE", "true").strip().lower() in ("1", "true", "yes")
FALLBACK_TO_GEMINI = os.getenv("FALLBACK_TO_GEMINI", "true").strip().lower() in ("1", "true", "yes")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()] or ["*"]
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
USAGE_FILE = Path(os.getenv("USAGE_FILE", str(BASE_DIR / "usage.json")))

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Olmabek AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================ TARIF LIMITLARI (kunlik) ============================
CHAT_LIMITS: Dict[str, Optional[int]] = {"standard": 25, "pro": 150, "ultra": None}
IMAGE_LIMITS: Dict[str, Optional[int]] = {"standard": 8, "pro": 20, "ultra": 80}
CODE_TOKEN_LIMITS: Dict[str, Optional[int]] = {"standard": 25_000, "pro": 250_000, "ultra": 2_000_000}
SITE_LIMITS: Dict[str, Optional[int]] = {"standard": 4, "pro": 25, "ultra": None}

QUOTA_MESSAGES = {
    "chat": "Bugungi xabarlar limiti tugadi. Pro yoki Ultra tarifiga o'ting.",
    "images": "Bugungi rasm limiti tugadi. Pro yoki Ultra tarifiga o'ting.",
    "tokens": "Kunlik Olmabek Code token limiti tugadi. Pro yoki Ultra tarifiga o'ting.",
    "sites": "Kunlik veb-sayt yaratish limiti tugadi. Pro yoki Ultra tarifiga o'ting.",
}
LIMITS_BY_KIND = {"chat": CHAT_LIMITS, "images": IMAGE_LIMITS, "tokens": CODE_TOKEN_LIMITS, "sites": SITE_LIMITS}

# ============================ FOYDALANISH HISOBI ============================
_lock = threading.Lock()
usage_db: Dict[str, dict] = {}


def _load_usage() -> None:
    try:
        data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            usage_db.update(data)
    except (OSError, ValueError):
        pass


def _save_usage() -> None:
    try:
        today = date.today().isoformat()
        fresh = {k: v for k, v in usage_db.items() if v.get("day") == today}
        fd, tmp = tempfile.mkstemp(dir=str(USAGE_FILE.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(fresh, f, ensure_ascii=False)
        os.replace(tmp, USAGE_FILE)
    except OSError:
        pass


_load_usage()


def _record(uid: str) -> dict:
    today = date.today().isoformat()
    rec = usage_db.get(uid)
    if not rec or rec.get("day") != today:
        rec = {"day": today, "chat": 0, "images": 0, "tokens": 0, "sites": 0}
        usage_db[uid] = rec
    return rec


def tier_of(plan: Optional[str]) -> str:
    return plan if plan in CODE_TOKEN_LIMITS else "standard"


def left_of(uid: str, plan: str, admin: bool, kind: str) -> Optional[int]:
    if admin:
        return None
    limit = LIMITS_BY_KIND[kind][tier_of(plan)]
    if limit is None:
        return None
    with _lock:
        return max(0, limit - _record(uid)[kind])


def require_quota(uid: str, plan: str, admin: bool, kind: str) -> None:
    left = left_of(uid, plan, admin, kind)
    if left is not None and left <= 0:
        raise HTTPException(status_code=429, detail=QUOTA_MESSAGES[kind])


def consume(uid: str, plan: str, admin: bool, kind: str, amount: int = 1) -> Optional[int]:
    if not admin:
        with _lock:
            _record(uid)[kind] += max(0, int(amount))
            _save_usage()
    return left_of(uid, plan, admin, kind)


# ============================ MODELLAR ============================
class ChatMessage(BaseModel):
    role: str
    content: str


class Identity(BaseModel):
    plan: Optional[str] = "standard"
    user_id: Optional[str] = "guest"
    is_admin: Optional[bool] = False


class ChatRequest(Identity):
    message: str
    history: Optional[List[ChatMessage]] = []
    mode: Optional[str] = "chat"
    deep: Optional[bool] = False
    stack: Optional[str] = "auto"
    currentCode: Optional[str] = ""
    plain_stream: Optional[bool] = False


class ImageRequest(Identity):
    prompt: str


class SiteBuildRequest(Identity):
    prompt: str
    current_code: Optional[str] = ""


class CodexRequest(Identity):
    prompt: str
    code: Optional[str] = None
    history: Optional[List[ChatMessage]] = []


def who(req: Identity) -> Tuple[str, str, bool]:
    uid = (req.user_id or "guest").strip()[:120] or "guest"
    return uid, tier_of(req.plan), bool(req.is_admin)


# ============================ GEMINI YORDAMCHILARI ============================
def require_client() -> genai.Client:
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="Serverda GEMINI_API_KEY sozlanmagan. .env fayliga kalitni yozing.",
        )
    return client


def friendly(e: Exception) -> str:
    t = str(e)
    low = t.lower()
    if "api key" in low or "api_key" in low or "permission" in low or "403" in low or "401" in low:
        return "Gemini kaliti noto'g'ri yoki ruxsat yo'q. .env dagi GEMINI_API_KEY ni tekshiring."
    if "429" in t or "quota" in low or "resource_exhausted" in low or "rate" in low:
        return "Gemini limiti vaqtincha tugadi. Birozdan keyin qayta urinib ko'ring."
    if "404" in t or "not found" in low:
        return "Model topilmadi. GET /api/models ro'yxatidan tekshiring."
    if any(k in low for k in ("name or service not known", "getaddrinfo", "connection", "timed out", "timeout", "unreachable")):
        return "Gemini serveriga ulanib bo'lmadi. Internetni tekshiring."
    if "safety" in low or "blocked" in low:
        return "So'rov xavfsizlik filtri tomonidan to'xtatildi. Boshqacha yozib ko'ring."
    return t[:300]


def build_contents(history: Optional[List[ChatMessage]], text: str, current_code: Optional[str] = None, code_limit: int = 18000):
    contents = []
    for m in (history or [])[-16:]:
        if not m.content or not m.content.strip():
            continue
        role = "model" if m.role == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m.content[:8000])]))
    final = text
    if current_code and current_code.strip():
        final = f"Hozirgi mavjud kod:\n```\n{current_code[:code_limit]}\n```\n\nFoydalanuvchi talabi:\n{text}"
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=final)]))
    return contents


def read_usage(um: Any, out_chars: int, in_chars: int) -> Tuple[int, int, int]:
    p = (getattr(um, "prompt_token_count", 0) or 0) if um else 0
    o = (getattr(um, "candidates_token_count", 0) or 0) if um else 0
    t = (getattr(um, "total_token_count", 0) or 0) if um else 0
    if not t:
        o = max(1, int(out_chars / 3.5))
        p = max(1, int(in_chars / 4))
        t = p + o
    return p, o, t


_resolved: Dict[str, str] = {}
_discovered: Dict[str, Any] = {"t": 0.0, "names": []}
_SKIP = ("image", "tts", "live", "embed", "audio", "native", "robotics", "computer", "aqa", "imagen", "veo", "learnlm")


def is_model_error(e: Exception) -> bool:
    t = str(e).lower()
    return "404" in t or "not found" in t or "not supported" in t or "is not found for api version" in t


def discover_models(ai: genai.Client) -> List[str]:
    if _discovered["names"] and time.time() - _discovered["t"] < 600:
        return _discovered["names"]
    try:
        names = []
        for m in ai.models.list():
            nm = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if nm.startswith("gemini") and (not actions or "generateContent" in actions) and not any(k in nm for k in _SKIP):
                names.append(nm)
        names.sort(key=lambda n: (0 if "flash" in n and "lite" not in n else 1, n))
        _discovered.update(t=time.time(), names=names)
    except Exception:
        pass
    return _discovered["names"]


def model_queue(primary: str) -> List[str]:
    out: List[str] = []
    for m in [_resolved.get(primary), primary] + FALLBACK_MODELS:
        if m and m not in out:
            out.append(m)
    return out


def generate_with_fallback(ai: genai.Client, primary: str, contents, system: str, temperature: float):
    queue, last, extended = model_queue(primary), None, False
    while queue:
        m = queue.pop(0)
        try:
            resp = ai.models.generate_content(model=m, contents=contents, config=types.GenerateContentConfig(system_instruction=system, temperature=temperature))
            _resolved[primary] = m
            return resp, m
        except Exception as e:
            last = e
            if not is_model_error(e):
                raise
            if not queue and not extended:
                extended = True
                queue = [n for n in discover_models(ai) if n != m][:4]
    raise last if last else RuntimeError("Model topilmadi")


def iter_gemini(ai: genai.Client, model: str, contents, system: str, temperature: float, max_tokens: Optional[int] = None) -> Iterator[Tuple[str, Any]]:
    cfg: Dict[str, Any] = {"system_instruction": system, "temperature": temperature}
    if max_tokens:
        cfg["max_output_tokens"] = max_tokens
    queue, last, extended = model_queue(model), None, False
    while queue:
        m, started = queue.pop(0), False
        try:
            for chunk in ai.models.generate_content_stream(model=m, contents=contents, config=types.GenerateContentConfig(**cfg)):
                started = True
                yield (getattr(chunk, "text", None) or ""), getattr(chunk, "usage_metadata", None)
            _resolved[model] = m
            return
        except Exception as e:
            last = e
            if started or not is_model_error(e):
                raise
            if not queue and not extended:
                extended = True
                queue = [n for n in discover_models(ai) if n != m][:4]
    raise last if last else RuntimeError("Model topilmadi")


def make_image(ai: genai.Client, prompt: str) -> bytes:
    last: Optional[Exception] = None
    for m in [IMAGE_MODEL] + IMAGE_FALLBACKS:
        try:
            if m.startswith("imagen"):
                r = ai.models.generate_images(model=m, prompt=prompt, config=types.GenerateImagesConfig(number_of_images=1, output_mime_type="image/png", aspect_ratio="1:1"))
                if r.generated_images:
                    return r.generated_images[0].image.image_bytes
            else:
                r = ai.models.generate_content(model=m, contents=prompt, config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]))
                for cand in (r.candidates or []):
                    for part in (cand.content.parts or []):
                        inline = getattr(part, "inline_data", None)
                        if inline is not None and getattr(inline, "data", None):
                            return inline.data
            last = RuntimeError("Rasm yaratilmadi. Tavsifni o'zgartirib ko'ring.")
        except Exception as e:
            last = e
    raise last if last else RuntimeError("Rasm yaratilmadi")


# ============================ CLOUDFLARE WORKERS AI ============================
class CFError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _http() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))


def cf_url(model: str) -> str:
    return f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"


def cf_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}


def cf_error_text(status: int, raw: str) -> str:
    msg = ""
    try:
        msg = ((json.loads(raw).get("errors") or [{}])[0].get("message")) or ""
    except (ValueError, AttributeError, IndexError):
        msg = (raw or "")[:200]
    low = msg.lower()
    if status in (401, 403) or "authentication" in low:
        return "Cloudflare tokeni noto'g'ri yoki Workers AI ruxsati yo'q. .env dagi CF_API_TOKEN ni tekshiring."
    if "neurons" in low or "daily free allocation" in low or status == 429:
        return "Cloudflare kunlik bepul limiti tugadi."
    if status == 404 or "no such model" in low or "not found" in low:
        return "Cloudflare modeli topilmadi. CF_CHAT_MODEL / CF_IMAGE_MODEL ni tekshiring."
    return f"Cloudflare xatosi ({status}): {msg[:200]}"


def cf_text(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    if isinstance(obj.get("response"), str):
        return obj["response"]
    ch = obj.get("choices")
    if isinstance(ch, list) and ch:
        part = ch[0].get("delta") or ch[0].get("message") or {}
        c = part.get("content")
        return c if isinstance(c, str) else ""
    if "result" in obj:
        return cf_text(obj["result"])
    return ""


def cf_messages(system: str, history: Optional[List[ChatMessage]], message: str) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": system}]
    for m in (history or [])[-16:]:
        if m.content and m.content.strip():
            msgs.append({"role": "assistant" if m.role == "assistant" else "user", "content": m.content[:8000]})
    msgs.append({"role": "user", "content": message})
    return msgs


def cf_chat_models(plan: str) -> List[str]:
    out: List[str] = []
    for m in [CF_CHAT_MODEL_PRO if plan in ("pro", "ultra") else CF_CHAT_MODEL, CF_CHAT_MODEL] + CF_CHAT_FALLBACKS:
        if m and m not in out:
            out.append(m)
    return out


def cf_stream(model: str, messages: List[Dict[str, str]], max_tokens: int) -> Iterator[str]:
    body = {"messages": messages, "stream": True, "max_tokens": max_tokens, "temperature": 0.7}
    with _http() as c, c.stream("POST", cf_url(model), headers=cf_headers(), json=body) as r:
        if r.status_code != 200:
            r.read()
            raise CFError(r.status_code, cf_error_text(r.status_code, r.text))
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                t = cf_text(json.loads(data))
            except ValueError:
                continue
            if t:
                yield t


def cf_stream_with_fallback(plan: str, messages: List[Dict[str, str]]) -> Iterator[str]:
    max_tokens = 4096 if plan in ("pro", "ultra") else 2048
    last: Optional[Exception] = None
    for model in cf_chat_models(plan):
        started = False
        try:
            for t in cf_stream(model, messages, max_tokens):
                started = True
                yield t
            return
        except CFError as e:
            last = e
            if started or e.status not in (400, 404, 500, 502, 503):
                raise
    raise last if last else CFError(500, "Cloudflare modeli topilmadi")


def cf_complete(system: str, user: str, max_tokens: int = 220, timeout: float = 15.0, model: Optional[str] = None) -> str:
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": 0.2}
    with httpx.Client(timeout=timeout) as c:
        r = c.post(cf_url(model or CF_CHAT_MODEL), headers=cf_headers(), json=body)
    if r.status_code != 200:
        raise CFError(r.status_code, cf_error_text(r.status_code, r.text))
    return cf_text(r.json()).strip()


def cf_to_english(prompt: str) -> str:
    try:
        out = cf_complete(
            "You turn an image description (any language) into one vivid, concise English prompt for a text-to-image model. "
            "Keep every style word. Output ONLY the English prompt, no quotes, no explanations.",
            prompt[:1500],
        )
        return out.strip('"\' ')[:1800] or prompt
    except Exception:
        return prompt


def cf_image_b64(prompt: str, steps: int) -> str:
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as c:
        r = c.post(cf_url(CF_IMAGE_MODEL), headers=cf_headers(), json={"prompt": prompt[:1800], "steps": steps})
    if r.status_code != 200:
        raise CFError(r.status_code, cf_error_text(r.status_code, r.text))
    if (r.headers.get("content-type") or "").startswith("image/"):
        return base64.b64encode(r.content).decode("utf-8")
    try:
        j = r.json()
        res = j.get("result") if isinstance(j, dict) and isinstance(j.get("result"), dict) else j
        img = res.get("image") if isinstance(res, dict) else None
    except ValueError:
        img = None
    if not img:
        raise CFError(502, "Cloudflare rasm qaytarmadi. Tavsifni o'zgartirib ko'ring.")
    return img


def chat_uses_cf() -> bool:
    return CF_READY


def image_uses_cf() -> bool:
    return CF_READY


def chat_model_label(plan: str) -> str:
    return cf_chat_models(plan)[0] if chat_uses_cf() else select_model(plan, False)


def error_text(e: Exception) -> str:
    return e.message if isinstance(e, CFError) else friendly(e)


def chat_text_stream(plan: str, deep: bool, history: Optional[List[ChatMessage]], message: str) -> Iterator[str]:
    system = chat_system_instruction(plan, deep)
    if chat_uses_cf():
        started = False
        try:
            for t in cf_stream_with_fallback(plan, cf_messages(system, history, message)):
                started = True
                yield t
            return
        except Exception as e:
            if started or not (FALLBACK_TO_GEMINI and client is not None):
                raise
            print(f"[chat] Cloudflare ishlamadi ({error_text(e)}) -> Gemini zaxirasi")
    ai = require_client()
    for t, _ in iter_gemini(ai, select_model(plan, False), build_contents(history, message), system, 0.7):
        if t:
            yield t


def sse_line(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def clean_html_output(raw_text: str) -> str:
    text = (raw_text or "").strip()
    fence = re.search(r"```(?:html|htm)?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    doc = re.search(r"(<!DOCTYPE\s+html[\s\S]*<\/html>)", text, re.IGNORECASE)
    if doc:
        return doc.group(1).strip()
    if text.startswith("```html"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ============================ SYSTEM KO'RSATMALAR ============================
def code_system_instruction(plan: str, stack: str) -> str:
    if stack == "react":
        guide = "Foydalanuvchi React komponentini xohlaydi. To'liq yozing (default export App, Tailwind CSS, inline SVG yoki toza CSS)."
    elif stack == "python":
        guide = "Foydalanuvchi Python kodini xohlaydi. Toza, tipizatsiyalangan, izohli va ishga tayyor ```python``` kodi yozing."
    else:
        guide = ("Veb-ilova, sahifa, o'yin yoki UI so'ralganda DOIM bitta to'liq, mustaqil ```html``` fayli "
                 "(<!DOCTYPE html>, ichki <style> va <script>) yozing. Shunda Preview oynasida darhol ishlaydi.")
    depth = {
        "ultra": "Siz ULTRA darajadagi Senior Principal Architect rejimidasiz: maksimal sifat, mukammal responsive layout, mikro-animatsiyalar va xatosiz arxitektura.",
        "pro": "Siz PRO darajadagi Senior Full-Stack dasturchisiz: to'liq funksional, zamonaviy va chiroyli kod yozing.",
    }.get(plan, "Siz Standart yordamchisiz: aniq, ixcham va to'g'ri ishlaydigan kod yozing.")
    return "\n\n".join([
        "Siz OLMABEK CODE — OLMABEK AI platformasining dasturlash muharririsiz.",
        depth,
        guide,
        "Javobni o'zbek tilida 1-2 qisqa gap bilan boshlang, so'ng to'liq kodni bitta fenced blokda bering.",
        "Kodni hech qachon '...' deb qisqartirmang.",
    ])


def chat_system_instruction(plan: str, deep: bool) -> str:
    if plan == "ultra" and deep:
        depth = "Murakkab savollarni chuqur, bosqichma-bosqich, misollar va aniq xulosalar bilan tahlil qilib bering."
    elif plan == "pro":
        depth = "Aniq, tezkor, tizimli va foydali javoblar bering."
    else:
        depth = "Lo'nda, tushunarli va samimiy o'zbek tilida javob bering."
    return (f"Siz OLMABEK AI — o'zbek tilida mukammal fikrlaydigan sun'iy intellekt yordamchisiz. {depth} "
            "Zarur bo'lganda markdown sarlavhalar, ro'yxatlar va kod bloklaridan foydalaning.")


def codex_system_instruction(plan: str) -> str:
    depth = ("Maksimal sifat: mukammal responsive, silliq animatsiyalar, boy interaktiv holatlar." if plan == "ultra"
             else "Zamonaviy, toza va to'liq ishlaydigan kod yozing.")
    return "\n\n".join([
        "Sen Olmabek Codex — dasturlash agentisan. Foydalanuvchi o'zbek tilida yozadi.",
        "Javob formati QAT'IY: 1) 1-3 gapli qisqa reja (o'zbek tilida); 2) BITTA ```html kod bloki — to'liq, mustaqil, bitta fayl "
        "(<!DOCTYPE html>, CSS <style> ichida, JS <script> ichida); 3) blokdan keyin 1-2 gapli izoh (o'zbek tilida).",
        "Tashqi kutubxona kerak bo'lsa faqat https://cdnjs.cloudflare.com yoki https://cdn.jsdelivr.net dan ulang.",
        "Sahifa himoyalangan oynada ishlaydi: localStorage/cookie ishlatmang (kerak bo'lsa try/catch bilan). Rasm uchun CSS gradient yoki inline SVG ishlating.",
        "Mavjud kod berilgan bo'lsa, uni o'zgartirib TO'LIQ yangilangan faylni qaytaring (qisman emas). Kodda hech qachon API kalit yozmang.",
        depth,
        "Matnlar o'zbek tilida bo'lsin. Kodni '...' bilan qisqartirmang.",
    ])


def build_site_system_instruction(plan: str, admin: bool) -> str:
    premium = plan in ("pro", "ultra") or admin
    tier = ("Zamonaviy, animatsiyali, mukammal responsive dizayn, interaktiv JavaScript va boy kontent bilan yarat."
            if premium else "Toza, tushunarli, chiroyli va to'liq ishlaydigan oddiy veb-sahifa (HTML + CSS + JS) yarat.")
    return (
        "Sen professional Frontend dasturchi va UI/UX dizaynersan.\n"
        "Vazifang: foydalanuvchi so'roviga ko'ra tayyor, yagona faylli, to'liq ishlaydigan veb-sayt kodini yozish.\n"
        f"1. {tier}\n"
        "2. Barcha CSS va JS bitta <!DOCTYPE html> hujjati ichida bo'lsin (iframe Preview uchun).\n"
        "3. Hech qanday tushuntirish yoki markdown yozma! Faqat toza HTML kodning o'zini qaytar.\n"
        "4. Matnlar asosan o'zbek tilida bo'lsin."
    )


def select_model(plan: str, is_code: bool) -> str:
    return CODE_MODEL if (is_code and plan in ("pro", "ultra")) or plan == "ultra" else CHAT_MODEL


# ============================ XATOLAR ============================
@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: HTTPException):
    msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    body = {"error": msg, "detail": msg}
    if exc.status_code == 404 and msg == "Not Found":
        msg = f"Endpoint topilmadi: {request.method} {request.url.path}"
        body = {"error": msg, "detail": msg, "hint": "Mavjud yo'llar: /api/chat, /api/chat/stream, /api/image, /api/code, /api/code/stream, /api/codex, /api/build-site, /api/limits, /api/models, /api/health"}
    return JSONResponse(body, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    msg = "So'rov noto'g'ri yuborildi (maydonlar to'liq emas)."
    return JSONResponse({"error": msg, "detail": str(exc.errors())[:300]}, status_code=422)


@app.exception_handler(Exception)
async def server_error(_: Request, exc: Exception):
    msg = friendly(exc)
    return JSONResponse({"error": msg, "detail": msg}, status_code=500)


# ============================ FRONTEND ============================
def _static(name: str):
    path = BASE_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Fayl topilmadi")
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


def health_payload() -> dict:
    return {
        "ok": True, "status": "ok", "service": "Olmabek AI Server",
        "geminiConfigured": bool(GEMINI_API_KEY),
        "cloudflareConfigured": CF_READY,
        "chatProvider": "cloudflare" if chat_uses_cf() else "gemini",
        "imageProvider": "cloudflare" if image_uses_cf() else "gemini",
        "siteProvider": "cloudflare" if CF_READY else "gemini",
        "codeProvider": "gemini",
        "cfChatModel": CF_CHAT_MODEL if CF_READY else None,
        "cfImageModel": CF_IMAGE_MODEL if CF_READY else None,
        "geminiChatModel": CHAT_MODEL,
        "geminiCodeModel": CODE_MODEL,
    }


@app.get("/")
def root():
    return _static("index.html") if (BASE_DIR / "index.html").is_file() else health_payload()


@app.get("/index.html")
def index_file():
    return _static("index.html")


@app.get("/style.css")
def style_file():
    return _static("style.css")


@app.get("/script.js")
def script_file():
    return _static("script.js")


# ============================ SERVIS ENDPOINTLARI ============================
@app.get("/api/health")
def health_check():
    return health_payload()


@app.get("/api/models")
def list_models():
    ai = require_client()
    try:
        names = []
        for m in ai.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions or "generateImages" in actions or "predict" in actions:
                names.append((m.name or "").replace("models/", ""))
        return {"models": sorted(set(n for n in names if n))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=friendly(e))


@app.get("/api/limits")
def get_limits(user_id: str = "guest", plan: str = "standard", is_admin: bool = False):
    tier = tier_of(plan)
    uid = (user_id or "guest")[:120]
    return {
        "plan": tier,
        "chatLeft": left_of(uid, tier, is_admin, "chat"),
        "imagesLeft": left_of(uid, tier, is_admin, "images"),
        "codeTokensLeft": left_of(uid, tier, is_admin, "tokens"),
        "codeTokensLimit": CODE_TOKEN_LIMITS[tier],
        "sitesLeft": left_of(uid, tier, is_admin, "sites"),
        "sitesLimit": SITE_LIMITS[tier],
    }


# ============================ 1. CHAT (Cloudflare) va CODE (Gemini) — JSON ============================
@app.post("/api/chat")
@app.post("/api/code")
def generate_sync(req: ChatRequest):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Xabar matni bo'sh")
    uid, plan, admin = who(req)
    is_code = req.mode == "code"
    require_quota(uid, plan, admin, "tokens" if is_code else "chat")

    if not is_code:
        try:
            reply = "".join(chat_text_stream(plan, bool(req.deep), req.history, message)).strip() or "Javob bo'sh qaytdi."
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=error_text(e))
        left = consume(uid, plan, admin, "chat", 1)
        return {"reply": reply, "usage": {"tokensLeft": left, "model": chat_model_label(plan)}}

    ai = require_client()
    system = code_system_instruction(plan, req.stack or "auto")
    contents = build_contents(req.history, message, req.currentCode)
    try:
        resp, used_model = generate_with_fallback(ai, select_model(plan, True), contents, system, 0.5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=friendly(e))
    reply = resp.text or "Javob bo'sh qaytdi."
    p, o, t = read_usage(getattr(resp, "usage_metadata", None), len(reply), len(message))
    left = consume(uid, plan, admin, "tokens", t)
    return {"reply": reply, "usage": {"promptTokens": p, "outputTokens": o, "totalTokens": t, "tokensLeft": left, "model": used_model}}


# ============================ 2. OQIMLI CHAT va CODE ============================
def stream_handler(req: ChatRequest, force_code: bool = False):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Xabar matni bo'sh")
    uid, plan, admin = who(req)
    is_code = force_code or req.mode == "code"
    require_quota(uid, plan, admin, "tokens" if is_code else "chat")
    plain = bool(req.plain_stream)
    media = "text/plain; charset=utf-8" if plain else "text/event-stream; charset=utf-8"
    headers = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}

    if not is_code:
        def chat_events():
            try:
                for text in chat_text_stream(plan, bool(req.deep), req.history, message):
                    yield text if plain else sse_line({"type": "chunk", "text": text})
                left = consume(uid, plan, admin, "chat", 1)
                if not plain:
                    yield sse_line({"type": "usage", "tokensLeft": left, "model": chat_model_label(plan)})
                    yield sse_line({"type": "done"})
            except Exception as e:
                msg = error_text(e)
                yield f"\n⚠️ Xatolik: {msg}" if plain else sse_line({"type": "error", "error": msg})
        return StreamingResponse(chat_events(), media_type=media, headers=headers)

    ai = require_client()
    system = code_system_instruction(plan, req.stack or "auto")
    contents = build_contents(req.history, message, req.currentCode)
    primary_model = select_model(plan, True)

    def code_events():
        chars, um_last = 0, None
        try:
            for text, um in iter_gemini(ai, primary_model, contents, system, 0.5):
                if um:
                    um_last = um
                if text:
                    chars += len(text)
                    yield text if plain else sse_line({"type": "chunk", "text": text})
            p, o, t = read_usage(um_last, chars, len(message))
            left = consume(uid, plan, admin, "tokens", t)
            if not plain:
                yield sse_line({"type": "usage", "promptTokens": p, "outputTokens": o, "totalTokens": t, "tokensLeft": left, "model": primary_model})
                yield sse_line({"type": "done"})
        except Exception as e:
            yield f"\n⚠️ Xatolik: {friendly(e)}" if plain else sse_line({"type": "error", "error": friendly(e)})

    return StreamingResponse(code_events(), media_type=media, headers=headers)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    return stream_handler(req, force_code=False)


@app.post("/api/code/stream")
def code_stream(req: ChatRequest):
    return stream_handler(req, force_code=True)


# ============================ 3. CODEX (faqat Ultra, Gemini) ============================
@app.get("/api/codex")
def codex_status():
    return {"ok": True, "model": CODE_MODEL, "key_set": bool(GEMINI_API_KEY)}


@app.post("/api/codex")
def codex(req: CodexRequest):
    ai = require_client()
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Topshiriq bo'sh")
    uid, plan, admin = who(req)
    if not admin and plan != "ultra":
        raise HTTPException(status_code=403, detail="Olmabek Codex faqat Ultra tarifida ishlaydi.")
    require_quota(uid, plan, admin, "tokens")

    contents = build_contents(req.history, prompt[:8000], req.code, code_limit=60000)
    system = codex_system_instruction(plan)

    def events():
        chars, um_last = 0, None
        try:
            for text, um in iter_gemini(ai, CODE_MODEL, contents, system, 0.6, max_tokens=32768):
                if um:
                    um_last = um
                if text:
                    chars += len(text)
                    yield text
            _, _, t = read_usage(um_last, chars, len(prompt) + len(req.code or ""))
            consume(uid, plan, admin, "tokens", t)
        except Exception as e:
            yield f"XATO: {friendly(e)}"

    return StreamingResponse(events(), media_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


# ============================ 4. RASM (Cloudflare Flux, zaxira Gemini) ============================
@app.post("/api/image")
def image(req: ImageRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Rasm tavsifi bo'sh")
    uid, plan, admin = who(req)
    require_quota(uid, plan, admin, "images")

    b64: Optional[str] = None
    if image_uses_cf():
        try:
            b64 = cf_image_b64(cf_to_english(prompt) if IMAGE_TRANSLATE else prompt, CF_IMAGE_STEPS[plan])
        except Exception as e:
            if not (FALLBACK_TO_GEMINI and client is not None):
                raise HTTPException(status_code=500, detail=error_text(e))
            print(f"[image] Cloudflare ishlamadi ({error_text(e)}) -> Gemini zaxirasi")
    if b64 is None:
        ai = require_client()
        try:
            b64 = base64.b64encode(make_image(ai, prompt[:1500])).decode("utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=friendly(e))
    left = consume(uid, plan, admin, "images", 1)
    return {"image": b64, "imagesLeft": left}


# ============================ 5. VEB-SAYT YARATISH (Cloudflare, zaxira Gemini) ============================
def cf_generate_site(system: str, user: str, plan: str) -> str:
    """Cloudflare orqali HTML sayt generatsiyasi (non-stream)."""
    model = CF_CHAT_MODEL_PRO if plan in ("pro", "ultra") else CF_CHAT_MODEL
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 8192 if plan in ("pro", "ultra") else 6144,
        "temperature": 0.6,
    }
    last: Optional[Exception] = None
    for m in [model, CF_CHAT_MODEL] + CF_CHAT_FALLBACKS:
        try:
            with httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0)) as c:
                r = c.post(cf_url(m), headers=cf_headers(), json=body)
            if r.status_code != 200:
                raise CFError(r.status_code, cf_error_text(r.status_code, r.text))
            text = cf_text(r.json()).strip()
            if text:
                return text
        except Exception as e:
            last = e
    raise last if last else CFError(500, "Cloudflare sayt yaratmadi")


@app.post("/api/build-site")
def build_site(req: SiteBuildRequest):
    uid, plan, admin = who(req)
    require_quota(uid, plan, admin, "sites")
    system = build_site_system_instruction(plan, admin)

    if req.current_code and req.current_code.strip():
        user = (f"Mavjud veb-sayt kodini talabga moslab tahrirla va yangilangan TO'LIQ kodni qaytar.\n\n"
                f"TALAB: {req.prompt}\n\nMAVJUD KOD:\n{req.current_code}")
    else:
        user = f"Quyidagi talab asosida yangi veb-sayt yarat:\n{req.prompt}"

    html: Optional[str] = None
    used_model = ""
    tokens_used = 0

    # 1) Cloudflare (asosiy)
    if CF_READY:
        try:
            raw = cf_generate_site(system, user, plan)
            html = clean_html_output(raw)
            used_model = "cloudflare"
            tokens_used = max(1, int((len(user) + len(html)) / 4))
        except Exception as e:
            if not (FALLBACK_TO_GEMINI and client is not None):
                raise HTTPException(status_code=500, detail=error_text(e))
            print(f"[build-site] Cloudflare ishlamadi ({error_text(e)}) -> Gemini zaxirasi")

    # 2) Gemini zaxirasi
    if html is None:
        ai = require_client()
        primary = CODE_MODEL if plan == "ultra" else CHAT_MODEL
        try:
            resp, used_model = generate_with_fallback(ai, primary, user, system, 0.6)
        except Exception as e:
            raise HTTPException(status_code=500, detail=friendly(e))
        html = clean_html_output(resp.text or "")
        _, _, tokens_used = read_usage(getattr(resp, "usage_metadata", None), len(html), len(user))

    if not html:
        raise HTTPException(status_code=500, detail="Kod yaratilmadi")

    consume(uid, plan, admin, "tokens", tokens_used)
    sites_left = consume(uid, plan, admin, "sites", 1)
    return {
        "ready": True,
        "html": html,
        "sitesLeft": sites_left,
        "tokensUsed": tokens_used,
        "tokensLeft": left_of(uid, plan, admin, "tokens"),
        "model": used_model,
    }


if __name__ == "__main__":
    import uvicorn

    if not GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY topilmadi — codex ishlamaydi, faqat chat/rasm/sayt ishlaydi.")
    if not CF_READY:
        print("⚠️  CF_ACCOUNT_ID yoki CF_API_TOKEN topilmadi — chat/rasm/sayt Gemini zaxirasida ishlaydi.")

    print(f"▶ Olmabek AI server: http://{HOST}:{PORT}")
    print(f"   Chat:  {'Cloudflare' if CF_READY else 'Gemini'}")
    print(f"   Rasm:  {'Cloudflare' if CF_READY else 'Gemini'}")
    print(f"   Sayt:  {'Cloudflare' if CF_READY else 'Gemini'}")
    print(f"   Kod/Codex: Gemini ({CODE_MODEL})")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)