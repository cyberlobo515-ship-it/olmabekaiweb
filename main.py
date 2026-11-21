import os
import re
import json
import base64
from datetime import date
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

# ===== SOZLAMALAR =====
# Render Environment Variables bo'limiga GEMINI_API_KEY ni qo'shing
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-pro")
CODE_PRO_MODEL = os.environ.get("GEMINI_CODE_MODEL", "gemini-2.5-pro")
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "imagen-3.0-generate-002")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Olmabek AI Backend")

# ✅ CORS TUZATILDI: allow_credentials=False (credentials yubormaymiz)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ 404 DIAGNOSTIKA: Agar yana 404 chiqsa, qaysi URL topilmaganini aniq ko'rsatadi
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": f"Endpoint topilmadi: {request.method} {request.url.path}",
            "hint": "Mavjud yo'llar: /api/chat, /api/code, /api/codex, /api/code/stream, /api/codex/stream, /api/build-site, /api/codex/build",
        },
    )


# ===== TARIF LIMITLARI =====
CODE_TOKEN_LIMITS: Dict[str, int] = {
    "standard": 25_000,
    "pro": 250_000,
    "ultra": 2_000_000,
}

SITE_LIMITS: Dict[str, int] = {
    "standard": 4,
    "pro": 25,
    "ultra": 999_999,
}

usage_db: Dict[str, dict] = {}


# ===== MA'LUMOT MODELLARI =====
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    mode: Optional[str] = "chat"
    plan: Optional[str] = "standard"
    deep: Optional[bool] = False
    stack: Optional[str] = "auto"
    currentCode: Optional[str] = ""
    user_id: Optional[str] = "guest"
    is_admin: Optional[bool] = False
    plain_stream: Optional[bool] = False


class ImageRequest(BaseModel):
    prompt: str


class SiteBuildRequest(BaseModel):
    prompt: str
    current_code: Optional[str] = ""
    plan: Optional[str] = "standard"
    user_id: Optional[str] = "guest"
    is_admin: Optional[bool] = False


# ===== YORDAMCHI FUNKSIYALAR =====
def get_today() -> str:
    return date.today().isoformat()


def get_or_reset_record(user_id: str) -> dict:
    today_str = get_today()
    record = usage_db.get(user_id)
    if not record or record.get("day") != today_str:
        record = {"day": today_str, "tokens_used": 0, "sites_used": 0}
        usage_db[user_id] = record
    return record


def get_code_tokens_left(user_id: str, plan: str, is_admin: bool) -> int:
    if is_admin:
        return CODE_TOKEN_LIMITS["ultra"]
    tier = plan if plan in CODE_TOKEN_LIMITS else "standard"
    limit = CODE_TOKEN_LIMITS[tier]
    record = get_or_reset_record(user_id)
    return max(0, limit - record["tokens_used"])


def consume_code_tokens(user_id: str, plan: str, is_admin: bool, tokens: int) -> int:
    if is_admin:
        return CODE_TOKEN_LIMITS["ultra"]
    tier = plan if plan in CODE_TOKEN_LIMITS else "standard"
    limit = CODE_TOKEN_LIMITS[tier]
    record = get_or_reset_record(user_id)
    record["tokens_used"] += max(0, tokens)
    return max(0, limit - record["tokens_used"])


def check_and_increment_site_limit(user_id: str, plan: str, is_admin: bool) -> dict:
    if is_admin or plan == "ultra":
        return {"allowed": True, "left": 999_999}
    tier = plan if plan in SITE_LIMITS else "standard"
    max_limit = SITE_LIMITS[tier]
    record = get_or_reset_record(user_id)
    if record["sites_used"] >= max_limit:
        return {"allowed": False, "left": 0, "limit": max_limit}
    record["sites_used"] += 1
    return {"allowed": True, "left": max_limit - record["sites_used"], "limit": max_limit}


def build_gemini_contents(history: List[ChatMessage], new_message: str, current_code: Optional[str] = None):
    contents = []
    for m in (history or [])[-16:]:
        if not m.content or not m.content.strip():
            continue
        role = "model" if m.role == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m.content)]))
    final_prompt = new_message
    if current_code and current_code.strip():
        final_prompt = (
            f"Hozirgi mavjud kod konteksti:\n```\n{current_code[:18000]}\n```\n\n"
            f"Foydalanuvchi talabi:\n{new_message}"
        )
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=final_prompt)]))
    return contents


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


def get_code_system_instruction(plan: str, stack: str) -> str:
    if stack == "react":
        stack_guidance = (
            "Foydalanuvchi React JSX/TSX komponentini xohlaydi. Komponentni to'liq yozing "
            "(default export App bo'lsin, Tailwind CSS ishlatilsin, inline SVG yoki toza CSS ishlating)."
        )
    elif stack == "python":
        stack_guidance = (
            "Foydalanuvchi Python kodini xohlaydi. Toza, tipizatsiyalangan, izohli va "
            "ishga tushirishga tayyor ```python``` kodi yozing."
        )
    else:
        stack_guidance = (
            "Veb-ilova, sahifa, o'yin yoki UI so'ralganda DOIM bitta to'liq, mustaqil ishlaydigan "
            "```html``` fayli (`<!DOCTYPE html>...`) ko'rinishida yozing (Tailwind CDN, ichki "
            "`<style>` va `<script>` bilan). Shunda Preview oynasida darhol mukammal ishlaydi!"
        )
    if plan == "ultra":
        tier_depth = (
            "Siz ULTRA darajadagi Senior Principal Architect rejimidasiz: maksimal sifat, "
            "mukammal responsive layout, mikro-animatsiyalar, boy interaktiv holatlar va xatosiz arxitektura yarating."
        )
    elif plan == "pro":
        tier_depth = (
            "Siz PRO darajadagi Senior Full-Stack dasturchisiz: to'liq funksional, zamonaviy va "
            "estetik jihatdan yuqori saviyali kod yozing."
        )
    else:
        tier_depth = "Siz Standart yordamchisiz: aniq, ixcham va to'g'ri ishlaydigan kod yozing."
    return "\n\n".join([
        "Siz OLMABEK CODE — OLMABEK AI platformasining Google AI Studio asosidagi professional dasturlash muharririsiz.",
        tier_depth,
        stack_guidance,
        "Javobni o'zbek tilida 1-2 qisqa gap bilan boshlang, so'ng to'liq kodni bitta asosiy fenced kod blokida (masalan ```html ... ``` yoki ```python ... ```) taqdim eting.",
        "Kodni hech qachon '...' deb qisqartirmang — har bir funksional qism to'liq yozilsin.",
    ])


def get_chat_system_instruction(plan: str, deep: bool) -> str:
    if plan == "ultra" and deep:
        depth = "Murakkab savollarni chuqur, bosqichma-bosqich, misollar va aniq xulosalar bilan tahlil qilib bering."
    elif plan == "pro":
        depth = "Aniq, tezkor, tizimli va foydali javoblar bering."
    else:
        depth = "Lo'nda, tushunarli va samimiy o'zbek tilida javob bering."
    return (
        f"Siz OLMABEK AI — o'zbek tilida mukammal fikrlaydigan sun'iy intellekt yordamchisiz. {depth} "
        "Zarur bo'lganda markdown sarlavhalar, ro'yxatlar va kod bloklaridan foydalaning."
    )


def select_model(plan: str, is_code: bool) -> str:
    if is_code and plan in ("pro", "ultra"):
        return CODE_PRO_MODEL
    if plan == "ultra":
        return CODE_PRO_MODEL
    return CHAT_MODEL


def sse_line(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def require_client() -> genai.Client:
    if not GEMINI_API_KEY or client is None:
        raise HTTPException(status_code=500, detail="Serverda GEMINI_API_KEY sozlanmagan")
    return client


# ===== ASOSIY ENDPOINTLAR =====

@app.get("/")
@app.get("/api/health")
def health_check():
    return {
        "ok": True,
        "status": "ok",
        "service": "Olmabek AI Server",
        "model": CHAT_MODEL,
        "codeProModel": CODE_PRO_MODEL,
        "imageModel": IMAGE_MODEL,
        "geminiConfigured": bool(GEMINI_API_KEY),
    }


@app.get("/api/limits")
def get_limits(user_id: str = "guest", plan: str = "standard", is_admin: bool = False):
    tier = plan if plan in CODE_TOKEN_LIMITS else "standard"
    record = get_or_reset_record(user_id)
    tokens_left = get_code_tokens_left(user_id, tier, is_admin)
    sites_left = 999_999 if (is_admin or tier == "ultra") else max(0, SITE_LIMITS[tier] - record["sites_used"])
    return {
        "plan": tier,
        "codeTokensLeft": tokens_left,
        "codeTokensLimit": CODE_TOKEN_LIMITS[tier],
        "sitesLeft": sites_left,
        "sitesLimit": SITE_LIMITS[tier],
    }


# 1. CHAT VA OLMABEK CODE (SYNC JSON)
@app.post("/api/chat")
@app.post("/api/code")
@app.post("/api/codex")
async def generate_sync_endpoint(req: ChatRequest, request: Request):
    ai = require_client()
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Xabar matni bo'sh")

    plan = req.plan or "standard"
    user_id = req.user_id or "guest"
    is_admin = bool(req.is_admin)
    is_code = req.mode in ("code", "codex") or "codex" in request.url.path

