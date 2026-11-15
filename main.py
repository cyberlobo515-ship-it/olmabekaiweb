"""
OLMABEK AI — AI + Payment + Telegram (bitta async loop, threading yo'q)
Ishga tushirish:
    pip install -r requirements.txt
    python main.py
"""

import os
import json
import time
import base64
import random
import logging
import asyncio
from pathlib import Path

import aiohttp
from aiohttp import web
from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(format="%(asctime)s | %(levelname)-7s | %(message)s", level=logging.INFO)
log = logging.getLogger("olmabek")

# =========================================================
# CONFIG
# =========================================================
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN  = os.environ.get("CF_API_TOKEN", "").strip()
TG_TOKEN      = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID", "").strip()
PORT          = int(os.environ.get("PORT", 10000))

FAST_MODEL   = "@cf/meta/llama-3.1-8b-instruct-fast"
THINK_MODEL  = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
IMAGE_MODEL  = "@cf/black-forest-labs/flux-1-schnell"
VISION_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"

SYSTEM_PROMPT = """
Sen — dasturchi @xamrakulovshokirbek tomonidan yaratilgan aqlli, zamonaviy va universal sun'iy intellekt — Olmabek AI san.

Sening asosiy vazifang:
- Foydalanuvchilarga har qanday sohada (dasturlash, ta'lim, ijod, kundalik savollar va tahlillarda) tez, aniq va foydali yordam berish.
- Javoblarni chiroyli vizual ko'rinishda, tushunarli, punktlarga ajratilgan va o'rinli emojilar bilan taqdim etish.

Qat'iy qoidalar:
1. Faqat o'zbek tilida, ravon va xushmuomala javob ber.
2. "Sen kimsan?" / "Seni kim yaratgan?" deb so'ralsa: "Meni dasturchi @xamrakulovshokirbek yaratgan. Men uning loyihasi asosida sizga yordam berish uchun ishlab chiqilgan aqlli sun'iy intellekt — Olmabek AI man!" deb javob ber.
3. O'zingni Meta, OpenAI, Google yoki Cloudflare mahsuloti deb tanishtirma.
"""

BASE_DIR      = Path(__file__).parent
PAYMENTS_FILE = BASE_DIR / "payments.json"

# =========================================================
# STORAGE
# =========================================================
_payments = {}
_payments_lock = asyncio.Lock()

def load_payments():
    global _payments
    if PAYMENTS_FILE.exists():
        try:
            _payments = json.loads(PAYMENTS_FILE.read_text(encoding="utf-8"))
            log.info(f"💾 {len(_payments)} ta to'lov yuklandi")
        except Exception as e:
            log.error(f"load: {e}")

def save_payments():
    try:
        PAYMENTS_FILE.write_text(json.dumps(_payments, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.error(f"save: {e}")

# =========================================================
# CLOUDFLARE AI (async)
# =========================================================
def cf_url(model):
    return f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"

def cf_headers():
    return {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}

async def cf_call(model, payload, timeout=90):
    async with aiohttp.ClientSession() as s:
        async with s.post(cf_url(model), headers=cf_headers(), json=payload, timeout=timeout) as r:
            if r.status != 200:
                err = await r.text()
                raise Exception(f"CF {r.status}: {err[:300]}")
            data = await r.json()
            return (data.get("result", {}).get("response") or "").strip()

async def cf_image(prompt, steps=4):
    async with aiohttp.ClientSession() as s:
        async with s.post(cf_url(IMAGE_MODEL), headers=cf_headers(),
                          json={"prompt": prompt, "steps": steps}, timeout=90) as r:
            if r.status != 200:
                err = await r.text()
                raise Exception(f"CF {r.status}: {err[:300]}")
            data = await r.json()
            b64 = data.get("result", {}).get("image")
            if not b64:
                raise Exception("Rasm bo'sh")
            return base64.b64decode(b64)

# =========================================================
# TELEGRAM (to'g'ridan-to'g'ri HTTP)
# =========================================================
async def tg_send_receipt(payment):
    if not TG_TOKEN or not ADMIN_CHAT_ID:
        log.warning("⚠️ TG sozlanmagan")
        return
    try:
        img = base64.b64decode(payment['receipt_b64'])
        form = aiohttp.FormData()
        form.add_field('chat_id', ADMIN_CHAT_ID)
        form.add_field('photo', img, filename='receipt.png', content_type='image/png')
        caption = (
            f"💳 <b>Yangi to'lov</b>\n\n"
            f"🆔 <code>{payment['id']}</code>\n"
            f"👤 {payment['name']}\n"
            f"📧 {payment['email']}\n"
            f"📦 Tarif: <b>{payment['plan'].upper()}</b>\n"
            f"💰 {payment['price']} so'm\n"
            f"🕐 {time.strftime('%Y-%m-%d %H:%M')}"
        )
        form.add_field('caption', caption)
        form.add_field('parse_mode', 'HTML')
        form.add_field('reply_markup', json.dumps({
            'inline_keyboard': [[
                {'text': '✅ Tasdiqlash', 'callback_data': f'approve_{payment["id"]}'},
                {'text': '❌ Rad etish', 'callback_data': f'reject_{payment["id"]}'}
            ]]
        }))
        async with aiohttp.ClientSession() as s:
            async with s.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                              data=form, timeout=60) as r:
                data = await r.json()
                if data.get('ok'):
                    log.info(f"📨 Chek yuborildi: {payment['id']}")
                else:
                    log.error(f"TG send: {data.get('description', '?')}")
    except Exception as e:
        log.error(f"TG send xato: {e}")

async def tg_answer_callback(cb_id, text):
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(f"https://api.telegram.org/bot{TG_TOKEN}/answerCallbackQuery",
                         json={"callback_query_id": cb_id, "text": text}, timeout=10)
    except Exception: pass

async def tg_edit_caption(chat_id, msg_id, caption):
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(f"https://api.telegram.org/bot{TG_TOKEN}/editMessageCaption",
                         json={
                             "chat_id": chat_id, "message_id": msg_id,
                             "caption": caption, "parse_mode": "HTML"
                         }, timeout=10)
    except Exception: pass

async def tg_handle_callback(cb):
    data = cb.get('data') or ''
    if '_' not in data:
        return
    action, pid = data.split('_', 1)
    cb_id = cb.get('id')
    msg = cb.get('message', {})
    chat_id = msg.get('chat', {}).get('id')
    msg_id = msg.get('message_id')
    old_caption = msg.get('caption') or ''

    async with _payments_lock:
        p = _payments.get(pid)
        if not p:
            await tg_answer_callback(cb_id, "To'lov topilmadi")
            return
        p['status'] = 'approved' if action == 'approve' else 'rejected'
        p['reviewed'] = int(time.time())
        save_payments()

    emoji = '✅' if action == 'approve' else '❌'
    label = 'TASDIQLANDI' if action == 'approve' else 'RAD ETILDI'
    await tg_edit_caption(chat_id, msg_id, old_caption + f"\n\n{emoji} <b>{label}</b>")
    await tg_answer_callback(cb_id, f"{emoji} {label}")
    log.info(f"🔔 {pid} → {action}")

async def tg_polling():
    """Telegram getUpdates — bitta async loop ichida"""
    if not TG_TOKEN:
        log.info("🤖 TG bot o'chirilgan (token yo'q)")
        return

    # Eski webhook'ni o'chirish (agar bo'lsa)
    try:
        async with aiohttp.ClientSession() as s:
            await s.get(f"https://api.telegram.org/bot{TG_TOKEN}/deleteWebhook", timeout=10)
    except Exception: pass

    log.info("🤖 Telegram polling boshlandi")
    offset = 0

    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as r:
                    data = await r.json()
                    if not data.get('ok'):
                        log.error(f"TG getUpdates: {data.get('description', '?')}")
                        await asyncio.sleep(5)
                        continue
                    for upd in data.get('result', []):
                        offset = upd['update_id'] + 1
                        cb = upd.get('callback_query')
                        if cb:
                            await tg_handle_callback(cb)
        except asyncio.CancelledError:
            log.info("🛑 Telegram polling to'xtatildi")
            break
        except Exception as e:
            log.error(f"TG poll xato: {e}")
            await asyncio.sleep(3)

# =========================================================
# HTTP HANDLERS
# =========================================================
@web.middleware
async def cors_mw(request, handler):
    if request.method == 'OPTIONS':
        resp = web.Response(status=204)
    else:
        try:
            resp = await handler(request)
        except web.HTTPException as e:
            resp = e
        except Exception as e:
            log.exception("Handler xatosi")
            resp = web.json_response({"error": str(e)[:300]}, status=500)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

async def h_index(request):
    return web.FileResponse(BASE_DIR / "index.html")

async def h_style(request):
    return web.FileResponse(BASE_DIR / "style.css", headers={"Content-Type": "text/css"})

async def h_script(request):
    return web.FileResponse(BASE_DIR / "script.js", headers={"Content-Type": "application/javascript"})

async def h_health(request):
    return web.json_response({
        "status": "ok",
        "service": "Olmabek AI + Payment",
        "ai_ready": bool(CF_ACCOUNT_ID and CF_API_TOKEN),
        "tg_ready": bool(TG_TOKEN and ADMIN_CHAT_ID),
        "mode": "polling",
        "payments": len(_payments)
    })

async def h_chat(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "JSON xato"}, status=400)

    message = (body.get("message") or "").strip()
    if not message:
        return web.json_response({"error": "message bo'sh"}, status=400)
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return web.json_response({"error": "CF keys yo'q"}, status=500)

    mode = body.get("mode", "fast")
    history = body.get("history") or []
    model = THINK_MODEL if mode == "think" else FAST_MODEL

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-20:]:
        r, c = h.get("role"), (h.get("content") or "").strip()
        if r in ("user", "assistant") and c:
            msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": message})

    try:
        reply = await cf_call(model, {"messages": msgs, "max_tokens": 2048})
        return web.json_response({"reply": reply or "Javob bo'sh qaytdi.", "mode": mode})
    except Exception as e:
        log.exception("Chat")
        return web.json_response({"error": str(e)[:300]}, status=500)

async def h_image(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "JSON xato"}, status=400)

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return web.json_response({"error": "prompt bo'sh"}, status=400)
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return web.json_response({"error": "CF keys yo'q"}, status=500)

    try:
        en = prompt
        try:
            tp = {"messages": [
                {"role": "system", "content": "Translate to detailed English prompt for FLUX. Add 'photorealistic, 8k, sharp focus'. Output ONLY the prompt."},
                {"role": "user", "content": prompt}],
                "max_tokens": 120}
            en = await cf_call(FAST_MODEL, tp, timeout=30) or prompt
        except Exception:
            pass
        img = await cf_image(en)
        return web.json_response({
            "image": base64.b64encode(img).decode("ascii"),
            "prompt": prompt,
            "prompt_english": en
        })
    except Exception as e:
        log.exception("Image")
        return web.json_response({"error": str(e)[:300]}, status=500)

async def h_vision(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "JSON xato"}, status=400)

    img = body.get("image", "")
    cap = (body.get("caption") or "Ushbu rasmda nima tasvirlangan? O'zbek tilida batafsil tahlil qilib ber.").strip()
    if not img:
        return web.json_response({"error": "image bo'sh"}, status=400)

    try:
        payload = {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cap}],
            "image": list(base64.b64decode(img)),
            "max_tokens": 1500}
        reply = await cf_call(VISION_MODEL, payload)
        return web.json_response({"reply": reply or "Javob bo'sh qaytdi."})
    except Exception as e:
        log.exception("Vision")
        return web.json_response({"error": str(e)[:300]}, status=500)

async def h_send_receipt(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "JSON xato"}, status=400)

    receipt = (body.get("receipt") or "").strip()
    plan = (body.get("plan") or "").strip()
    if not receipt or not plan:
        return web.json_response({"error": "receipt yoki plan yo'q"}, status=400)
    if not TG_TOKEN or not ADMIN_CHAT_ID:
        return web.json_response({"error": "Telegram sozlanmagan"}, status=500)

    pid = f"p_{int(time.time())}_{random.randint(1000, 9999)}"
    payment = {
        'id': pid, 'plan': plan,
        'email': body.get('email', ''),
        'name': body.get('name', ''),
        'price': body.get('price', ''),
        'status': 'pending',
        'created': int(time.time()),
        'receipt_b64': receipt
    }
    async with _payments_lock:
        _payments[pid] = payment
        save_payments()

    asyncio.create_task(tg_send_receipt(payment))
    log.info(f"💳 Yangi to'lov: {pid} ({plan})")
    return web.json_response({"payment_id": pid, "status": "pending"})

async def h_payment_status(request):
    pid = request.match_info.get("pid", "")
    async with _payments_lock:
        p = _payments.get(pid)
    if not p:
        return web.json_response({"error": "topilmadi"}, status=404)
    return web.json_response({
        "status": p["status"],
        "plan": p["plan"],
        "reviewed": p.get("reviewed")
    })

async def h_favicon(request):
    # Inline SVG favicon
    svg = b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#0A0A0B"/><g fill="none" stroke="#D4AF37" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><path d="M32 24L19 19L11 29L13 43L22 53L42 53L51 43L53 29L45 19Z"/><path d="M32 22L38 13L48 10L43 19Z"/></g><text x="32" y="43" fill="#D4AF37" font-family="Arial" font-size="14" font-weight="800" text-anchor="middle">AI</text></svg>'''
    return web.Response(body=svg, content_type="image/svg+xml")

# =========================================================
# STARTUP
# =========================================================
async def on_startup(app):
    load_payments()
    log.info("=" * 60)
    log.info("  OLMABEK AI — AI + Payment + Telegram")
    log.info("=" * 60)
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        log.error("❌ CF_ACCOUNT_ID yoki CF_API_TOKEN yo'q")
    if not TG_TOKEN or not ADMIN_CHAT_ID:
        log.warning("⚠️ Telegram sozlanmagan — to'lov ishlamaydi")

    # Telegram polling — async task sifatida
    app['tg_task'] = asyncio.create_task(tg_polling())
    log.info(f"🚀 Server: http://0.0.0.0:{PORT}")

async def on_cleanup(app):
    task = app.get('tg_task')
    if task:
        task.cancel()
        try: await task
        except Exception: pass

def main():
    app = web.Application(client_max_size=20 * 1024 * 1024, middlewares=[cors_mw])
    app.router.add_get("/", h_index)
    app.router.add_get("/index.html", h_index)
    app.router.add_get("/style.css", h_style)
    app.router.add_get("/script.js", h_script)
    app.router.add_get("/favicon.ico", h_favicon)
    app.router.add_get("/health", h_health)
    app.router.add_get("/api/health", h_health)
    app.router.add_post("/api/chat", h_chat)
    app.router.add_post("/api/image", h_image)
    app.router.add_post("/api/vision", h_vision)
    app.router.add_post("/api/send-receipt", h_send_receipt)
    app.router.add_get("/api/payment-status/{pid}", h_payment_status)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    web.run_app(app, host="0.0.0.0", port=PORT, print=None)

if __name__ == "__main__":
    main()