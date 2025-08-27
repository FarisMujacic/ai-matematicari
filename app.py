# app.py — Async TEXT + IMAGE (Cloud Tasks + Firestore + GCS), bez Mathpix-a
# Troškovno optimizirano: vision-light triage + gpt-5-mini solver, cache, idempotencija, short prompts, limit history

from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify
from dotenv import load_dotenv
import os, re, base64, json, html, datetime, logging, hashlib
from datetime import timedelta
from uuid import uuid4
from io import BytesIO

from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from flask_cors import CORS

import gspread
from google.oauth2.service_account import Credentials as SACreds
import google.auth

try:
    from google.cloud import storage
except Exception:
    storage = None

# Optional server-side resize (ako Pillow postoji)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ---------------- Bootstrapping ----------------
load_dotenv(override=False)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("matbot")

SECURE_COOKIES = os.getenv("COOKIE_SECURE", "0") == "1"
app = Flask(__name__)
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=SECURE_COOKIES,
    SESSION_COOKIE_NAME="matbot_session_v2",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    SEND_FILE_MAX_AGE_DEFAULT=0,
    ETAG_DISABLED=True,
)
CORS(app, supports_credentials=True)
app.secret_key = os.getenv("SECRET_KEY", "tajna_lozinka")

MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# prag za "velike" slike za sync putanju (i dalje radimo async resize/triage)
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "1500000"))  # ≈1.5 MB

OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "240"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

MINI_SOLVER_MAX_TOKENS = int(os.getenv("MINI_SOLVER_MAX_TOKENS", "600"))
VISION_LIGHT_MAX_TOKENS = int(os.getenv("VISION_LIGHT_MAX_TOKENS", "350"))
VISION_FULL_MAX_TOKENS = int(os.getenv("VISION_FULL_MAX_TOKENS", "500"))

VISION_DETAIL_LOW_FIRST = os.getenv("VISION_DETAIL_LOW_FIRST", "1") == "1"
FALLBACK_VISION_FULL = os.getenv("FALLBACK_VISION_FULL", "0") == "1"

IMAGE_RESIZE_MAX = int(os.getenv("IMAGE_RESIZE_MAX", "1200"))
IMAGE_JPEG_QUALITY = int(os.getenv("IMAGE_JPEG_QUALITY", "75"))

CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "168"))  # 7 dana

# --- OpenAI ---
_OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not _OPENAI_API_KEY:
    log.error("OPENAI_API_KEY nije postavljen u okruženju.")
client = OpenAI(api_key=_OPENAI_API_KEY, timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)
MODEL_VISION_LIGHT = os.getenv("OPENAI_MODEL_VISION_LIGHT") or os.getenv("OPENAI_MODEL_VISION", "gpt-5")
MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-5-mini")
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-5")

# ---------------- Sheets (ENV/ADC friendly) ----------------
SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

GSHEET_ID   = os.getenv("GSHEET_ID", "").strip()
GSHEET_NAME = os.getenv("GSHEET_NAME", "matematika-bot").strip()

sheet = None
try:
    b64 = os.getenv("GOOGLE_SHEETS_CREDENTIALS_B64", "").strip()
    if b64:
        info  = json.loads(base64.b64decode(b64).decode("utf-8"))
        creds = SACreds.from_service_account_info(info, scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        log.info("Sheets via service_account_b64 (sa=%s)", info.get("client_email"))
    elif os.path.exists("credentials.json"):
        creds = SACreds.from_service_account_file("credentials.json", scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        log.info("Sheets via service_account_file")
    else:
        adc_creds, _ = google.auth.default(scopes=SHEETS_SCOPES)
        gc = gspread.authorize(adc_creds)
        log.info("Sheets via ADC default credentials")

    ss = gc.open_by_key(GSHEET_ID) if GSHEET_ID else gc.open(GSHEET_NAME)
    sheet = ss.sheet1
    log.info("Sheets enabled (title=%s id=%s)", getattr(ss, "title", "?"), getattr(ss, "id", "?"))
except Exception as e:
    log.warning("Sheets disabled: %s", e)
    sheet = None

def sheets_append_row_safe(values):
    if not sheet:
        return False
    try:
        sheet.append_row(values, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        log.warning("Sheets append error: %s", e)
        return False

def log_to_sheet(job_id, razred, user_text, odgovor_html, source_tag, model_name):
    # Kolone: vrijeme, razred, pitanje, odgovor(HTML skraćen), izvor|model, job_id
    ts = datetime.datetime.utcnow().isoformat()
    # trim da ne trpamo ogromne HTML-ove u Sheets
    ut = (user_text or "")[:400]
    oh = (odgovor_html or "")[:800]
    sheets_append_row_safe([ts, razred, ut, oh, f"{source_tag}|{model_name}", job_id])

# ---------------- GCS ----------------
GCS_BUCKET = (os.getenv("GCS_BUCKET") or "").strip()
GCS_SIGNED_GET = os.getenv("GCS_SIGNED_GET", "1") == "1"
storage_client = None
if GCS_BUCKET and storage is not None:
    try:
        storage_client = storage.Client()
        log.info("GCS enabled (bucket=%s, signed_get=%s)", GCS_BUCKET, GCS_SIGNED_GET)
    except Exception as e:
        log.error("GCS client init failed: %s", e)
        storage_client = None
else:
    if not GCS_BUCKET:
        log.info("GCS disabled (no GCS_BUCKET set).")
    else:
        log.warning("google-cloud-storage lib not available, GCS disabled.")

# ===================== Business logic helpers =====================
PROMPTI_PO_RAZREDU = {
    "5": "Pomoćnik iz matematike za 5. razred. Objašnjavaj jasno i kratko, korak po korak.",
    "6": "Pomoćnik iz matematike za 6. razred. Objašnjavaj jasno i kratko, korak po korak.",
    "7": "Pomoćnik iz matematike za 7. razred. Objašnjavaj jasno i kratko, korak po korak.",
    "8": "Pomoćnik iz matematike za 8. razred. Objašnjavaj jasno i kratko, korak po korak.",
    "9": "Pomoćnik iz matematike za 9. razred. Objašnjavaj jasno i kratko, korak po korak.",
}
DOZVOLJENI_RAZREDI = set(PROMPTI_PO_RAZREDU.keys())

ORDINAL_WORDS = {
    "prvi": 1, "drugi": 2, "treći": 3, "treci": 3, "četvrti": 4, "cetvrti": 4,
    "peti": 5, "šesti": 6, "sesti": 6, "sedmi": 7, "osmi": 8, "deveti": 9, "deseti": 10,
    "zadnji": -1, "posljednji": -1
}
_task_num_re = re.compile(
    r"(?:zadatak\s*(?:broj\s*)?(\d{1,4}))|(?:\b(\d{1,4})\s*\.)|(?:\b(" + "|".join(ORDINAL_WORDS.keys()) + r")\b)",
    flags=re.IGNORECASE
)
FOLLOWUP_TASK_RE = re.compile(r"^\s*\d{2,5}\s*[a-z]\)?\s*$", re.IGNORECASE)

def extract_requested_tasks(text: str):
    if not text:
        return []
    tasks = []
    for m in _task_num_re.finditer(text):
        if m.group(1):
            tasks.append(int(m.group(1)))
        elif m.group(2):
            tasks.append(int(m.group(2)))
        elif m.group(3):
            tasks.append(ORDINAL_WORDS.get(m.group(3).lower()))
    out, seen = [], set()
    for n in tasks:
        if n not in seen:
            out.append(n); seen.add(n)
    return out

def latexify_fractions(text):
    def zamijeni(m):
        return f"\\(\\frac{{{m.group(1)}}}{{{m.group(2)}}}\\)"
    return re.sub(r'\b(\d{1,4})/(\d{1,4})\b', zamijeni, text)

def add_plot_div_once(odgovor_html: str, expression: str) -> str:
    marker = f'class="plot-request"'
    expr_attr = f'data-expression="{html.escape(expression)}"'
    if (marker in odgovor_html) and (expr_attr in odgovor_html):
        return odgovor_html
    return odgovor_html + f'<div class="plot-request" data-expression="{html.escape(expression)}"></div>'

TRIGGER_PHRASES = [
    r"\bnacrtaj\b", r"\bnacrtati\b", r"\bcrtaj\b", r"\biscrtaj\b", r"\bskiciraj\b",
    r"\bgraf\b", r"\bgrafik\b", r"\bprika[žz]i\s+graf\b", r"\bplot\b", r"\bvizualizuj\b",
    r"\bnasrtaj\b"
]
NEGATION_PHRASES = [
    r"\bbez\s+grafa\b", r"\bne\s+crt(a|aj)\b", r"\bnemoj\s+crtati\b", r"\bne\s+treba\s+graf\b"
]
_trigger_re  = re.compile("|".join(TRIGGER_PHRASES), flags=re.IGNORECASE)
_negation_re = re.compile("|".join(NEGATION_PHRASES), flags=re.IGNORECASE)
def should_plot(text: str) -> bool:
    if not text:
        return False
    if _negation_re.search(text):
        return False
    return _trigger_re.search(text) is not None

_FUNC_PAT = re.compile(
    r"(?:y\s*=\s*[^;,\n]+)|(?:[fFgG]\s*\(\s*x\s*\)\s*=\s*[^;,\n]+)",
    flags=re.IGNORECASE
)
def extract_plot_expression(user_text: str, razred: str = "", history=None) -> str | None:
    if not user_text:
        return None
    m = _FUNC_PAT.search(user_text)
    if m:
        expr = m.group(0).strip()
        expr = re.sub(r"\s+", " ", expr)
        return expr
    return None

# Geo/diagram heuristika iz teksta
_GEOM_HINTS = re.compile(r"\b(trokut|trougao|pravougaonik|kvadrat|krug|poluprava|ugao|kut|radius|dijagram|geometrij)\b", re.IGNORECASE)

def is_geometry_like(text: str) -> bool:
    if not text: return False
    return _GEOM_HINTS.search(text) is not None

# ===================== OpenAI helpers =====================
def _openai_chat(model: str, messages: list, timeout: float = None, max_tokens: int | None = None):
    try:
        cli = client if timeout is None else client.with_options(timeout=timeout)
        return cli.chat.completions.create(
            model=model,
            messages=messages,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=max_tokens
        )
    except (APIConnectionError, APIStatusError, RateLimitError) as e:
        raise e
    except Exception as e:
        raise e

def _short_system_prompt(razred: str, only_clause: str = "", strict_geom_policy: str = ""):
    base = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])
    return (
        base +
        " Odgovaraj bosanskim (ijekavica). Budi sažet, ali jasan, koraci kratko."
        " Ako pitanje nije matematika: 'Molim te, postavi matematičko pitanje.'"
        " Ako ne znaš tačno: 'Obrati se instruktorima na info@matematicari.com'."
        " Bez ASCII grafova." +
        ((" " + only_clause) if only_clause else "") +
        ((" " + strict_geom_policy) if strict_geom_policy else "")
    )

# ===================== TEXT pipeline (mini) =====================
def answer_with_text_pipeline(pure_text: str, razred: str, history, requested):
    only_clause = ""
    strict_geom_policy = (
        "Ako problem uključuje geometriju: koristi samo eksplicitne podatke; ne pretpostavljaj; navedi naziv teorema."
    )
    if requested:
        only_clause = "Riješi ISKLJUČIVO zadatke: " + ", ".join(map(str, requested)) + ". Ostale ignoriraj."

    system_message = {"role": "system", "content": _short_system_prompt(razred, only_clause, strict_geom_policy)}
    messages = [system_message]
    # limitiraj history na zadnja 2 para
    for msg in history[-2:]:
        messages.append({"role":"user","content": str(msg.get("user",""))[:1500]})
        messages.append({"role":"assistant","content": str(msg.get("bot",""))[:2500]})
    messages.append({"role":"user","content": pure_text})

    response = _openai_chat(MODEL_TEXT, messages, timeout=OPENAI_TIMEOUT, max_tokens=MINI_SOLVER_MAX_TOKENS)
    actual_model = getattr(response, "model", MODEL_TEXT)
    raw = response.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    html_out = f"<p>{latexify_fractions(raw)}</p>"
    return html_out, actual_model

# ===================== Vision-light EXTRACT (jeftino) =====================
def _vision_messages_base(razred: str, history, only_clause: str, strict_geom_policy: str):
    system_message = {"role": "system", "content": _short_system_prompt(razred, only_clause, strict_geom_policy)}
    messages = [system_message]
    for msg in history[-2:]:
        messages.append({"role":"user","content": str(msg.get("user",""))[:1500]})
        messages.append({"role":"assistant","content": str(msg.get("bot",""))[:2500]})
    return messages

def _vision_clauses(requested_tasks):
    only_clause = ""
    if requested_tasks:
        only_clause = "Riješi ISKLJUČIVO sljedeće zadatke (ostale ignoriraj): " + ", ".join(map(str, requested_tasks)) + "."
    strict_geom_policy = "Radi tačno; ne pretpostavljaj svojstva bez oznake; konačan odgovor ako je moguće."
    return only_clause, strict_geom_policy

def _detect_exercise_numbers_from_image(user_content):
    try:
        sys = {
            "role": "system",
            "content": "Samo pročitaj sliku i izdvoji VIDljive brojeve zadataka (primjeri: 497, 498). Vrati isključivo brojeve, zarezom. Ako nema, odgovori 'NONE'."
        }
        msgs = [sys, {"role": "user", "content": user_content}]
        resp = _openai_chat(MODEL_VISION_LIGHT, msgs, timeout=20, max_tokens=50)
        txt = (resp.choices[0].message.content or "").strip()
        if txt.upper() == "NONE":
            return []
        return [int(n) for n in re.findall(r"\d{2,5}", txt)]
    except Exception as e:
        log.warning("Detect numbers fail: %s", e)
        return []

def _safe_json_extract(text: str) -> dict:
    # pokuša izvući prvi JSON objekt iz teksta
    try:
        # direktan JSON?
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

def vision_light_extract_text_and_flags(user_content):
    """
    Prompt: vrati JSON: {"has_diagram": bool, "text": "...", "numbers":[...]}
    """
    sys = {
        "role": "system",
        "content": "Samo ekstrakcija. Vrati JEDAN JSON: {\"has_diagram\":true/false, \"text\":\"...\", \"numbers\":[...]}."
    }
    usr = {
        "role": "user",
        "content": user_content + [
            {"type":"text","text":"1) Detektuj da li postoji crtež/diagram/geometrijska skica (true/false). "
                                  "2) Sažmi i PREPIŠI problem u običan tekst (kratko). "
                                  "3) Ako se vide brojevi zadataka (497, 498...), dodaj u numbers."}
        ]
    }
    try:
        resp = _openai_chat(MODEL_VISION_LIGHT, [sys, usr], timeout=60, max_tokens=VISION_LIGHT_MAX_TOKENS)
        raw = (resp.choices[0].message.content or "").strip()
        j = _safe_json_extract(raw)
        has_diagram = bool(j.get("has_diagram"))
        text = str(j.get("text") or "").strip()
        numbers = j.get("numbers") or []
        if not numbers:
            # fallback kratka detekcija brojeva
            numbers = _detect_exercise_numbers_from_image(user_content)
        return has_diagram, text, numbers, getattr(resp, "model", MODEL_VISION_LIGHT)
    except Exception as e:
        log.warning("vision_light_extract_text_and_flags fail: %s", e)
        return False, "", [], MODEL_VISION_LIGHT

def _map_ordinals_to_detected(requested, detected):
    if not requested:
        return []
    out = []
    for t in requested:
        if isinstance(t, int) and t <= 0:
            if detected:
                out.append(detected[-1])
        elif isinstance(t, int) and t <= 10 and detected and t <= len(detected):
            out.append(detected[t-1])
        else:
            out.append(t)
    dedup, seen = [], set()
    for n in out:
        if n not in seen:
            dedup.append(n); seen.add(n)
    return dedup

# ============== Jeftina ruta slike: Vision-light -> tekst -> mini solver; opcionalni fallback na full Vision
def _build_image_url_part(url: str, low: bool = True):
    if low:
        return {"type":"image_url","image_url":{"url": url, "detail":"low"}}
    else:
        return {"type":"image_url","image_url":{"url": url}}

def _bytes_to_data_url_jpeg(b: bytes) -> str:
    b64 = base64.b64encode(b).decode()
    return f"data:image/jpeg;base64,{b64}"

def _compress_image_if_possible(image_bytes: bytes) -> bytes:
    if not PIL_AVAILABLE:
        return image_bytes
    try:
        im = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = im.size
        mx = max(w, h)
        if mx > IMAGE_RESIZE_MAX:
            scale = IMAGE_RESIZE_MAX / float(mx)
            im = im.resize((int(w*scale), int(h*scale)))
        out = BytesIO()
        im.save(out, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
        return out.getvalue()
    except Exception as e:
        log.warning("compress_image_if_possible failed: %s", e)
        return image_bytes

# ===================== CACHE helpers (Firestore) =====================
def _firestore_client():
    from google.cloud import firestore  # type: ignore
    return firestore.Client(project=PROJECT_ID or None)

def _cache_key(razred: str, user_text: str, image_bytes: bytes | None, image_url: str | None) -> str:
    h = hashlib.sha256()
    h.update((razred or "").encode())
    h.update((user_text or "").encode())
    if image_bytes:
        h.update(image_bytes)
    if image_url:
        h.update(image_url.encode())
    return h.hexdigest()

def cache_get(fs, key: str):
    try:
        doc = fs.collection("cache").document(key).get()
        if not doc.exists: return None
        data = doc.to_dict() or {}
        # TTL provjera
        ts = data.get("created_at")
        if isinstance(ts, datetime.datetime):
            age = datetime.datetime.utcnow().replace(tzinfo=None) - ts.replace(tzinfo=None)
            if age.total_seconds() > CACHE_TTL_HOURS*3600:
                return None
        return data
    except Exception as e:
        log.warning("cache_get error: %s", e)
        return None

def cache_put(fs, key: str, result: dict):
    try:
        from google.cloud import firestore as gcf  # type: ignore
        fs.collection("cache").document(key).set({
            "created_at": gcf.SERVER_TIMESTAMP,
            "result": result
        }, merge=True)
    except Exception as e:
        log.warning("cache_put error: %s", e)

# ===================== Image flows =====================
def route_image_flow_url(image_url: str, razred: str, history, requested_tasks=None, user_text=None):
    # Vision-light ekstrakcija (detail low -> jeftino)
    low = VISION_DETAIL_LOW_FIRST
    user_content = []
    if user_text:
        user_content.append({"type":"text","text": f"Korisnički tekst: {user_text}"})
    user_content.append({"type":"text","text":"Na slici je matematički zadatak."})
    user_content.append(_build_image_url_part(image_url, low=low))

    # Extract tekst + flags
    has_diagram, extracted_text, detected, used_model_vl = vision_light_extract_text_and_flags(user_content)

    # Ako korisnik nije precizirao a ima više zadataka: vrati poruku da izabere (jeftino)
    mapped = _map_ordinals_to_detected(requested_tasks or [], detected)
    only_clause, strict_geom_policy = _vision_clauses(mapped or requested_tasks)

    if not (mapped or (requested_tasks or [])) and len(detected) >= 2:
        html_msg = (
            "<p>Vidim više zadataka na slici: <b>"
            + ", ".join(map(str, detected))
            + "</b>.</p><p>Napiši koje tačno želiš da riješim "
            "(npr. <code>497</code> ili <code>497, 498</code> ili "
            "<code>prvi</code>/<code>zadnji</code>), pa ću riješiti samo njih.</p>"
        )
        return html_msg, "triage", used_model_vl

    # Rješenje gpt-5-mini na tekstu (ako nema teksta, fallback)
    base_text = extracted_text.strip() or (user_text or "")
    if base_text:
        html_out, model_used = answer_with_text_pipeline(base_text, razred, history, mapped or requested_tasks or [])
        return html_out, "vision_light+mini", f"{used_model_vl}=>{model_used}"

    # Ako baš nema kvalitetnog teksta i dozvoljen fallback → full vision
    if FALLBACK_VISION_FULL:
        messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)
        messages.append({"role": "user", "content": user_content})
        try:
            resp = _openai_chat(MODEL_VISION, messages, timeout=OPENAI_TIMEOUT, max_tokens=VISION_FULL_MAX_TOKENS)
            actual_model = getattr(resp, "model", MODEL_VISION)
            raw = resp.choices[0].message.content
            raw = strip_ascii_graph_blocks(raw)
            return f"<p>{latexify_fractions(raw)}</p>", "vision_full", f"{used_model_vl}=>{actual_model}"
        except Exception as e:
            log.warning("OpenAI vision URL (fallback) error: %r", e)

    return "<p><b>Greška:</b> Ne mogu izvući tekst sa slike. Pokušaj ponovo s jasnijom slikom.</p>", "vision_light_error", used_model_vl

def route_image_flow(slika_bytes: bytes, razred: str, history, requested_tasks=None, user_text=None):
    # Opcioni server-side resize/kompresija (štedi)
    comp = _compress_image_if_possible(slika_bytes)
    data_url = _bytes_to_data_url_jpeg(comp)
    low = VISION_DETAIL_LOW_FIRST

    user_content = []
    if user_text:
        user_content.append({"type": "text", "text": f"Korisnički tekst: {user_text}"})
    user_content.append({"type": "text", "text": "Na slici je matematički zadatak."})
    user_content.append(_build_image_url_part(data_url, low=low))

    has_diagram, extracted_text, detected, used_model_vl = vision_light_extract_text_and_flags(user_content)
    mapped = _map_ordinals_to_detected(requested_tasks or [], detected)
    only_clause, strict_geom_policy = _vision_clauses(mapped or requested_tasks)

    if not (mapped or (requested_tasks or [])) and len(detected) >= 2:
        html_msg = (
            "<p>Vidim više zadataka na slici: <b>"
            + ", ".join(map(str, detected))
            + "</b>.</p><p>Napiši koje tačno želiš da riješim "
            "(npr. <code>497</code> ili <code>497, 498</code> ili "
            "<code>prvi</code>/<code>zadnji</code>), pa ću riješiti samo njih.</p>"
        )
        return html_msg, "triage", used_model_vl

    base_text = extracted_text.strip() or (user_text or "")
    if base_text:
        html_out, model_used = answer_with_text_pipeline(base_text, razred, history, mapped or requested_tasks or [])
        return html_out, "vision_light+mini", f"{used_model_vl}=>{model_used}"

    if FALLBACK_VISION_FULL:
        messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)
        messages.append({"role": "user", "content": user_content})
        try:
            resp = _openai_chat(MODEL_VISION, messages, timeout=OPENAI_TIMEOUT, max_tokens=VISION_FULL_MAX_TOKENS)
            actual_model = getattr(resp, "model", MODEL_VISION)
            raw = resp.choices[0].message.content
            raw = strip_ascii_graph_blocks(raw)
            return f"<p>{latexify_fractions(raw)}</p>", "vision_full", f"{used_model_vl}=>{actual_model}"
        except Exception as e:
            log.warning("OpenAI vision base64 (fallback) error: %r", e)

    return "<p><b>Greška:</b> Ne mogu izvući tekst sa slike. Pokušaj ponovo s jasnijom slikom.</p>", "vision_light_error", used_model_vl

# ===================== Request helpers =====================
def get_history_from_request():
    try:
        hx = request.form.get("history_json")
        if hx:
            data = json.loads(hx)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return None

# ===================== GCS helpers =====================
def gcs_upload_filestorage(f):
    if not (storage_client and GCS_BUCKET):
        return None
    ext = os.path.splitext(f.filename or "")[1].lower() or ".jpg"
    blob_name = f"uploads/{uuid4().hex}{ext}"
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    try:
        f.stream.seek(0)
        blob.upload_from_file(f.stream, content_type=f.mimetype or "application/octet-stream")
        if GCS_SIGNED_GET:
            url = blob.generate_signed_url(
                version="V4",
                expiration=datetime.timedelta(minutes=45),
                method="GET",
            )
        else:
            try:
                blob.make_public()
                url = blob.public_url
            except Exception:
                url = blob.generate_signed_url(
                    version="V4",
                    expiration=datetime.timedelta(minutes=45),
                    method="GET",
                )
        return url
    except Exception as e:
        log.error("GCS upload failed: %s", e)
        return None

@app.post("/gcs/signed-upload")
def gcs_signed_upload():
    if not (storage_client and GCS_BUCKET):
        return jsonify({"error": "GCS not configured"}), 400
    try:
        data = request.get_json(force=True, silent=True) or {}
        content_type = (data.get("contentType") or "image/jpeg").strip()
        ext = ".jpg"
        if "png" in content_type: ext = ".png"
        if "heic" in content_type: ext = ".heic"
        blob_name = f"uploads/{uuid4().hex}{ext}"

        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)

        upload_url = blob.generate_signed_url(
            version="V4",
            expiration=datetime.timedelta(minutes=15),
            method="PUT",
            content_type=content_type,
        )

        if GCS_SIGNED_GET:
            read_url = blob.generate_signed_url(
                version="V4",
                expiration=datetime.timedelta(minutes=45),
                method="GET",
            )
        else:
            try:
                blob.make_public()
                read_url = blob.public_url
            except Exception:
                read_url = blob.generate_signed_url(
                    version="V4",
                    expiration=datetime.timedelta(minutes=45),
                    method="GET",
                )

        return jsonify({"uploadUrl": upload_url, "readUrl": read_url})
    except Exception as e:
        log.error("signed-upload error: %s", e)
        return jsonify({"error": "failed to create signed url"}), 500

@app.get("/uploads/<name>")
def uploads(name):
    return send_from_directory(UPLOAD_DIR, name)

# ===================== Sync forma (neobavezno) =====================
@app.route("/", methods=["GET", "POST"])
def index():
    plot_expression_added = False
    history = get_history_from_request() or session.get("history", [])
    razred = (request.form.get("razred") or session.get("razred") or "").strip()

    if request.method == "POST":
        if razred not in DOZVOLJENI_RAZREDI:
            return render_template("index.html",
                                   history=history, razred=razred,
                                   error="Molim odaberi razred."), 400
        session["razred"] = razred

        try:
            pitanje = (request.form.get("pitanje", "") or "").strip()
            slika = request.files.get("slika")
            image_url = (request.form.get("image_url") or "").strip()
            is_ajax = request.form.get("ajax") == "1" or request.headers.get("X-Requested-With") == "XMLHttpRequest"

            # IMAGE via URL (sada ide jeftina ruta: vision-light -> mini)
            if image_url:
                combined_text = pitanje
                requested = extract_requested_tasks(combined_text)
                odgovor, used_path, used_model = route_image_flow_url(
                    image_url, razred, history, requested_tasks=requested, user_text=combined_text
                )
                session["last_image_url"] = image_url

                if (not plot_expression_added) and should_plot(combined_text):
                    expr = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expr:
                        odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True

                display_user = (combined_text + " [slika]") if combined_text else "[slika]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sheets_append_row_safe([pitanje, odgovor[:800], f"{used_path}|{used_model}"])

                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # IMAGE via FILE (sync režim ostaje ograničen veličinom)
            if slika and slika.filename:
                slika.stream.seek(0, os.SEEK_END); size_bytes = slika.stream.tell(); slika.stream.seek(0)
                combined_text = pitanje
                requested = extract_requested_tasks(combined_text)

                if size_bytes > MAX_IMAGE_BYTES:
                    kb = size_bytes // 1024; max_kb = MAX_IMAGE_BYTES // 1024
                    display_user = (combined_text + " [SLIKA]") if combined_text else "[SLIKA]"
                    bot_msg = (f"<p><b>Slika je prevelika ({kb} KB).</b> "
                               f"Smanji veličinu (max {max_kb} KB) i pokušaj ponovo.</p>")
                    history.append({"user": display_user, "bot": bot_msg})
                    history = history[-8:]; session["history"] = history
                    if is_ajax: return render_template("index.html", history=history, razred=razred)
                    return redirect(url_for("index"))

                body = slika.read()
                odgovor, used_path, used_model = route_image_flow(
                    body, razred, history, requested_tasks=requested, user_text=combined_text
                )

                # spremi kopiju za follow-up
                try:
                    ext = os.path.splitext(slika.filename or "")[1].lower() or ".jpg"
                    fname = f"{uuid4().hex}{ext}"
                    with open(os.path.join(UPLOAD_DIR, fname), "wb") as fp:
                        fp.write(body)
                    public_url = (request.url_root.rstrip("/") + "/uploads/" + fname)
                    session["last_image_url"] = public_url
                except Exception as _e:
                    log.warning("Couldn't persist small image copy: %s", _e)

                if (not plot_expression_added) and should_plot(combined_text):
                    expr = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expr:
                        odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True

                display_user = (combined_text + " [SLIKA]") if combined_text else "[SLIKA]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sheets_append_row_safe([pitanje, odgovor[:800], f"{used_path}|{used_model}"])

                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # PURE TEXT (sync)
            requested = extract_requested_tasks(pitanje)
            last_url = session.get("last_image_url")

            if last_url and (requested or (pitanje and FOLLOWUP_TASK_RE.match(pitanje))):
                odgovor, used_path, used_model = route_image_flow_url(
                    last_url, razred, history, requested_tasks=requested, user_text=pitanje
                )
                if (not plot_expression_added) and should_plot(pitanje):
                    expr = extract_plot_expression(pitanje, razred=razred, history=history)
                    if expr:
                        odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True

                history.append({"user": pitanje, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sheets_append_row_safe([pitanje, odgovor[:800], f"{used_path}|{used_model}"])

                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            odgovor, actual_model = answer_with_text_pipeline(pitanje, razred, history, requested)
            if (not plot_expression_added) and should_plot(pitanje):
                expr = extract_plot_expression(pitanje, razred=razred, history=history)
                if expr:
                    odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True

            history.append({"user": pitanje, "bot": odgovor.strip()})
            history = history[-8:]; session["history"] = history
            sheets_append_row_safe([pitanje, odgovor[:800], f"text|{actual_model}"])

        except Exception as e:
            log.error("FATAL index.POST: %r", e)
            err_html = f"<p><b>Greška servera:</b> {html.escape(str(e))}</p>"
            history.append({"user": request.form.get('pitanje') or "[SLIKA]", "bot": err_html})
            history = history[-8:]; session["history"] = history
            if request.form.get("ajax") == "1":
                return render_template("index.html", history=history, razred=razred)
            return redirect(url_for("index"))

    return render_template("index.html", history=history, razred=razred)

# ===================== Health & utils =====================
@app.errorhandler(413)
def too_large(e):
    msg = (f"<p><b>Greška:</b> Fajl je prevelik (limit {MAX_MB} MB). "
           f"Pokušaj ponovo ili smanji kvalitet.</p>")
    return render_template("index.html", history=[{"user":"[SLIKA]", "bot": msg}], razred=session.get("razred")), 413

@app.route("/clear", methods=["POST"])
def clear():
    if request.form.get("confirm_clear") == "1":
        session.pop("history", None)
        session.pop("razred", None)
        session.pop("last_image_url", None)
    if request.form.get("ajax") == "1":
        return render_template("index.html", history=[], razred=None)
    return redirect("/")

@app.get("/healthz")
def healthz():
    return {"ok": True}, 200

@app.get("/_healthz")
def _healthz():
    return {"ok": True}, 200

@app.get("/_ah/health")
def ah_health():
    return "OK", 200

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["Vary"] = "Cookie"
    ancestors = os.getenv("FRAME_ANCESTORS", "").strip()
    if ancestors:
        resp.headers["Content-Security-Policy"] = f"frame-ancestors {ancestors}"
    try:
        del resp.headers["X-Frame-Options"]
    except KeyError:
        pass
    return resp

def strip_ascii_graph_blocks(text: str) -> str:
    fence = re.compile(r"```([\s\S]*?)```", flags=re.MULTILINE)
    def looks_like_ascii_graph(block: str) -> bool:
        sample = block.strip()
        if len(sample) == 0: return False
        allowed = set(" \t\r\n-_|*^><().,/\\0123456789xyXY")
        ratio_allowed = sum(c in allowed for c in sample) / len(sample)
        lines = sample.splitlines()
        return (ratio_allowed > 0.9) and (3 <= len(lines) <= 40)
    def repl(m):
        block = m.group(1)
        return "" if looks_like_ascii_graph(block) else m.group(0)
    graf_re = re.compile(r"(Grafički prikaz.*?:\s*)?```[\s\S]*?```", re.IGNORECASE)
    text = graf_re.sub(lambda m: "" if "```" in m.group(0) else m.group(0), text)
    return fence.sub(repl, text)

# ===================== ASINHRONO: Cloud Tasks + Firestore + GCS =====================
PROJECT_ID        = (os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or "").strip()
REGION            = os.getenv("REGION", "europe-west1")
TASKS_QUEUE       = os.getenv("TASKS_QUEUE", "matbot-queue")
TASKS_TARGET_URL  = os.getenv("TASKS_TARGET_URL")  # npr. https://<run-url>/tasks/process
TASKS_SECRET      = os.getenv("TASKS_SECRET", "super-secret")

def _tasks_client():
    from google.cloud import tasks_v2  # type: ignore
    return tasks_v2.CloudTasksClient()

def _create_task(payload: dict, job_id: str):
    if not TASKS_TARGET_URL:
        raise RuntimeError("TASKS_TARGET_URL is not set")
    if not PROJECT_ID:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT/GCP_PROJECT is not set")
    tc = _tasks_client()
    from google.cloud import tasks_v2  # type: ignore
    parent = tc.queue_path(PROJECT_ID, REGION, TASKS_QUEUE)
    task_name = tc.task_path(PROJECT_ID, REGION, TASKS_QUEUE, job_id)  # idempotentno ime

    task = {
        "name": task_name,
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": TASKS_TARGET_URL,
            "headers": {
                "Content-Type": "application/json",
                "X-Tasks-Secret": TASKS_SECRET
            },
            "body": json.dumps(payload).encode(),
        }
    }

    # Kreiraj ili ignoriraj ako već postoji
    try:
        return tc.create_task(request={"parent": parent, "task": task})
    except Exception as e:
        if "AlreadyExists" in str(e):
            log.info("Task %s already exists, ignoring duplicate", job_id)
            return None
        raise

@app.route("/submit", methods=["POST", "OPTIONS"])
def submit_async():
    if request.method == "OPTIONS":
        return ("", 204)

    fs = _firestore_client()
    from google.cloud import firestore as gcf  # type: ignore

    # iz forme/klijenta (FormData ili query)
    razred = (request.form.get("razred") or request.args.get("razred") or "").strip()
    user_text = (request.form.get("user_text") or request.form.get("pitanje") or "").strip()
    image_url = (request.form.get("image_url") or request.args.get("image_url") or "").strip()

    # ako dođe JSON (tekst-only ili tekst + image_url/b64)
    data = request.get_json(silent=True) or {}
    if data:
        razred    = (data.get("razred")    or razred).strip()
        user_text = (data.get("pitanje")   or data.get("user_text") or user_text).strip()
        image_url = (data.get("image_url") or image_url).strip()

    requested = extract_requested_tasks(user_text)
    job_id = str(uuid4())

    fs.collection("jobs").document(job_id).set({
        "status": "pending",
        "created_at": gcf.SERVER_TIMESTAMP,
        "razred": razred,
        "user_text": user_text,
        "requested": requested,
    })

    payload = {
        "job_id": job_id,
        "razred": razred,
        "user_text": user_text,
        "requested": requested,
        "bucket": GCS_BUCKET,
        "image_path": None,
        "image_url": image_url or None
    }

    # ako je stigao FILE → pohrani u GCS
    if "file" in request.files:
        if not (storage_client and GCS_BUCKET):
            return jsonify({"error": "GCS not configured (GCS_BUCKET)"}), 400
        f = request.files["file"]
        name = f"uploads/{job_id}/{f.filename or 'image.bin'}"
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(name)
        blob.upload_from_file(f, content_type=f.mimetype or "application/octet-stream")
        payload["image_path"] = name
    else:
        # opciono: JSON base64 slika
        image_b64 = (data.get("image_b64") if data else None)
        if image_b64 and (storage_client and GCS_BUCKET):
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            raw = base64.b64decode(image_b64)
            name = f"uploads/{job_id}/image.bin"
            bucket = storage_client.bucket(GCS_BUCKET)
            blob = bucket.blob(name)
            blob.upload_from_string(raw, content_type="application/octet-stream")
            payload["image_path"] = name

    # enqueue (idempotentno ime)
    _create_task(payload, job_id)
    return jsonify({"job_id": job_id, "status": "queued"}), 202

@app.get("/status/<job_id>")
def async_status(job_id):
    fs = _firestore_client()
    doc = fs.collection("jobs").document(job_id).get()
    if not doc.exists:
        # bolje vratiti pending nego 404, da frontend ne prekida polling
        return jsonify({"status": "pending"}), 200
    return jsonify(doc.to_dict()), 200

@app.post("/tasks/process")
def tasks_process():
    if request.headers.get("X-Tasks-Secret") != TASKS_SECRET:
        return "Forbidden", 403

    payload = request.get_json(force=True)
    job_id     = payload["job_id"]
    bucket     = payload.get("bucket")
    image_path = payload.get("image_path")
    image_url  = payload.get("image_url")
    razred     = (payload.get("razred") or "").strip()
    user_text  = (payload.get("user_text") or "").strip()
    requested  = payload.get("requested") or []
    if razred not in DOZVOLJENI_RAZREDI:
        razred = "5"

    fs = _firestore_client()
    from google.cloud import firestore as gcf  # type: ignore

    # Idempotencija: ako već "processing"/"done" → ne dupliraj
    job_ref = fs.collection("jobs").document(job_id)
    snap = job_ref.get()
    if snap.exists:
        st = (snap.to_dict() or {}).get("status")
        if st in ("processing", "done"):
            return "OK", 200

    # markiraj processing (zatim nastavi)
    job_ref.set({"status":"processing"}, merge=True)

    try:
        history = []
        image_bytes = None

        # Priprema image bytes (za cache key i eventualno vision)
        if image_path and storage_client:
            blob = storage_client.bucket(bucket).blob(image_path)
            image_bytes = blob.download_as_bytes()
        elif image_url:
            # za cache koristimo samo URL string
            pass

        # CACHE lookup
        cache_key = _cache_key(razred, user_text, image_bytes, image_url)
        cached = cache_get(fs, cache_key)
        if cached and cached.get("result"):
            result = cached["result"]
            job_ref.set({
                "status": "done",
                "result": result,
                "finished_at": gcf.SERVER_TIMESTAMP,
                "razred": razred,
                "user_text": user_text,
                "requested": requested,
            }, merge=True)
            # Sheets log (kratko)
            log_to_sheet(job_id, razred, user_text, result.get("html") or "", result.get("path") or "cache", result.get("model") or "cache")
            return "OK", 200

        # ---------------- TRIAGE + GLAVNA OBRADA (jeftino) ----------------
        odgovor_html, used_path, used_model = None, None, None

        # Ako postoji slika
        if image_path or image_url:
            if image_path and image_bytes is None and storage_client:
                blob = storage_client.bucket(bucket).blob(image_path)
                image_bytes = blob.download_as_bytes()

            if image_bytes is not None:
                odgovor_html, used_path, used_model = route_image_flow(
                    image_bytes, razred, history=history, requested_tasks=requested, user_text=user_text
                )
            else:
                odgovor_html, used_path, used_model = route_image_flow_url(
                    image_url, razred, history=history, requested_tasks=requested, user_text=user_text
                )
        else:
            # čisti tekst → mini
            odgovor_html, used_model = answer_with_text_pipeline(user_text, razred, history, requested)
            used_path = "text"

        # ---------------- SPREMI REZULTAT ----------------
        result = {"html": odgovor_html, "path": used_path, "model": used_model}
        job_ref.set({
            "status": "done",
            "result": result,
            "finished_at": gcf.SERVER_TIMESTAMP,
            "razred": razred,
            "user_text": user_text,
            "requested": requested,
        }, merge=True)

        # cache put
        cache_put(fs, cache_key, result)

        # Sheets log (jednom po jobu)
        log_to_sheet(job_id, razred, user_text, odgovor_html, used_path, used_model)

        return "OK", 200

    except Exception as e:
        log.exception("Task processing failed")
        fs.collection("jobs").document(job_id).set({
            "status": "error",
            "error": str(e),
            "finished_at": gcf.SERVER_TIMESTAMP
        }, merge=True)
        # vrati 500 ako želiš da Cloud Tasks pokuša retry
        return "ERROR", 500


# ===================== Run =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
