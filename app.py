from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify
from dotenv import load_dotenv
import os, re, base64, json, html, datetime, logging, mimetypes, threading, traceback
from datetime import timedelta
from uuid import uuid4
import requests
from urllib.parse import urlparse
from openai import OpenAI
from flask_cors import CORS

# --- Optional PIL (for image heuristics and selftest) ---
try:
    from PIL import Image, ImageStat, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

import gspread
from google.oauth2.service_account import Credentials as SACreds
import google.auth

# --- Optional GCP clients ---
try:
    from google.cloud import storage as gcs_lib  # type: ignore
except Exception:
    gcs_lib = None
try:
    from google.cloud import firestore as fs_lib  # type: ignore
except Exception:
    fs_db = None
    fs_lib = None
try:
    from google.cloud import tasks_v2  # type: ignore
except Exception:
    tasks_v2 = None

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

LOCAL_MODE = os.getenv("LOCAL_MODE", "0") == "1"
USE_FIRESTORE = os.getenv("USE_FIRESTORE", "1") == "1" and not LOCAL_MODE

MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "200"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

HARD_TIMEOUT_S = float(os.getenv("HARD_TIMEOUT_S", "120"))
OPENAI_TIMEOUT = HARD_TIMEOUT_S
OPENAI_MAX_RETRIES = 2

SYNC_SOFT_TIMEOUT_S = float(os.getenv("SYNC_SOFT_TIMEOUT_S", "8"))
HEAVY_TOKEN_THRESHOLD = int(os.getenv("HEAVY_TOKEN_THRESHOLD", "1500"))

def _budgeted_timeout(default: float | int = None, margin: float = 5.0) -> float:
    run_lim = float(os.getenv("RUN_TIMEOUT_SECONDS", "300") or 300)
    want = float(default if default is not None else OPENAI_TIMEOUT)
    return max(5.0, min(want, run_lim - margin))

# --- OpenAI client ---
_OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not _OPENAI_API_KEY:
    log.error("OPENAI_API_KEY nije postavljen u okruženju.")
client = OpenAI(api_key=_OPENAI_API_KEY, timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)

MODEL_VISION_LIGHT = os.getenv("OPENAI_MODEL_VISION_LIGHT") or os.getenv("OPENAI_MODEL_VISION", "gpt-5")
MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-5-mini")
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-5")

# --- Mathpix: auto-enable i default "prefer" ---
MATHPIX_APP_ID  = (os.getenv("MATHPIX_APP_ID")  or os.getenv("MATHPIX_API_ID")  or "").strip()
MATHPIX_APP_KEY = (os.getenv("MATHPIX_APP_KEY") or os.getenv("MATHPIX_API_KEY") or "").strip()

_use_flag = (os.getenv("USE_MATHPIX", "").strip())
MATHPIX_MODE = (os.getenv("MATHPIX_MODE", "prefer").strip().lower())

if _use_flag == "0" or MATHPIX_MODE in ("off", "disable", "disabled"):
    USE_MATHPIX = False
else:
    USE_MATHPIX = bool(MATHPIX_APP_ID and MATHPIX_APP_KEY) or (_use_flag == "1") or (MATHPIX_MODE in ("prefer","force","on"))

def _mathpix_enabled() -> bool:
    return bool(MATHPIX_APP_ID and MATHPIX_APP_KEY) and USE_MATHPIX

# --- Google Sheets ---
SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
GSHEET_ID   = os.getenv("GSHEET_ID", "").strip()
GSHEET_NAME = os.getenv("GSHEET_NAME", "matematika-bot").strip()

sheet = None
_sheets_diag = {
    "enabled": False, "mode": None, "sa_email": None,
    "spreadsheet_title": None, "spreadsheet_id": None,
    "worksheet_title": None, "error": None,
}
def _try_get_sa_email_from_creds(creds):
    email = getattr(creds, "service_account_email", None)
    if email: return email
    try:
        info = getattr(creds, "_service_account_email", None) or getattr(creds, "_subject", None)
        if info: return info
    except Exception:
        pass
    return None

try:
    gc = None
    b64 = os.getenv("GOOGLE_SHEETS_CREDENTIALS_B64", "").strip()
    if b64:
        info  = json.loads(base64.b64decode(b64).decode("utf-8"))
        creds = SACreds.from_service_account_info(info, scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        _sheets_diag["mode"] = "b64"; _sheets_diag["sa_email"] = info.get("client_email")
    elif os.path.exists("credentials.json"):
        creds = SACreds.from_service_account_file("credentials.json", scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        _sheets_diag["mode"] = "file"; _sheets_diag["sa_email"] = _try_get_sa_email_from_creds(creds)
    else:
        adc_creds, _ = google.auth.default(scopes=SHEETS_SCOPES)
        gc = gspread.authorize(adc_creds)
        _sheets_diag["mode"] = "adc"; _sheets_diag["sa_email"] = _try_get_sa_email_from_creds(adc_creds)

    if not GSHEET_ID and not GSHEET_NAME:
        raise RuntimeError("GSHEET_ID ili GSHEET_NAME moraju biti postavljeni.")
    ss = gc.open_by_key(GSHEET_ID) if GSHEET_ID else gc.open(GSHEET_NAME)
    try: ws = ss.sheet1
    except Exception: ws = ss.get_worksheet(0)
    sheet = ws
    _sheets_diag.update({
        "enabled": True,
        "spreadsheet_title": getattr(ss, "title", None),
        "spreadsheet_id": getattr(ss, "id", None),
        "worksheet_title": getattr(ws, "title", None),
    })
except Exception as e:
    _sheets_diag["error"] = str(e); sheet = None

def sheets_append_row_safe(values):
    if not sheet: return False
    try:
        sheet.append_row(values, value_input_option="USER_ENTERED"); return True
    except Exception:
        return False

def log_to_sheet(job_id, razred, user_text, odgovor_html, source_tag, model_name):
    ts = datetime.datetime.utcnow().isoformat()
    sheets_append_row_safe([ts, razred, user_text, odgovor_html, f"{source_tag}|{model_name}", job_id])

# --- GCS & Firestore ---
GCS_BUCKET = (os.getenv("GCS_BUCKET") or "").strip()
GCS_SIGNED_GET = os.getenv("GCS_SIGNED_GET", "1") == "1"
storage_client = None
if not LOCAL_MODE and GCS_BUCKET and gcs_lib is not None:
    try:
        storage_client = gcs_lib.Client()
    except Exception:
        storage_client = None

fs_db = None
JOB_STORE = {}
if USE_FIRESTORE and fs_lib is not None:
    try:
        fs_db = fs_lib.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT") or None)
    except Exception:
        fs_db = None

def store_job(job_id: str, data: dict, merge: bool = True):
    if fs_db:
        fs_db.collection("jobs").document(job_id).set(data, merge=merge)
    else:
        JOB_STORE[job_id] = {**JOB_STORE.get(job_id, {}), **data}

def read_job(job_id: str) -> dict:
    if fs_db:
        doc = fs_db.collection("jobs").document(job_id).get()
        return (doc.to_dict() or {}) if doc.exists else {}
    return JOB_STORE.get(job_id, {})

# --- Pedagoški promptovi (osnovni po razredu) ---
PROMPTI_PO_RAZREDU = {
    "5": "Ti si pomoćnik iz matematike za učenike 5. razreda osnovne škole. Objašnjavaj jednostavnim i razumljivim jezikom. Pomaži učenicima da razumiju zadatke iz prirodnih brojeva, osnovnih računskih operacija, jednostavne geometrije i tekstualnih zadataka. Svako rješenje objasni jasno, korak po korak.",
    "6": "Ti si pomoćnik iz matematike za učenike 6. razreda osnovne škole. Odgovaraj detaljno i pedagoški, koristeći primjere prikladne njihovom uzrastu. Pomaži im da razumiju razlomke, decimalne brojeve, procente, geometriju i tekstualne zadatke. Objasni rješenje jasno i korak po korak.",
    "7": "Ti si pomoćnik iz matematike za učenike 7. razreda osnovne škole. Pomaži u razumijevanju složenijih zadataka iz algebre, geometrije i funkcija. Koristi jasan, primjeren jezik i objasni svaki korak logično i precizno.",
    "8": "Ti si pomoćnik iz matematike za učenike 8. razreda osnovne škole. Fokusiraj se na linearne izraze, sisteme jednačina, geometriju i statistiku. Objasni postupke detaljno, korak po korak.",
    "9": "Ti si pomoćnik iz matematike za učenike 9. razreda osnovne škole. Pomaži u zadacima iz algebre, funkcija, geometrije i statistike. Koristi jasan i stručan jezik, ali primjeren nivou učenika. Objasni svaki korak rješenja jasno i precizno."
}
DOZVOLJENI_RAZREDI = set(PROMPTI_PO_RAZREDU.keys())

# --- COMMON TEACHING RULES (ljudski, univerzalno) ---
COMMON_RULES = (
    " Razgovaraj prirodno i strpljivo, kao nastavnik. "
    " Ako učenik pošalje nejasnu poruku ili napiše da nije shvatio, PODRAZUMIJEVAJ da se to odnosi na TVOJ POSLJEDNJI odgovor ili zadatak, "
    " osim ako učenik izričito ne kaže da mijenja temu. "
    " U tom slučaju objasni isti sadržaj DRUGAČIJE (jednostavnije, intuitivno ili sa kratkim paralelnim primjerom), "
    " a na kraju dodaj kratku rečenicu: 'Ako ti i dalje nije jasno, napiši šta ti tačno nije jasno.' "
    " Kod razlomaka koristi termine 'brojnik' i 'nazivnik' (osim ako korisnik uporno koristi druge nazive). "
    " Za linearne funkcije koristi isključivo zapis y = kx + n, gdje je k koeficijent pravca, a n odsječak na y-osi. "
    " Ako korisnik postavi podpitanje bez jasne reference, PODRAZUMIJEVAJ da se odnosi na POSLJEDNJI rješavani zadatak i NEMOJ pitati 'koji zadatak', osim ako korisnik eksplicitno promijeni temu. "
    " Po difoltu odgovaraj bosanskim (ijekavica) i ne koristi ASCII grafove osim ako su traženi."
)

ORDINAL_WORDS = {
    "prvi": 1, "drugi": 2, "treći": 3, "treci": 3, "četvrti": 4, "cetvrti": 4,
    "peti": 5, "šesti": 6, "sesti": 6, "sedmi": 7, "osmi": 8, "deveti": 9, "deseti": 10,
    "zadnji": -1, "posljednji": -1
}
_task_num_re = re.compile(
    r"(?:zadatak\s*(?:broj\s*)?(\d{1,4}))|(?:\b(\d{1,4})\s*\.)|(?:\b(" + "|".join(ORDINAL_WORDS.keys()) + r")\b)",
    flags=re.IGNORECASE
)

# --- FOLLOW-UP detekcija ---
FOLLOWUP_LETTER_RE = re.compile(r"^\s*([a-hčćđšž])\)?\s*$", re.IGNORECASE)
FOLLOWUP_PHRASES_RE = re.compile(r"\b(pod|tačka|tacka|stavka)\s*([a-hčćđšž])\)?\b", re.IGNORECASE)
FOLLOWUP_GENERIC_RE = re.compile(
    r"\b(zašto|zasto|kako|može|moze|pojasni|objasni|dalje|nastavi|korak|sljedeći|sledeci|još|jos)\b",
    re.IGNORECASE
)
FOLLOWUP_TASK_RE = re.compile(r"^\s*(\d{1,4}\s*[a-hčćđšž]\)?|[a-hčćđšž]\)?)\s*$", re.IGNORECASE)

def is_followup_like(text: str) -> bool:
    if not text: return False
    if FOLLOWUP_LETTER_RE.match(text): return True
    if FOLLOWUP_PHRASES_RE.search(text): return True
    if FOLLOWUP_GENERIC_RE.search(text) and len(text) <= 120: return True
    return False

def extract_requested_tasks(text: str):
    if not text: return []
    tasks = []
    for m in _task_num_re.finditer(text):
        if m.group(1): tasks.append(int(m.group(1)))
        elif m.group(2): tasks.append(int(m.group(2)))
        elif m.group(3): tasks.append(ORDINAL_WORDS.get(m.group(3).lower()))
    out, seen = [], set()
    for n in tasks:
        if n not in seen: out.append(n); seen.add(n)
    return out

def latexify_fractions(text):
    def zamijeni(m):
        return "\\(\\frac{" + m.group(1) + "}{" + m.group(2) + "}\\)"
    return re.sub(r'\b(\d{1,4})/(\d{1,4})\b', zamijeni, text)

def to_html_paragraphs(raw: str) -> str:
    # Izbjegavamo backslash u f-string izrazima: pripremi prije interpolacije
    safe = html.escape(raw).replace('\n', '<br>')
    return "<p>" + safe + "</p>"

def add_plot_div_once(odgovor_html: str, expression: str) -> str:
    marker = 'class="plot-request"'
    expr_attr = 'data-expression="' + html.escape(expression) + '"'
    if (marker in odgovor_html) and (expr_attr in odgovor_html):
        return odgovor_html
    return odgovor_html + '<div class="plot-request" data-expression="' + html.escape(expression) + '"></div>'

TRIGGER_PHRASES = [r"\bnacrtaj\b", r"\bnacrtati\b", r"\bcrtaj\b", r"\biscrtaj\b", r"\bskiciraj\b", r"\bgraf\b", r"\bgrafik\b", r"\bprika[žz]i\s+graf\b", r"\bplot\b", r"\bvizualizuj\b", r"\bnasrtaj\b"]
NEGATION_PHRASES = [r"\bbez\s+grafa\b", r"\bne\s+crt(a|aj)\b", r"\bnemoj\s+crtati\b", r"\bne\s+treba\s+graf\b"]
_trigger_re  = re.compile("|".join(TRIGGER_PHRASES), flags=re.IGNORECASE)
_negation_re = re.compile("|".join(NEGATION_PHRASES), flags=re.IGNORECASE)

def should_plot(text: str) -> bool:
    if not text: return False
    if _negation_re.search(text): return False
    return _trigger_re.search(text) is not None

_FUNC_PAT = re.compile(r"(?:y\s*=\s*[^;,\n]+)|(?:[fFgG]\s*\(\s*x\s*\)\s*=\s*[^;,\n]+)", flags=re.IGNORECASE)
def extract_plot_expression(user_text: str, razred: str = "", history=None) -> str | None:
    if not user_text: return None
    m = _FUNC_PAT.search(user_text)
    if m:
        expr = re.sub(r"\s+", " ", m.group(0).strip())
        return expr
    return None

def _short_name_for_display(name: str, maxlen: int = 60) -> str:
    n = os.path.basename(name or "").strip() or "nepoznato"
    if len(n) > maxlen:
        n = n[:maxlen-3] + "..."
    return html.escape(n)

def _name_from_url(u: str) -> str:
    try:
        p = urlparse(u)
        base = os.path.basename(p.path) or ""
        return _short_name_for_display(base if base else u.split("?")[0].split("/")[-1] or u)
    except Exception:
        return _short_name_for_display(u)

def _openai_chat(model: str, messages: list, timeout: float = None, max_tokens: int | None = None):
    def _do(params):
        cli = client if timeout is None else client.with_options(timeout=timeout)
        return cli.chat.completions.create(**params)
    params = {"model": model, "messages": messages}
    if max_tokens is not None:
        # kompatibilnost sa različitim SDK verzijama:
        try:
            params["max_completion_tokens"] = max_tokens
        except Exception:
            pass
    try:
        return _do(params)
    except Exception as e:
        msg = str(e)
        if "max_completion_tokens" in msg or "Unsupported parameter" in msg:
            params.pop("max_completion_tokens", None)
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            return _do(params)
        raise

def answer_with_text_pipeline(pure_text: str, razred: str, history, requested,
                              timeout_override: float | None = None,
                              is_followup: bool = False,
                              last_problem_text: str = ""):
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])
    only_clause = ""
    strict_geom_policy = " Ako problem uključuje geometriju: 1) koristi samo eksplicitno date podatke; 2) ne pretpostavljaj ništa bez oznake; 3) navedi nazive teorema (npr. unutrašnji naspramni, Thales...)."
    if requested:
        only_clause = " Riješi ISKLJUČIVO sljedeće zadatke: " + ", ".join(map(str, requested)) + ". Sve ostale primjere ignoriraj."
    system_message = {"role": "system", "content": (prompt_za_razred + COMMON_RULES + only_clause + strict_geom_policy)}
    messages = [system_message]
    for msg in history[-5:]:
        messages.append({"role":"user","content": msg["user"]})
        messages.append({"role":"assistant","content": msg["bot"]})

    # Prefiks za follow-up na TEKSTUALNI zadatak
    followup_prefix = ""
    if is_followup:
        if last_problem_text:
            followup_prefix = (
                "[PODPITANJE NA PRETHODNI ZADATAK]\n"
                "Nastavi objašnjenje istog zadatka. Ne pitaj 'koji zadatak'.\n"
                "Originalni tekst zadatka (posljednji korisnički unos): " + last_problem_text + "\n\n"
            )
        else:
            followup_prefix = (
                "[PODPITANJE NA PRETHODNI ZADATAK]\n"
                "Nastavi objašnjenje istog zadatka. Ne pitaj 'koji zadatak'.\n\n"
            )

    messages.append({"role":"user","content": followup_prefix + pure_text})
    response = _openai_chat(MODEL_TEXT, messages, timeout=timeout_override or OPENAI_TIMEOUT)
    actual_model = getattr(response, "model", MODEL_TEXT)
    raw = response.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    html_out = to_html_paragraphs(latexify_fractions(raw))
    return html_out, actual_model

def _vision_messages_base(razred: str, history, only_clause: str, strict_geom_policy: str):
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])
    system_message = {"role": "system", "content": (prompt_za_razred + COMMON_RULES + " " + only_clause + " " + strict_geom_policy)}
    messages = [system_message]
    for msg in history[-5:]:
        messages.append({"role": "user", "content": msg["user"]})
        messages.append({"role": "assistant", "content": msg["bot"]})
    return messages

def _vision_clauses():
    return "", " Radi tačno i oprezno. Ako nešto nedostaje, navedi šta nedostaje i stani."

def _sniff_image_mime(raw: bytes) -> str:
    if len(raw) >= 12:
        if raw.startswith(b"\x89PNG\r\n\x1a\n"): return "image/png"
        if raw[:3] == b"\xff\xd8\xff": return "image/jpeg"
        if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"): return "image/gif"
        if raw.startswith(b"BM"): return "image/bmp"
        if raw.startswith(b"II*\x00") or raw.startswith(b"MM\x00*"): return "image/tiff"
        if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP": return "image/webp"
    return "image/jpeg"

def _bytes_to_data_url(raw: bytes, mime_hint: str | None = None) -> str:
    mime = mime_hint if (mime_hint and mime_hint.startswith("image/")) else _sniff_image_mime(raw)
    b64 = base64.b64encode(raw).decode()
    return "data:" + mime + ";base64," + b64

def _heuristic_plain_text_image(img_bytes: bytes) -> bool:
    try:
        if len(img_bytes) > 4_000_000:
            return False
        if not HAVE_PIL:
            return True
        from io import BytesIO
        im = Image.open(BytesIO(img_bytes)).convert("RGB")
        w, h = im.size
        if w*h > 8_000_000:
            return False
        stat = ImageStat.Stat(im)
        mean = sum(stat.mean) / 3.0
        var  = sum(stat.var) / 3.0
        is_whiteish = mean > 200
        low_var = var < 1200
        return is_whiteish and low_var
    except Exception:
        return False

def mathpix_ocr_to_text(img_bytes: bytes) -> tuple[str | None, float]:
    if not _mathpix_enabled():
        return (None, 0.0)
    try:
        headers = {"app_id": MATHPIX_APP_ID, "app_key": MATHPIX_APP_KEY, "Content-type": "application/json"}
        img_b64 = base64.b64encode(img_bytes).decode()
        payload = {
            "src": "data:image/png;base64," + img_b64,
            "formats": ["text"],
            "data_options": {"include_asciimath": False, "include_latex": False},
            "rm_spaces": True
        }
        r = requests.post("https://api.mathpix.com/v3/text", headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            return (None, 0.0)
        j = r.json() or {}
        plain = (j.get("text") or "").strip()
        conf  = float(j.get("confidence") or 0.0)
        if not plain:
            return (None, 0.0)
        plain = (plain.replace("÷", "/").replace("×", "*").replace("–", "-").replace("—", "-"))
        return (plain, conf)
    except Exception:
        return (None, 0.0)

def route_image_flow_url(image_url: str, razred: str, history, user_text=None, timeout_override: float | None = None):
    only_clause, strict_geom_policy = _vision_clauses()
    try:
        r = requests.get(image_url, timeout=15)
        r.raise_for_status()
        mime_hint = r.headers.get("Content-Type") or None
        return route_image_flow(
            r.content, razred, history,
            user_text=user_text,
            timeout_override=timeout_override,
            mime_hint=mime_hint
        )
    except Exception as e:
        log.error("route_image_flow_url: download failed: %s", e)
    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)
    user_content = []
    if user_text:
        user_content.append({"type": "text", "text": "Korisnički tekst: " + user_text})
    user_content.append({"type": "text", "text": "Na slici je matematički zadatak."})
    user_content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages.append({"role": "user", "content": user_content})
    resp = _openai_chat(MODEL_VISION, messages, timeout=timeout_override or OPENAI_TIMEOUT)
    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return to_html_paragraphs(latexify_fractions(raw)), "vision_url", actual_model

def route_image_flow(slika_bytes: bytes, razred: str, history, user_text=None, timeout_override: float | None = None, mime_hint: str | None = None):
    try_mathpix = _mathpix_enabled() and (MATHPIX_MODE in ("prefer","force","on") or _heuristic_plain_text_image(slika_bytes))
    if try_mathpix:
        plain, conf = mathpix_ocr_to_text(slika_bytes)
        if plain:
            try:
                html_out, actual_model = answer_with_text_pipeline(
                    pure_text=plain if not user_text else (user_text + "\n\n" + plain),
                    razred=razred, history=history, requested=extract_requested_tasks(user_text or ""),
                    timeout_override=timeout_override or OPENAI_TIMEOUT
                )
                return html_out, "mathpix", actual_model
            except Exception:
                pass
    only_clause, strict_geom_policy = _vision_clauses()
    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)
    data_url = _bytes_to_data_url(slika_bytes, mime_hint=mime_hint)
    user_content = []
    if user_text: user_content.append({"type": "text", "text": "Korisnički tekst: " + user_text})
    user_content.append({"type": "text", "text": "Na slici je matematički zadatak."})
    user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    messages.append({"role": "user", "content": user_content})
    resp = _openai_chat(MODEL_VISION, messages, timeout=timeout_override or OPENAI_TIMEOUT)
    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return to_html_paragraphs(latexify_fractions(raw)), "vision_direct", actual_model

def get_history_from_request():
    try:
        hx = request.form.get("history_json")
        if hx:
            data = json.loads(hx)
            if isinstance(data, list): return data
    except Exception:
        pass
    return None

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

def gcs_upload_bytes(job_id: str, raw: bytes, filename_hint: str = "image.bin", content_type: str | None = None) -> str | None:
    if not (storage_client and GCS_BUCKET):
        return None
    ext = os.path.splitext(filename_hint or "")[1].lower() or ".bin"
    blob_name = "uploads/" + job_id + "/" + uuid4().hex + ext
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    try:
        blob.upload_from_string(raw, content_type=content_type or "application/octet-stream")
        return blob_name
    except Exception:
        return None

def gcs_upload_filestorage(f):
    if not (storage_client and GCS_BUCKET):
        return None
    ext = os.path.splitext(f.filename or "")[1].lower() or ".jpg"
    blob_name = "uploads/" + uuid4().hex + ext
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    try:
        f.stream.seek(0)
        blob.upload_from_file(f.stream, content_type=f.mimetype or "application/octet-stream")
        if GCS_SIGNED_GET:
            url = blob.generate_signed_url(version="V4", expiration=datetime.timedelta(minutes=45), method="GET")
        else:
            try:
                blob.make_public()
                url = blob.public_url
            except Exception:
                url = blob.generate_signed_url(version="V4", expiration=datetime.timedelta(minutes=45), method="GET")
        return url
    except Exception:
        return None

# ---------------- Web routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    plot_expression_added = False
    history = get_history_from_request() or session.get("history", [])
    razred = (request.form.get("razred") or session.get("razred") or "").strip()
    if request.method == "POST":
        if razred not in DOZVOLJENI_RAZREDI:
            return render_template("index.html", history=history, razred=razred, error="Molim odaberi razred."), 400
        session["razred"] = razred
        try:
            pitanje = (request.form.get("pitanje", "") or "").strip()
            slika = request.files.get("slika")
            image_url = (request.form.get("image_url") or "").strip()
            is_ajax = request.form.get("ajax") == "1" or request.headers.get("X-Requested-With") == "XMLHttpRequest"

            followup_flag = is_followup_like(pitanje)

            # --- IMAGE VIA URL ---
            if image_url:
                combined_text = pitanje
                odgovor, used_path, used_model = route_image_flow_url(image_url, razred, history, user_text=combined_text, timeout_override=HARD_TIMEOUT_S)
                session["last_image_url"] = image_url
                session["last_problem_kind"] = "image"
                if combined_text:
                    session["last_problem_user_text"] = combined_text

                if (not plot_expression_added) and should_plot(combined_text):
                    expr = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True

                file_label = _name_from_url(image_url)
                display_user = (combined_text + " [slika: " + file_label + "]") if combined_text else "[slika: " + file_label + "]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sync_job_id = "sync-" + uuid4().hex[:8]; log_to_sheet(sync_job_id, razred, combined_text, odgovor, "vision_url", used_model)
                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # --- IMAGE VIA FILE UPLOAD ---
            if slika and slika.filename:
                combined_text = pitanje
                body = slika.read()
                odgovor, used_path, used_model = route_image_flow(body, razred, history, user_text=combined_text, timeout_override=HARD_TIMEOUT_S, mime_hint=slika.mimetype or None)
                try:
                    ext = os.path.splitext(slika.filename or "")[1].lower() or ".img"
                    fname = uuid4().hex + ext
                    with open(os.path.join(UPLOAD_DIR, fname), "wb") as fp: fp.write(body)
                    public_url = (request.url_root.rstrip("/") + "/uploads/" + fname)
                    session["last_image_url"] = public_url
                except Exception:
                    pass

                session["last_problem_kind"] = "image"
                if combined_text:
                    session["last_problem_user_text"] = combined_text

                if (not plot_expression_added) and should_plot(combined_text):
                    expr = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True

                orig_name = _short_name_for_display(slika.filename or "upload")
                display_user = (combined_text + " [slika: " + orig_name + "]") if combined_text else "[slika: " + orig_name + "]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sync_job_id = "sync-" + uuid4().hex[:8]; log_to_sheet(sync_job_id, razred, combined_text, odgovor, "vision_direct", used_model)
                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # --- FOLLOW-UP NA PRETHODNU SLIKU (ili eksplicitno numerisan podzadatak) ---
            requested = extract_requested_tasks(pitanje)
            last_url = session.get("last_image_url")
            last_kind = session.get("last_problem_kind")

            if last_url and (followup_flag or requested or (pitanje and FOLLOWUP_TASK_RE.match(pitanje))):
                odgovor, used_path, used_model = route_image_flow_url(last_url, razred, history, user_text=pitanje, timeout_override=HARD_TIMEOUT_S)

                session["last_image_url"] = last_url
                session["last_problem_kind"] = "image"

                if (not plot_expression_added) and should_plot(pitanje):
                    expr = extract_plot_expression(pitanje, razred=razred, history=history)
                    if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True

                file_label = _name_from_url(last_url)
                display_user = (pitanje + " [slika: " + file_label + "]") if pitanje else "[slika: " + file_label + "]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sync_job_id = "sync-" + uuid4().hex[:8]; log_to_sheet(sync_job_id, razred, pitanje, odgovor, "vision_url", used_model)
                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # --- TEKSTUALNI ZADATAK (sa follow-up prefiksom po potrebi) ---
            is_followup_text_now = bool(is_followup_like(pitanje) and last_kind == "text")
            odgovor, actual_model = answer_with_text_pipeline(
                pitanje, razred, history, requested,
                timeout_override=HARD_TIMEOUT_S,
                is_followup=is_followup_text_now,
                last_problem_text=session.get("last_problem_user_text", "")
            )

            if (not plot_expression_added) and should_plot(pitanje):
                expr = extract_plot_expression(pitanje, razred=razred, history=history)
                if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True

            history.append({"user": pitanje, "bot": odgovor.strip()})
            history = history[-8:]; session["history"] = history

            session["last_problem_kind"] = "text"
            session["last_problem_user_text"] = pitanje

            sync_job_id = "sync-" + uuid4().hex[:8]; log_to_sheet(sync_job_id, razred, pitanje, odgovor, "text", actual_model)
        except Exception as e:
            err_html = "<p><b>Greška servera:</b> " + html.escape(str(e)) + "</p>"
            history.append({"user": request.form.get('pitanje') or "[SLIKA]", "bot": err_html})
            history = history[-8:]; session["history"] = history
            if request.form.get("ajax") == "1":
                return render_template("index.html", history=history, razred=razred)
            return redirect(url_for("index"))
    return render_template("index.html", history=history, razred=razred)

@app.errorhandler(413)
def too_large(e):
    msg = "<p><b>Greška:</b> Fajl je prevelik (limit " + str(MAX_MB) + " MB). Pokušaj ponovo ili smanji kvalitet.</p>"
    return render_template("index.html", history=[{"user":"[SLIKA]", "bot": msg}], razred=session.get("razred")), 413

@app.route("/clear", methods=["POST"])
def clear():
    if request.form.get("confirm_clear") == "1":
        session.pop("history", None); session.pop("razred", None); session.pop("last_image_url", None)
        session.pop("last_problem_kind", None); session.pop("last_problem_user_text", None)
    if request.form.get("ajax") == "1": return render_template("index.html", history=[], razred=None)
    return redirect("/")

@app.get("/healthz")
def healthz(): return {"ok": True, "local_mode": LOCAL_MODE}, 200
@app.get("/_healthz")
def _healthz(): return {"ok": True}, 200
@app.get("/_ah/health")
def ah_health(): return "OK", 200

@app.get("/sheets/diag")
def sheets_diag(): return jsonify(_sheets_diag), 200

@app.post("/sheets/selftest")
def sheets_selftest():
    if not sheet:
        return jsonify({"ok": False, "error": _sheets_diag.get("error") or "Sheets not initialized"}), 500
    row = [datetime.datetime.utcnow().isoformat(), "selftest", "Hello from /sheets/selftest", "<p>OK</p>", "selftest|none", "self-" + uuid4().hex[:8]]
    ok = sheets_append_row_safe(row); return jsonify({"ok": ok}), (200 if ok else 500)

# --- Mathpix selftest (opcionalno) ---
@app.get("/mathpix/selftest")
def mathpix_selftest():
    if not _mathpix_enabled():
        return jsonify({"ok": False, "reason": "no-keys"}), 400
    if not HAVE_PIL:
        return jsonify({"ok": False, "reason": "no-PIL"}), 400
    try:
        import io
        img = Image.new("RGB", (320, 80), "white")
        d = ImageDraw.Draw(img)
        d.text((10, 20), "12/3 + 5", fill="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out, conf = mathpix_ocr_to_text(buf.getvalue())
        return jsonify({"ok": True, "text": out, "confidence": conf}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    resp.headers["Pragma"] = "no-cache"; resp.headers["Expires"] = "0"; resp.headers["Vary"] = "Cookie"
    # CSP za Thinkific (dozvoli embed unutar Thinkifica)
    resp.headers["Content-Security-Policy"] = "frame-ancestors https://*.thinkific.com"
    try: del resp.headers["X-Frame-Options"]
    except KeyError: pass
    return resp

@app.get("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

@app.post("/gcs/signed-upload")
def gcs_signed_upload():
    if not storage_client or not GCS_BUCKET or LOCAL_MODE:
        return jsonify({"ok": False, "reason": "no-gcs"}), 200
    data = request.get_json(force=True, silent=True) or {}
    content_type = (data.get("contentType") or "image/jpeg").strip()
    obj = "uploads/" + uuid4().hex + ".bin"
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(obj)
    put_url = blob.generate_signed_url(version="V4", expiration=datetime.timedelta(minutes=15), method="PUT", content_type=content_type)
    if GCS_SIGNED_GET:
        read_url = blob.generate_signed_url(version="V4", expiration=datetime.timedelta(minutes=45), method="GET")
    else:
        try:
            blob.make_public()
            read_url = blob.public_url
        except Exception:
            read_url = blob.generate_signed_url(version="V4", expiration=datetime.timedelta(minutes=45), method="GET")
    return jsonify({"uploadUrl": put_url, "readUrl": read_url}), 200

# --- Cloud Tasks (async) ---
PROJECT_ID        = (os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or "").strip()
REGION            = os.getenv("REGION", "europe-west1")
TASKS_QUEUE       = os.getenv("TASKS_QUEUE", "matbot-queue")
TASKS_TARGET_URL  = os.getenv("TASKS_TARGET_URL")
TASKS_SECRET      = os.getenv("TASKS_SECRET", "super-secret")

def _create_task_cloud(payload: dict):
    if not tasks_v2:
        raise RuntimeError("google-cloud-tasks nije instaliran")
    if not TASKS_TARGET_URL:
        raise RuntimeError("TASKS_TARGET_URL je obavezan")
    if not PROJECT_ID:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT/GCP_PROJECT je obavezan")
    tc = tasks_v2.CloudTasksClient()
    parent = tc.queue_path(PROJECT_ID, REGION, TASKS_QUEUE)
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": TASKS_TARGET_URL,
            "headers": {"Content-Type": "application/json", "X-Tasks-Secret": TASKS_SECRET},
            "body": json.dumps(payload).encode(),
        }
    }
    return tc.create_task(request={"parent": parent, "task": task})

def _enqueue(payload: dict):
    if LOCAL_MODE or (not tasks_v2) or (not TASKS_TARGET_URL) or (not PROJECT_ID):
        threading.Thread(target=_local_worker, daemon=True, args=(payload,)).start()
    else:
        _create_task_cloud(payload)

def _process_job_core(payload: dict) -> dict:
    job_id     = payload["job_id"]
    bucket     = payload.get("bucket")
    image_path = payload.get("image_path")
    image_url  = payload.get("image_url")
    image_inline_b64 = payload.get("image_inline_b64")
    razred     = (payload.get("razred") or "").strip()
    user_text  = (payload.get("user_text") or "").strip()
    requested  = payload.get("requested") or []
    if razred not in DOZVOLJENI_RAZREDI: razred = "5"
    history = []
    task_ai_timeout = _budgeted_timeout(default=HARD_TIMEOUT_S, margin=5.0)
    if image_path:
        if not storage_client:
            raise RuntimeError("GCS storage client not initialized (image_path zadat).")
        blob = storage_client.bucket(bucket).blob(image_path)
        img_bytes = blob.download_as_bytes()
        mime_hint = blob.content_type or mimetypes.guess_type(image_path)[0] or None
        odgovor_html, used_path, used_model = route_image_flow(img_bytes, razred, history=history, user_text=user_text, timeout_override=task_ai_timeout, mime_hint=mime_hint)
    elif image_inline_b64:
        img_bytes = base64.b64decode(image_inline_b64)
        odgovor_html, used_path, used_model = route_image_flow(img_bytes, razred, history=history, user_text=user_text, timeout_override=task_ai_timeout, mime_hint=None)
    elif image_url:
        odgovor_html, used_path, used_model = route_image_flow_url(image_url, razred, history, user_text=user_text, timeout_override=task_ai_timeout)
    else:
        odgovor_html, used_model = answer_with_text_pipeline(user_text, razred, history, requested, timeout_override=task_ai_timeout)
        used_path = "text"
    result = {"html": odgovor_html, "path": used_path, "model": used_model}
    return {
        "status": "done",
        "result": result,
        "finished_at": datetime.datetime.utcnow().isoformat() + "Z",
        "razred": razred,
        "user_text": user_text,
        "requested": requested,
    }

def _local_worker(payload: dict):
    job_id = payload["job_id"]
    try:
        out = _process_job_core(payload)
        store_job(job_id, out, merge=True)
        try: log_to_sheet(job_id, out.get("razred"), out.get("user_text"), out["result"]["html"], out["result"]["path"], out["result"]["model"])
        except Exception: pass
    except Exception as e:
        err_html = "<p><b>Nije uspjela obrada.</b> Pokušaj ponovo ili pošalji jasniji unos.</p><p><code>" + html.escape(str(e)) + "</code></p>"
        store_job(job_id, {"status": "done", "result": {"html": err_html, "path": "error", "model": "n/a"}, "finished_at": datetime.datetime.utcnow().isoformat() + "Z"}, merge=True)

def estimate_tokens(text: str) -> int:
    if not text: return 0
    return max(0, len(text) // 4)

def looks_heavy(user_text: str, has_image: bool) -> bool:
    toks = estimate_tokens(user_text or "")
    return has_image or toks > HEAVY_TOKEN_THRESHOLD

def _sync_process_once(razred: str, user_text: str, requested: list, image_url: str | None, file_bytes: bytes | None, file_mime: str | None, timeout_s: float,
                       is_followup: bool = False, last_problem_text: str = "") -> dict:
    try:
        history = []
        if image_url:
            html_out, used_path, used_model = route_image_flow_url(image_url, razred, history, user_text=user_text, timeout_override=timeout_s)
            return {"ok": True, "result": {"html": html_out, "path": used_path, "model": used_model}}
        if file_bytes:
            html_out, used_path, used_model = route_image_flow(file_bytes, razred, history, user_text=user_text, timeout_override=timeout_s, mime_hint=file_mime or None)
            return {"ok": True, "result": {"html": html_out, "path": used_path, "model": used_model}}
        html_out, used_model = answer_with_text_pipeline(user_text, razred, history, requested, timeout_override=timeout_s, is_followup=is_followup, last_problem_text=last_problem_text)
        return {"ok": True, "result": {"html": html_out, "path": "text", "model": used_model}}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _prepare_async_payload(job_id: str, razred: str, user_text: str, requested: list, image_url: str | None, file_bytes: bytes | None, file_name: str | None, file_mime: str | None, image_b64_str: str | None) -> dict:
    payload = {
        "job_id": job_id, "razred": razred, "user_text": user_text, "requested": requested,
        "bucket": GCS_BUCKET, "image_path": None, "image_url": image_url or None,
        "image_inline_b64": None,
    }
    if file_bytes:
        if not LOCAL_MODE and (storage_client and GCS_BUCKET):
            path = gcs_upload_bytes(job_id, file_bytes, filename_hint=(file_name or "image.bin"), content_type=file_mime or "application/octet-stream")
            if path: payload["image_path"] = path
        else:
            payload["image_inline_b64"] = base64.b64encode(file_bytes).decode()
        return payload
    if image_b64_str:
        b64_clean = image_b64_str.split(",", 1)[1] if "," in image_b64_str else image_b64_str
        if not LOCAL_MODE and (storage_client and GCS_BUCKET):
            try:
                raw = base64.b64decode(b64_clean)
            except Exception:
                raw = b""
            path = gcs_upload_bytes(job_id, raw, filename_hint="image.bin", content_type="application/octet-stream")
            if path: payload["image_path"] = path
        else:
            payload["image_inline_b64"] = b64_clean
        return payload
    return payload

@app.route("/submit", methods=["POST", "OPTIONS"])
def submit():
    if request.method == "OPTIONS":
        return ("", 204)
    razred = (request.form.get("razred") or request.args.get("razred") or "").strip()
    user_text = (request.form.get("user_text") or request.form.get("pitanje") or "").strip()
    image_url = (request.form.get("image_url") or request.args.get("image_url") or "").strip()
    mode = (request.form.get("mode") or request.args.get("mode") or "auto").strip().lower()
    data = request.get_json(silent=True) or {}
    if data:
        razred    = (data.get("razred")    or razred).strip()
        user_text = (data.get("pitanje")   or data.get("user_text") or user_text).strip()
        image_url = (data.get("image_url") or image_url).strip()
        mode      = (data.get("mode")      or mode).strip().lower()
    if razred not in DOZVOLJENI_RAZREDI:
        razred = "5"

    requested = extract_requested_tasks(user_text)
    file_storage = request.files.get("file")
    file_bytes = None
    file_mime = None
    file_name = None
    if file_storage and file_storage.filename:
        file_bytes = file_storage.read()
        file_mime = file_storage.mimetype or "application/octet-stream"
        file_name = file_storage.filename
    image_b64_str = (data.get("image_b64") if data else None)
    has_image = bool(image_url or file_bytes or image_b64_str)

    # FOLLOW-UP u /submit: implicitno koristi prošlu sliku
    followup_flag = is_followup_like(user_text)
    last_kind = session.get("last_problem_kind")
    last_url = session.get("last_image_url")
    if (not has_image) and followup_flag and last_kind == "image" and last_url:
        image_url = last_url
        has_image = True

    if mode not in ("auto", "sync", "async"):
        mode = "auto"

    if mode == "async":
        job_id = str(uuid4())
        store_job(job_id, {"status": "pending", "created_at": datetime.datetime.utcnow().isoformat() + "Z", "razred": razred, "user_text": user_text, "requested": requested}, merge=True)
        payload = _prepare_async_payload(job_id, razred, user_text, requested, image_url or None, file_bytes, file_name, file_mime, image_b64_str)
        if image_url or file_bytes or image_b64_str:
            session["last_problem_kind"] = "image"
            if image_url:
                session["last_image_url"] = image_url
        else:
            session["last_problem_kind"] = "text"
            session["last_problem_user_text"] = user_text
        try:
            _enqueue(payload)
            return jsonify({"mode": "async", "job_id": job_id, "status": "queued", "local_mode": LOCAL_MODE}), 202
        except Exception as e:
            store_job(job_id, {"status": "error", "error": str(e)}, merge=True)
            return jsonify({"error": "submit_failed", "detail": str(e), "job_id": job_id}), 500

    heavy = looks_heavy(user_text, has_image=has_image)
    if mode == "auto" and heavy:
        job_id = str(uuid4())
        store_job(job_id, {"status": "pending", "created_at": datetime.datetime.utcnow().isoformat() + "Z", "razred": razred, "user_text": user_text, "requested": requested}, merge=True)
        payload = _prepare_async_payload(job_id, razred, user_text, requested, image_url or None, file_bytes, file_name, file_mime, image_b64_str)
        if image_url or file_bytes or image_b64_str:
            session["last_problem_kind"] = "image"
            if image_url:
                session["last_image_url"] = image_url
        else:
            session["last_problem_kind"] = "text"
            session["last_problem_user_text"] = user_text
        try:
            _enqueue(payload)
            return jsonify({"mode": "auto→async", "job_id": job_id, "status": "queued", "local_mode": LOCAL_MODE}), 202
        except Exception as e:
            store_job(job_id, {"status": "error", "error": str(e)}, merge=True)
            return jsonify({"error": "submit_failed", "detail": str(e)}), 500

    # sync pokušaj (+ follow-up prefiks ako je zadnji bio tekst)
    is_followup_text_now = bool(followup_flag and last_kind == "text")
    sync_try = _sync_process_once(
        razred=razred,
        user_text=user_text,
        requested=requested,
        image_url=(image_url or None),
        file_bytes=file_bytes,
        file_mime=file_mime,
        timeout_s=SYNC_SOFT_TIMEOUT_S,
        is_followup=is_followup_text_now,
        last_problem_text=session.get("last_problem_user_text", "")
    )
    if sync_try.get("ok"):
        html_out = sync_try["result"]["html"]
        if should_plot(user_text):
            expr = extract_plot_expression(user_text, razred=razred, history=[])
            if expr: html_out = add_plot_div_once(html_out, expr)
        try: log_to_sheet("sync-" + uuid4().hex[:8], razred, user_text, html_out, sync_try["result"]["path"], sync_try["result"]["model"])
        except Exception: pass
        mode_tag = "auto(sync)" if mode == "auto" else "sync"
        if image_url or file_bytes or image_b64_str:
            session["last_problem_kind"] = "image"
            if image_url:
                session["last_image_url"] = image_url
        else:
            session["last_problem_kind"] = "text"
            session["last_problem_user_text"] = user_text
        return jsonify({"mode": mode_tag, "result": {"html": html_out, "path": sync_try["result"]["path"], "model": sync_try["result"]["model"]}}), 200

    # fallback → async
    job_id = str(uuid4())
    store_job(job_id, {"status": "pending", "created_at": datetime.datetime.utcnow().isoformat() + "Z", "razred": razred, "user_text": user_text, "requested": requested}, merge=True)
    payload = _prepare_async_payload(job_id, razred, user_text, requested, image_url or None, file_bytes, file_name, file_mime, image_b64_str)
    if image_url or file_bytes or image_b64_str:
        session["last_problem_kind"] = "image"
        if image_url:
            session["last_image_url"] = image_url
    else:
        session["last_problem_kind"] = "text"
        session["last_problem_user_text"] = user_text
    try:
        _enqueue(payload)
        mode_tag = "auto(sync→async)" if mode == "auto" else "sync→async"
        return jsonify({"mode": mode_tag, "job_id": job_id, "status": "queued", "local_mode": LOCAL_MODE, "reason": sync_try.get("error", "soft-timeout-or-error")}), 202
    except Exception as e:
        store_job(job_id, {"status": "error", "error": str(e)}, merge=True)
        return jsonify({"error": "submit_failed", "detail": str(e), "job_id": job_id}), 500

@app.get("/status/<job_id>")
def async_status(job_id):
    data = read_job(job_id)
    if not data: return jsonify({"status": "pending"}), 200
    return jsonify(data), 200

@app.get("/result/<job_id>")
def async_result(job_id):
    data = read_job(job_id)
    if not data:
        return jsonify({"status": "pending"}), 202
    if data.get("status") == "done":
        return jsonify({"job_id": job_id, "result": data.get("result")}), 200
    if data.get("status") == "error":
        return jsonify({"job_id": job_id, "status": "error", "error": data.get("error")}), 500
    return jsonify({"job_id": job_id, "status": data.get("status", "pending")}), 202

@app.post("/tasks/process")
def tasks_process():
    if not LOCAL_MODE and request.headers.get("X-Tasks-Secret") != TASKS_SECRET:
        return "Forbidden", 403
    try:
        payload = request.get_json(force=True)
        job_id = payload["job_id"]
        out = _process_job_core(payload)
        store_job(job_id, out, merge=True)
        try:
            log_to_sheet(job_id, out.get("razred"), out.get("user_text"), out["result"]["html"], out["result"]["path"], out["result"]["model"])
        except Exception:
            pass
        return "OK", 200
    except Exception as e:
        err_html = "<p><b>Nije uspjela obrada.</b> Pokušaj ponovo ili pošalji jasniji unos.</p><p><code>" + html.escape(str(e)) + "</code></p>"
        job_id = (request.get_json(silent=True) or {}).get("job_id", "unknown-" + uuid4().hex[:6])
        store_job(job_id, {"status": "done", "result": {"html": err_html, "path": "error", "model": "n/a"}, "finished_at": datetime.datetime.utcnow().isoformat() + "Z"}, merge=True)
        return "OK", 200

@app.post("/set-razred")
def set_razred():
    g = (request.form.get("razred") or "").strip()
    if g:
        session["razred"] = g
        session["history"] = []
        session.pop("last_image_url", None)
        session.pop("last_problem_kind", None)
        session.pop("last_problem_user_text", None)
    return ("", 204)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    log.info("Starting app on port %s, LOCAL_MODE=%s", port, LOCAL_MODE)
    app.run(host="0.0.0.0", port=port, debug=debug)
