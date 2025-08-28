# app.py — Async TEXT + IMAGE (Cloud Tasks + Firestore + GCS) + LOCAL_MODE za razvoj (bez timeouta)
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify
from dotenv import load_dotenv
import os, re, base64, json, html, datetime, logging, mimetypes, threading, traceback
from datetime import timedelta
from uuid import uuid4

from openai import OpenAI
from flask_cors import CORS

import gspread
from google.oauth2.service_account import Credentials as SACreds
import google.auth

# --- Opcionalni GCP klijenti ---
try:
    from google.cloud import storage as gcs_lib  # type: ignore
except Exception:
    gcs_lib = None
try:
    from google.cloud import firestore as fs_lib  # type: ignore
except Exception:
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

# ------ MODE ------
LOCAL_MODE = os.getenv("LOCAL_MODE", "0") == "1"   # kad je 1 → nema Cloud Tasks / Firestore / GCS, sve ide lokalno
USE_FIRESTORE = os.getenv("USE_FIRESTORE", "1") == "1" and not LOCAL_MODE

MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "200"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- OpenAI ---
_OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not _OPENAI_API_KEY:
    log.error("OPENAI_API_KEY nije postavljen u okruženju.")
# Uklonjeni timeouti i custom retry-jevi
client = OpenAI(api_key=_OPENAI_API_KEY)

MODEL_VISION_LIGHT = os.getenv("OPENAI_MODEL_VISION_LIGHT") or os.getenv("OPENAI_MODEL_VISION", "gpt-5")
MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-5-mini")
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-5")

# ---------------- Sheets ----------------
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
        log.info("Sheets via service_account_b64 (sa=%s)", info.get("client_email"))
    elif os.path.exists("credentials.json"):
        creds = SACreds.from_service_account_file("credentials.json", scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        _sheets_diag["mode"] = "file"; _sheets_diag["sa_email"] = _try_get_sa_email_from_creds(creds)
        log.info("Sheets via service_account_file")
    else:
        adc_creds, _ = google.auth.default(scopes=SHEETS_SCOPES)
        gc = gspread.authorize(adc_creds)
        _sheets_diag["mode"] = "adc"; _sheets_diag["sa_email"] = _try_get_sa_email_from_creds(adc_creds)
        log.info("Sheets via ADC default credentials")

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
    log.info("Sheets enabled (title=%s id=%s ws=%s)", _sheets_diag["spreadsheet_title"], _sheets_diag["spreadsheet_id"], _sheets_diag["worksheet_title"])
except Exception as e:
    log.warning("Sheets disabled: %s", e)
    _sheets_diag["error"] = str(e); sheet = None

def sheets_append_row_safe(values):
    if not sheet: return False
    try:
        sheet.append_row(values, value_input_option="USER_ENTERED"); return True
    except Exception as e:
        log.warning("Sheets append error: %s", e); return False

def log_to_sheet(job_id, razred, user_text, odgovor_html, source_tag, model_name):
    ts = datetime.datetime.utcnow().isoformat()
    sheets_append_row_safe([ts, razred, user_text, odgovor_html, f"{source_tag}|{model_name}", job_id])

# ---------------- GCS ----------------
GCS_BUCKET = (os.getenv("GCS_BUCKET") or "").strip()
GCS_SIGNED_GET = os.getenv("GCS_SIGNED_GET", "1") == "1"
storage_client = None
if not LOCAL_MODE and GCS_BUCKET and gcs_lib is not None:
    try:
        storage_client = gcs_lib.Client()
        log.info("GCS enabled (bucket=%s, signed_get=%s)", GCS_BUCKET, GCS_SIGNED_GET)
    except Exception as e:
        log.error("GCS client init failed: %s", e); storage_client = None
else:
    if not GCS_BUCKET: log.info("GCS disabled (no GCS_BUCKET set or LOCAL_MODE=1).")
    elif gcs_lib is None: log.warning("google-cloud-storage lib not available, GCS disabled.")

# ---------------- Firestore or local store ----------------
fs_db = None
JOB_STORE = {}  # lokalni in-memory store kad Firestore nije u upotrebi

if USE_FIRESTORE and fs_lib is not None:
    try:
        fs_db = fs_lib.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT") or None)
        log.info("Firestore enabled.")
    except Exception as e:
        log.error("Firestore init failed: %s", e); fs_db = None

def store_job(job_id: str, data: dict, merge: bool = True):
    if fs_db:
        fs_db.collection("jobs").document(job_id).set(data, merge=merge)
    else:
        # merge simulacija
        JOB_STORE[job_id] = {**JOB_STORE.get(job_id, {}), **data}

def read_job(job_id: str) -> dict:
    if fs_db:
        doc = fs_db.collection("jobs").document(job_id).get()
        return (doc.to_dict() or {}) if doc.exists else {}
    return JOB_STORE.get(job_id, {})

# ===================== Helpers =====================
PROMPTI_PO_RAZREDU = {
    "5": "Ti si pomoćnik iz matematike za učenike 5. razreda osnovne škole. Objašnjavaj jednostavnim i razumljivim jezikom. Pomaži učenicima da razumiju zadatke iz prirodnih brojeva, osnovnih računskih operacija, jednostavne geometrije i tekstualnih zadataka. Svako rješenje objasni jasno, korak po korak.",
    "6": "Ti si pomoćnik iz matematike za učenike 6. razreda osnovne škole. Odgovaraj detaljno i pedagoški, koristeći primjere primjerene njihovom uzrastu. Pomaži im da razumiju razlomke, decimalne brojeve, procente, geometriju i tekstualne zadatke. Objasni rješenje jasno i korak po korak.",
    "7": "Ti si pomoćnik iz matematike za učenike 7. razreda osnovne škole. Pomaži im u razumijevanju složenijih zadataka iz algebre, geometrije i funkcija. Koristi jasan, primjeren jezik i objasni svaki korak logično i precizno.",
    "8": "Ti si pomoćnik iz matematike za učenike 8. razreda osnovne škole. Fokusiraj se na linearne izraze, sisteme jednačina, geometriju i statistiku. Pomaži učenicima da razumiju postupke i objasni svako rješenje detaljno, korak po korak.",
    "9": "Ti si pomoćnik iz matematike za učenike 9. razreda osnovne škole. Pomaži im u savladavanju zadataka iz algebre, funkcija, geometrije i statistike. Koristi jasan i stručan jezik, ali primjeren njihovom nivou. Objasni svaki korak rješenja jasno i precizno."
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
        return f"\\(\\frac{{{m.group(1)}}}{{{m.group(2)}}}\\)"
    return re.sub(r'\b(\d{1,4})/(\d{1,4})\b', zamijeni, text)

def add_plot_div_once(odgovor_html: str, expression: str) -> str:
    marker = f'class="plot-request"'
    expr_attr = f'data-expression="{html.escape(expression)}"'
    if (marker in odgovor_html) and (expr_attr in odgovor_html): return odgovor_html
    return odgovor_html + f'<div class="plot-request" data-expression="{html.escape(expression)}"></div>'

TRIGGER_PHRASES = [r"\bnacrtaj\b", r"\bnacrtati\b", r"\bcrtaj\b", r"\biscrtaj\b", r"\bskiciraj\b",
                   r"\bgraf\b", r"\bgrafik\b", r"\bprika[žz]i\s+graf\b", r"\bplot\b", r"\bvizualizuj\b", r"\bnasrtaj\b"]
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

# ===== OpenAI helper (bez timeouta) =====
def _openai_chat(model: str, messages: list, max_tokens: int | None = None):
    params = {"model": model, "messages": messages}
    if max_tokens is not None:
        params["max_completion_tokens"] = max_tokens
    try:
        return client.chat.completions.create(**params)
    except Exception as e:
        msg = str(e)
        if "max_completion_tokens" in msg or "Unsupported parameter: 'max_completion_tokens'" in msg:
            params.pop("max_completion_tokens", None)
            if max_tokens is not None: params["max_tokens"] = max_tokens
            return client.chat.completions.create(**params)
        raise

# ===== Pipelines =====
def answer_with_text_pipeline(pure_text: str, razred: str, history, requested):
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])
    only_clause = ""
    strict_geom_policy = (" Ako problem uključuje geometriju: "
                          "1) koristi samo eksplicitno date podatke; "
                          "2) ne pretpostavljaj ništa bez oznake; "
                          "3) navedi nazive teorema (npr. unutrašnji naspramni, Thales...).")
    if requested:
        only_clause = (" Riješi ISKLJUČIVO sljedeće zadatke: " + ", ".join(map(str, requested)) +
                       ". Sve ostale primjere ignoriraj.")
    system_message = {
        "role": "system",
        "content": (prompt_za_razred +
                    " Odgovaraj jezikom pitanja (po difoltu bosanski ijekavica). "
                    "Ako pitanje nije iz matematike, reci: 'Molim te, postavi matematičko pitanje.' "
                    "Ako ne znaš tačno rješenje, reci: 'Za ovaj zadatak se obrati instruktorima na info@matematicari.com'. "
                    "Ne crtati ASCII grafove osim ako je traženo." + only_clause + strict_geom_policy)
    }
    messages = [system_message]
    for msg in history[-5:]:
        messages.append({"role":"user","content": msg["user"]})
        messages.append({"role":"assistant","content": msg["bot"]})
    messages.append({"role":"user","content": pure_text})
    response = _openai_chat(MODEL_TEXT, messages)
    actual_model = getattr(response, "model", MODEL_TEXT)
    raw = response.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    html_out = f"<p>{latexify_fractions(raw)}</p>"
    return html_out, actual_model

def _vision_messages_base(razred: str, history, only_clause: str, strict_geom_policy: str):
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])
    system_message = {
        "role": "system",
        "content": (prompt_za_razred +
                    " Odgovaraj jezikom pitanja (bosanski/ijekavica). "
                    "Ne prikazuj ASCII grafove osim ako su izričito traženi. " +
                    only_clause + " " + strict_geom_policy)
    }
    messages = [system_message]
    for msg in history[-5:]:
        messages.append({"role": "user", "content": msg["user"]})
        messages.append({"role": "assistant", "content": msg["bot"]})
    return messages

def _vision_clauses():
    return "", " Radi tačno i oprezno. Ako nešto nedostaje, navedi šta nedostaje i stani."

# ---- MIME sniff bez imghdr ----
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
    return f"data:{mime};base64,{b64}"

def route_image_flow_url(image_url: str, razred: str, history, user_text=None):
    only_clause, strict_geom_policy = _vision_clauses()
    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)
    user_content = []
    if user_text: user_content.append({"type": "text", "text": f"Korisnički tekst: {user_text}"})
    user_content.append({"type": "text", "text": "Na slici je matematički zadatak."})
    user_content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages.append({"role": "user", "content": user_content})
    resp = _openai_chat(MODEL_VISION, messages)
    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{latexify_fractions(raw)}</p>", "vision_url", actual_model

def route_image_flow(slika_bytes: bytes, razred: str, history, user_text=None, mime_hint: str | None = None):
    only_clause, strict_geom_policy = _vision_clauses()
    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)
    data_url = _bytes_to_data_url(slika_bytes, mime_hint=mime_hint)
    user_content = []
    if user_text: user_content.append({"type": "text", "text": f"Korisnički tekst: {user_text}"})
    user_content.append({"type": "text", "text": "Na slici je matematički zadatak."})
    user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    messages.append({"role": "user", "content": user_content})
    resp = _openai_chat(MODEL_VISION, messages)
    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{latexify_fractions(raw)}</p>", "vision_direct", actual_model

# ===== Request helpers =====
def get_history_from_request():
    try:
        hx = request.form.get("history_json")
        if hx:
            data = json.loads(hx)
            if isinstance(data, list): return data
    except Exception:
        pass
    return None

# ===== Utils =====
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

# ===================== GCS upload helper (samo u cloud modu) =====================
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
            url = blob.generate_signed_url(version="V4", expiration=datetime.timedelta(minutes=45), method="GET")
        else:
            try: blob.make_public(); url = blob.public_url
            except Exception: url = blob.generate_signed_url(version="V4", expiration=datetime.timedelta(minutes=45), method="GET")
        return url
    except Exception as e:
        log.error("GCS upload failed: %s", e); return None

# ===================== Routes: sync forma (nebitno za lokal test) =====================
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

            if image_url:
                combined_text = pitanje
                odgovor, used_path, used_model = route_image_flow_url(image_url, razred, history, user_text=combined_text)
                session["last_image_url"] = image_url
                if (not plot_expression_added) and should_plot(combined_text):
                    expr = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True
                display_user = (combined_text + " [slika]") if combined_text else "[slika]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sync_job_id = f"sync-{uuid4().hex[:8]}"; log_to_sheet(sync_job_id, razred, combined_text, odgovor, used_path, used_model)
                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            if slika and slika.filename:
                combined_text = pitanje
                body = slika.read()
                odgovor, used_path, used_model = route_image_flow(body, razred, history, user_text=combined_text, mime_hint=slika.mimetype or None)
                try:
                    ext = os.path.splitext(slika.filename or "")[1].lower() or ".img"
                    fname = f"{uuid4().hex}{ext}"
                    with open(os.path.join(UPLOAD_DIR, fname), "wb") as fp: fp.write(body)
                    public_url = (request.url_root.rstrip("/") + "/uploads/" + fname)
                    session["last_image_url"] = public_url
                except Exception as _e:
                    log.warning("Couldn't persist small image copy: %s", _e)
                if (not plot_expression_added) and should_plot(combined_text):
                    expr = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True
                display_user = (combined_text + " [SLIKA]") if combined_text else "[SLIKA]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sync_job_id = f"sync-{uuid4().hex[:8]}"; log_to_sheet(sync_job_id, razred, combined_text, odgovor, used_path, used_model)
                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            requested = extract_requested_tasks(pitanje)
            last_url = session.get("last_image_url")
            if last_url and (requested or (pitanje and FOLLOWUP_TASK_RE.match(pitanje))):
                odgovor, used_path, used_model = route_image_flow_url(last_url, razred, history, user_text=pitanje)
                if (not plot_expression_added) and should_plot(pitanje):
                    expr = extract_plot_expression(pitanje, razred=razred, history=history)
                    if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True
                history.append({"user": pitanje, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sync_job_id = f"sync-{uuid4().hex[:8]}"; log_to_sheet(sync_job_id, razred, pitanje, odgovor, used_path, used_model)
                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            odgovor, actual_model = answer_with_text_pipeline(pitanje, razred, history, requested)
            if (not plot_expression_added) and should_plot(pitanje):
                expr = extract_plot_expression(pitanje, razred=razred, history=history)
                if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True
            history.append({"user": pitanje, "bot": odgovor.strip()}); history = history[-8:]; session["history"] = history
            sync_job_id = f"sync-{uuid4().hex[:8]}"; log_to_sheet(sync_job_id, razred, pitanje, odgovor, "text", actual_model)

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
        session.pop("history", None); session.pop("razred", None); session.pop("last_image_url", None)
    if request.form.get("ajax") == "1": return render_template("index.html", history=[], razred=None)
    return redirect("/")

@app.get("/healthz")
def healthz(): return {"ok": True, "local_mode": LOCAL_MODE}, 200
@app.get("/_healthz")
def _healthz(): return {"ok": True}, 200
@app.get("/_ah/health")
def ah_health(): return "OK", 200

# ---- Sheets dijagnostika ----
@app.get("/sheets/diag")
def sheets_diag(): return jsonify(_sheets_diag), 200

@app.post("/sheets/selftest")
def sheets_selftest():
    if not sheet:
        return jsonify({"ok": False, "error": _sheets_diag.get("error") or "Sheets not initialized"}), 500
    row = [datetime.datetime.utcnow().isoformat(), "selftest", "Hello from /sheets/selftest", "<p>OK</p>", "selftest|none", f"self-{uuid4().hex[:8]}"]
    ok = sheets_append_row_safe(row); return jsonify({"ok": ok}), (200 if ok else 500)

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    resp.headers["Pragma"] = "no-cache"; resp.headers["Expires"] = "0"; resp.headers["Vary"] = "Cookie"
    ancestors = os.getenv("FRAME_ANCESTORS", "").strip()
    if ancestors: resp.headers["Content-Security-Policy"] = f"frame-ancestors {ancestors}"
    try: del resp.headers["X-Frame-Options"]
    except KeyError: pass
    return resp

# ===================== ASINHRONO: Cloud Tasks + Firestore + GCS =====================
PROJECT_ID        = (os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or "").strip()
REGION            = os.getenv("REGION", "europe-west1")
TASKS_QUEUE       = os.getenv("TASKS_QUEUE", "matbot-queue")
TASKS_TARGET_URL  = os.getenv("TASKS_TARGET_URL")  # npr. https://<run-url>/tasks/process
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

# --- Core worker logika (zajednička i za cloud i za lokalni thread) ---
def _process_job_core(payload: dict) -> dict:
    job_id     = payload["job_id"]
    bucket     = payload.get("bucket")
    image_path = payload.get("image_path")
    image_url  = payload.get("image_url")
    image_inline_b64 = payload.get("image_inline_b64")  # samo za LOCAL_MODE
    razred     = (payload.get("razred") or "").strip()
    user_text  = (payload.get("user_text") or "").strip()
    requested  = payload.get("requested") or []
    if razred not in DOZVOLJENI_RAZREDI: razred = "5"

    log.info("worker start job_id=%s image_path=%s image_url=%s local_inline=%s",
             job_id, image_path, (image_url[:60] + "...") if image_url else None,
             "yes" if bool(image_inline_b64) else "no")

    history = []

    # Glavni tok (bez bilo kakvih timeout argumenata)
    if image_path:
        if not storage_client:
            raise RuntimeError("GCS storage client not initialized (image_path zadat).")
        blob = storage_client.bucket(bucket).blob(image_path)
        img_bytes = blob.download_as_bytes()
        mime_hint = blob.content_type or mimetypes.guess_type(image_path)[0] or None
        odgovor_html, used_path, used_model = route_image_flow(img_bytes, razred, history=history, user_text=user_text, mime_hint=mime_hint)
    elif image_inline_b64:
        img_bytes = base64.b64decode(image_inline_b64)
        odgovor_html, used_path, used_model = route_image_flow(img_bytes, razred, history=history, user_text=user_text, mime_hint=None)
    elif image_url:
        odgovor_html, used_path, used_model = route_image_flow_url(image_url, razred, history=history, user_text=user_text)
    else:
        odgovor_html, used_model = answer_with_text_pipeline(user_text, razred, history, requested)
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

# --- Lokalni worker thread ---
def _local_worker(payload: dict):
    job_id = payload["job_id"]
    try:
        out = _process_job_core(payload)
        store_job(job_id, out, merge=True)
        try: log_to_sheet(job_id, out.get("razred"), out.get("user_text"), out["result"]["html"], out["result"]["path"], out["result"]["model"])
        except Exception as _e: log.warning("Sheets log fail: %s", _e)
    except Exception as e:
        log.error("Local worker failed: %s\n%s", e, traceback.format_exc())
        err_html = ("<p><b>Nije uspjela obrada.</b> Pokušaj ponovo ili pošalji jasniji unos.</p>"
                    f"<p><code>{html.escape(str(e))}</code></p>")
        store_job(job_id, {"status": "done", "result": {"html": err_html, "path": "error", "model": "n/a"},
                           "finished_at": datetime.datetime.utcnow().isoformat() + "Z"}, merge=True)

# --------- SUBMIT (radi lokalno i u cloudu) ----------
@app.route("/submit", methods=["POST", "OPTIONS"])
def submit_async():
    if request.method == "OPTIONS":
        return ("", 204)

    razred = (request.form.get("razred") or request.args.get("razred") or "").strip()
    user_text = (request.form.get("user_text") or request.form.get("pitanje") or "").strip()
    image_url = (request.form.get("image_url") or request.args.get("image_url") or "").strip()

    data = request.get_json(silent=True) or {}
    if data:
        razred    = (data.get("razred")    or razred).strip()
        user_text = (data.get("pitanje")   or data.get("user_text") or user_text).strip()
        image_url = (data.get("image_url") or image_url).strip()

    requested = extract_requested_tasks(user_text)

    job_id = str(uuid4())

    # inicijalno upiši pending
    store_job(job_id, {"status": "pending", "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                       "razred": razred, "user_text": user_text, "requested": requested}, merge=True)

    payload = {
        "job_id": job_id, "razred": razred, "user_text": user_text, "requested": requested,
        "bucket": GCS_BUCKET, "image_path": None, "image_url": image_url or None,
        "image_inline_b64": None,
    }

    # FILE -> GCS (cloud) ili inline (local)
    if "file" in request.files:
        f = request.files["file"]
        if not LOCAL_MODE and (storage_client and GCS_BUCKET):
            name = f"uploads/{job_id}/{f.filename or 'image.bin'}"
            bucket = storage_client.bucket(GCS_BUCKET)
            blob = bucket.blob(name)
            blob.upload_from_file(f, content_type=f.mimetype or "application/octet-stream")
            payload["image_path"] = name
        else:
            raw = f.read()
            payload["image_inline_b64"] = base64.b64encode(raw).decode()

    else:
        image_b64 = (data.get("image_b64") if data else None)
        if image_b64:
            if "," in image_b64: image_b64 = image_b64.split(",", 1)[1]
            if not LOCAL_MODE and (storage_client and GCS_BUCKET):
                raw = base64.b64decode(image_b64)
                name = f"uploads/{job_id}/image.bin"
                bucket = storage_client.bucket(GCS_BUCKET)
                blob = bucket.blob(name)
                blob.upload_from_string(raw, content_type="application/octet-stream")
                payload["image_path"] = name
            else:
                payload["image_inline_b64"] = image_b64

    try:
        if LOCAL_MODE:
            threading.Thread(target=_local_worker, args=(payload,), daemon=True).start()
            return jsonify({"job_id": job_id, "status": "queued", "local_mode": True}), 202
        else:
            _create_task_cloud(payload)
            return jsonify({"job_id": job_id, "status": "queued", "local_mode": False}), 202
    except Exception as e:
        # Vrati detaljnu grešku umjesto "Internal Server Error"
        log.error("submit_async failed: %s\n%s", e, traceback.format_exc())
        store_job(job_id, {"status": "error", "error": str(e)}, merge=True)
        return jsonify({
            "error": "submit_failed",
            "detail": str(e),
            "hint": "Lokalno koristi LOCAL_MODE=1 (bez Cloud Tasks/Firestore/GCS) ili provjeri GCP kredencijale / queue.",
            "job_id": job_id
        }), 500

# --------- STATUS ----------
@app.get("/status/<job_id>")
def async_status(job_id):
    data = read_job(job_id)
    if not data: return jsonify({"status": "pending"}), 200
    return jsonify(data), 200

# --------- Cloud Tasks endpoint ----------
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
        except Exception as _e:
            log.warning("Sheets log fail: %s", _e)
        return "OK", 200
    except Exception as e:
        log.exception("Task processing failed")
        err_html = ("<p><b>Nije uspjela obrada.</b> Pokušaj ponovo ili pošalji jasniji unos.</p>"
                    f"<p><code>{html.escape(str(e))}</code></p>")
        job_id = (request.get_json(silent=True) or {}).get("job_id", f"unknown-{uuid4().hex[:6]}")
        store_job(job_id, {"status": "done", "result": {"html": err_html, "path": "error", "model": "n/a"},
                           "finished_at": datetime.datetime.utcnow().isoformat() + "Z"}, merge=True)
        return "OK", 200  # bez retrija

# ===================== Run =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    log.info("Starting app on port %s, LOCAL_MODE=%s", port, LOCAL_MODE)
    app.run(host="0.0.0.0", port=port, debug=debug)
