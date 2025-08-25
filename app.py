# app.py — sve ide preko OpenAI API (bez lokalnog evaluatora), Cloud Run friendly

from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify
from dotenv import load_dotenv
import os, re, base64, json, html, datetime, logging, traceback
from datetime import timedelta
from uuid import uuid4

from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from flask_cors import CORS

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# httpx je dependency OpenAI SDK-a (za robusnije konekcije / timeoute)
import httpx

# --- (opcionalno) HTML sanitizacija; ako bleach nije instaliran, safe_html je NO-OP ---
try:
    import bleach
    ALLOWED_TAGS = ["p","b","i","em","strong","u","sub","sup","ul","ol","li","span","br","code","pre","hr"]
    ALLOWED_ATTRS = {"span": ["class"], "p": ["class"]}
    def safe_html(s: str) -> str:
        return bleach.clean(s, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
except Exception:
    def safe_html(s: str) -> str:
        return s

# --- GCP / GCS helpers (keyless signing) ---
try:
    from google.cloud import storage
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
except Exception:
    storage = None

load_dotenv(override=True)

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
)
CORS(app, supports_credentials=True)
app.secret_key = os.getenv("SECRET_KEY", "tajna_lozinka")

MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "240"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

# Stabilniji http klijent (poželjno na Cloud Run-u)
http_client = httpx.Client(
    http2=False,
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
    timeout=httpx.Timeout(connect=10, read=OPENAI_TIMEOUT, write=30, pool=OPENAI_TIMEOUT),
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=OPENAI_TIMEOUT,
    max_retries=OPENAI_MAX_RETRIES,
    http_client=http_client,
)

MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-5-mini")
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-4o")

# --- Google Sheets (optional) ---
try:
    SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    CREDS_FILE = "credentials.json"
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    gs_client = gspread.authorize(creds)
    sheet = gs_client.open("matematika-bot").sheet1
except Exception as e:
    log.warning("Sheets disabled: %s", e)
    sheet = None

# --- GCS config ---
GCS_BUCKET = (os.getenv("GCS_BUCKET") or "").strip()
GCS_SIGNED_GET = os.getenv("GCS_SIGNED_GET", "1") == "1"
GCS_REQUIRED = os.getenv("GCS_REQUIRED", "1") == "1"
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
    "5": "Ti si pomoćnik iz matematike za učenike 5. razreda osnovne škole. Objašnjavaj jednostavnim i razumljivim jezikom. Pomaži učenicima da razumiju zadatke iz prirodnih brojeva, osnovnih računskih operacija, jednostavne geometrije i tekstualnih zadataka. Svako rješenje objasni jasno, korak po korak.",
    "6": "Ti si pomoćnik iz matematike za učenike 6. razreda osnovne škole. Odgovaraj detaljno i pedagoški, koristeći primjere primjerene njihovom uzrastu. Pomaži im da razumiju razlomke, decimalne brojeve, procente, geometriju i tekstualne zadatke. Objasni rješenje jasno i korak po korak.",
    "7": "Ti si pomoćnik iz matematike za učenike 7. razreda osnovne škole. Pomaži im u razumijevanju složenijih zadataka iz algebre, geometrije i funkcija. Koristi jasan, primjeren jezik i objasni svaki korak logično i precizno.",
    "8": "Ti si pomoćnik iz matematike za učenike 8. razreda osnovne škole. Fokusiraj se na linearne izraze, sisteme jednačina, geometriju i statistiku. Pomaži učenicima da razumiju postupke i objasni svako rješenje detaljno, korak po korak.",
    "9": "Ti si pomoćnik iz matematike za učenike 9. razreda osnovne škole. Pomaži im u savladavanju zadataka iz algebre, funkcija, geometrije i statistike. Koristi jasan i stručan jezik, ali primjeren njihovom nivou. Objasni svaki korak rješenja jasno i precizno."
}
DOZVOLJENI_RAZREDI = set(PROMPTI_PO_RAZREDU.keys())

ORDINAL_WORDS = {
    "prvi": 1, "drugi": 2, "treći": 3, "treci": 3, "četvrti": 4, "cetvrti": 4,
    "peti": 5, "šesti": 6, "sesti": 6, "sedmi": 7, "osmi": 8, "deveti": 9, "deseti": 10
}
_task_num_re = re.compile(
    r"(?:zadatak\s*(?:broj\s*)?(\d{1,2}))|(?:\b(\d{1,2})\s*\.)|(?:\b(" + "|".join(ORDINAL_WORDS.keys()) + r")\b)",
    flags=re.IGNORECASE
)

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
        if n and n not in seen:
            out.append(n); seen.add(n)
    return out

def latexify_fractions(text):
    def zamijeni(match):
        brojilac, imenilac = match.groups()
        return f"\\(\\frac{{{brojilac}}}{{{imenilac}}}\\)"
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

# ===================== OpenAI helpers =====================

def _compat_params(model: str, max_out: int = 800, temperature: float = 0.2, seed: int | None = 1234):
    """Kompat parametri: gpt-5 koristi max_completion_tokens, stariji koriste max_tokens."""
    params = {"model": model, "temperature": temperature}
    if seed is not None:
        params["seed"] = seed
    if "gpt-5" in model:
        params["max_completion_tokens"] = max_out
    else:
        params["max_tokens"] = max_out
    return params

def log_openai_error(e, ctx=""):
    rid = None; status = None; body = None
    try:
        resp = getattr(e, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
            try: rid = resp.headers.get("x-request-id")
            except Exception: pass
            try: body = resp.text[:800]
            except Exception: body = None
    except Exception:
        pass
    log.error("OpenAI/HTTP error [%s]: %s | status=%s | req_id=%s | body=%s",
              ctx, e.__class__.__name__, status, rid, body)
    log.error("Traceback: %s", traceback.format_exc())

def safe_llm_chat(model: str, messages: list, timeout: float | None = None,
                  max_out: int = 800, temperature: float = 0.2, seed: int | None = 1234):
    """Siguran chat poziv — vraća response ili None (ne diže 500)."""
    try:
        cli = client if timeout is None else client.with_options(timeout=timeout)
        params = _compat_params(model, max_out=max_out, temperature=temperature, seed=seed)
        params["messages"] = messages
        return cli.chat.completions.create(**params)
    except (APIConnectionError, APIStatusError, RateLimitError,
            httpx.TimeoutException, httpx.ConnectError,
            httpx.RemoteProtocolError, httpx.TooManyRedirects) as e:
        log_openai_error(e, ctx="chat.completions")
        return None
    except Exception as e:
        log_openai_error(e, ctx="chat.completions-unknown")
        return None

# ===================== Vision flows =====================

def _vision_messages_base(razred: str, history, only_clause: str, strict_geom_policy: str):
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])
    system_message = {
        "role": "system",
        "content": (
            prompt_za_razred +
            " Odgovaraj na jeziku pitanja; ako nisi siguran, koristi bosanski (ijekavica). "
            "Ne miješaj jezike i ne koristi engleske riječi u objašnjenjima. "
            "Ako nije matematika, reci: 'Molim te, postavi matematičko pitanje.' "
            "Ako ne znaš tačno rješenje, reci: 'Za ovaj zadatak se obrati instruktorima na info@matematicari.com'. "
            "Ne prikazuj ASCII grafove osim ako su izričito traženi. "
            + only_clause + " " + strict_geom_policy
        )
    }
    messages = [system_message]
    for msg in history[-5:]:
        messages.append({"role": "user", "content": msg["user"]})
        messages.append({"role": "assistant", "content": msg["bot"]})
    return messages

def _vision_clauses(requested_tasks):
    only_clause = ""
    if requested_tasks:
        only_clause = (
            " Riješi ISKLJUČIVO sljedeće zadatke (ignoriši sve ostale koji su vidljivi na slici), "
            f"tačno ove brojeve: {', '.join(map(str, requested_tasks))}. "
            "Ako ne možeš jasno identificirati tražene brojeve na slici, napiši: "
            "'Ne mogu izolovati traženi broj zadatka na slici.' i stani."
        )
    strict_geom_policy = (
        " Radi tačno i oprezno:\n"
        "1) PRVO, jasno prepiši koje uglove i oznake SI PROČITAO sa slike (npr. 53°, 65°, 98°).\n"
        "   Ako ijedan broj nije jasan, napiši: 'Ne mogu pouzdano pročitati podatke – napiši ih tekstom.' i stani.\n"
        "2) Nemoj pretpostavljati paralelnost, jednakokrakost, jednake uglove ili sličnost trokuta ako to nije eksplicitno označeno.\n"
        "3) Ako korisnik uz sliku da tekstualne podatke, ONI SU ISTINITI i imaju prioritet nad onim što vidiš.\n"
        "4) Daj konačan odgovor za traženi ugao x u stepenima i kratko objašnjenje (2–4 rečenice)."
    )
    return only_clause, strict_geom_policy

def route_image_flow_url(image_url: str, razred: str, history, requested_tasks=None):
    only_clause, strict_geom_policy = _vision_clauses(requested_tasks)
    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)

    user_content = [{"type": "text", "text": "Na slici je matematički zadatak."}]
    if requested_tasks:
        user_content[0]["text"] += f" Riješi isključivo zadatak(e): {', '.join(map(str, requested_tasks))}."
    else:
        user_content[0]["text"] += " Riješi samo ono što korisnik izričito traži."
    user_content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages.append({"role": "user", "content": user_content})

    resp = safe_llm_chat(MODEL_VISION, messages, timeout=OPENAI_TIMEOUT)
    if resp is None:
        return (
            "<p><b>Greška:</b> Desio se problem pri analizi slike. "
            "Pošalji nam sliku i broj zadatka na "
            "<a href='mailto:info@matematicari.com'>info@matematicari.com</a>.</p>",
            "vision_error", "n/a"
        )

    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{latexify_fractions(safe_html(raw))}</p>", "vision_url", actual_model

def route_image_flow(slika_bytes: bytes, razred: str, history, requested_tasks=None):
    image_b64 = base64.b64encode(slika_bytes).decode()
    only_clause, strict_geom_policy = _vision_clauses(requested_tasks)
    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)

    user_content = [{"type": "text", "text": "Na slici je matematički zadatak."}]
    if requested_tasks:
        user_content[0]["text"] += f" Riješi isključivo zadatak(e): {', '.join(map(str, requested_tasks))}."
    else:
        user_content[0]["text"] += " Riješi samo ono što korisnik izričito traži."
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    messages.append({"role": "user", "content": user_content})

    resp = safe_llm_chat(MODEL_VISION, messages, timeout=OPENAI_TIMEOUT)
    if resp is None:
        return (
            "<p><b>Greška:</b> Servis sporo odgovara ili je zauzet. "
            "Pošalji nam sliku i broj zadatka na "
            "<a href='mailto:info@matematicari.com'>info@matematicari.com</a>.</p>",
            "vision_direct_error", "n/a"
        )

    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{latexify_fractions(safe_html(raw))}</p>", "vision_direct", actual_model

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

# =============== GCS helpers (keyless signed URLs) ===============

def _gcs_credentials_for_signing():
    if storage is None:
        raise RuntimeError("GCS libs not available")
    creds, project_id = google.auth.default()
    creds.refresh(GoogleAuthRequest())
    svc_email = getattr(creds, "service_account_email", None) or os.getenv("GOOGLE_SERVICE_ACCOUNT_EMAIL")
    if not svc_email:
        raise RuntimeError("No service_account_email available for signing; set GOOGLE_SERVICE_ACCOUNT_EMAIL or use a SA on Cloud Run.")
    return creds, svc_email

def _signed_url(blob, method: str, minutes: int = 15, content_type: str | None = None):
    creds, svc_email = _gcs_credentials_for_signing()
    params = dict(
        version="v4",
        expiration=datetime.timedelta(minutes=minutes),
        method=method,
        service_account_email=svc_email,
        access_token=creds.token,
    )
    if content_type:
        params["content_type"] = content_type
    return blob.generate_signed_url(**params)

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
            url = _signed_url(blob, method="GET", minutes=45)
        else:
            try:
                blob.make_public()
                url = blob.public_url
            except Exception:
                url = _signed_url(blob, method="GET", minutes=45)
        return url
    except Exception as e:
        log.error("GCS upload failed: %s", e)
        return None

def gcs_upload_bytes(data: bytes, content_type: str = "image/jpeg", ext: str = ".jpg"):
    if not (storage_client and GCS_BUCKET):
        return None
    try:
        blob_name = f"uploads/{uuid4().hex}{ext}"
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        if GCS_SIGNED_GET:
            return _signed_url(blob, method="GET", minutes=45)
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return _signed_url(blob, method="GET", minutes=45)
    except Exception as e:
        log.warning("gcs_upload_bytes failed: %s", e)
        return None

# ===================== Reuse zadnje slike =====================

TASK_ONLY_RE = re.compile(
    r'^\s*(uradi|odradi|riješi|rijesi)?\s*(zadatak|broj)?\s*(' + r'\d{1,2}|' + "|".join(ORDINAL_WORDS.keys()) + r')\s*$',
    flags=re.IGNORECASE
)
def looks_like_task_only(text: str) -> bool:
    return bool(TASK_ONLY_RE.search(text or ""))

def remember_last_image(url: str):
    if url:
        session["last_image_url"] = url
        session.modified = True

# ===================== Routes =====================

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

        upload_url = _signed_url(blob, method="PUT", minutes=15, content_type=content_type)
        if GCS_SIGNED_GET:
            read_url = _signed_url(blob, method="GET", minutes=45)
        else:
            try:
                blob.make_public()
                read_url = blob.public_url
            except Exception:
                read_url = _signed_url(blob, method="GET", minutes=45)

        return jsonify({"uploadUrl": upload_url, "readUrl": read_url, "object": blob_name})
    except Exception as e:
        log.error("signed-upload error: %s", e)
        return jsonify({"error": "failed to create signed url", "detail": str(e)}), 500

@app.get("/uploads/<name>")
def uploads(name):
    return send_from_directory(UPLOAD_DIR, name)

@app.get("/favicon.ico")
def favicon():
    return ("", 204)

FALLBACK_HTML = (
    "<p><b>Greška:</b> Imamo tehnički problem pri obradi. "
    "Pošalji nam sliku i tačan broj zadatka na "
    "<a href='mailto:info@matematicari.com'>info@matematicari.com</a> "
    "i odgovorićemo u roku od 24h.</p>"
)

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

            # --- IMAGE VIA URL ---
            if image_url:
                combined_text = pitanje
                requested = extract_requested_tasks(combined_text)
                odgovor, used_path, used_model = route_image_flow_url(
                    image_url, razred, history, requested_tasks=requested
                )
                remember_last_image(image_url)

                if (not plot_expression_added) and should_plot(combined_text):
                    expression = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expression:
                        odgovor = add_plot_div_once(odgovor, expression); plot_expression_added = True

                history.append({"user": combined_text if combined_text else "[SLIKA-URL]", "bot": odgovor.strip()})
                history = history[-8:]
                session["history"] = history

                try:
                    if sheet:
                        mod_str = f"{used_path}|{used_model}"
                        sheet.append_row([combined_text if combined_text else "[SLIKA-URL]", odgovor, mod_str])
                except Exception as ee:
                    log.warning("Sheets append error: %s", ee)

                if is_ajax:
                    return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # --- IMAGE VIA FILE UPLOAD ---
            if slika and slika.filename:
                slika.stream.seek(0, os.SEEK_END)
                size_bytes = slika.stream.tell()
                slika.stream.seek(0)

                combined_text = pitanje
                requested = extract_requested_tasks(combined_text)

                if size_bytes <= 1_500_000:
                    body = slika.read()
                    odgovor, used_path, used_model = route_image_flow(
                        body, razred, history, requested_tasks=requested
                    )
                    # zapamti URL (best effort) da kasnije može "uradi 3"
                    guessed_ext = os.path.splitext(slika.filename or "")[1].lower() or ".jpg"
                    last_url = gcs_upload_bytes(body, content_type=slika.mimetype or "image/jpeg", ext=guessed_ext)
                    if last_url:
                        remember_last_image(last_url)
                else:
                    if not (storage_client and GCS_BUCKET) and (GCS_REQUIRED or os.getenv("K_SERVICE")):
                        return render_template("index.html", history=history, razred=razred,
                                               error="GCS nije konfigurisan – upload velikih slika nije moguć."), 400
                    gcs_url = gcs_upload_filestorage(slika)
                    if gcs_url:
                        remember_last_image(gcs_url)
                        odgovor, used_path, used_model = route_image_flow_url(
                            gcs_url, razred, history, requested_tasks=requested
                        )
                    else:
                        if GCS_REQUIRED or os.getenv("K_SERVICE"):
                            return render_template("index.html", history=history, razred=razred,
                                                   error="GCS upload nije uspio (CORS/permisije)."), 400
                        ext = os.path.splitext(slika.filename)[1].lower() or ".jpg"
                        fname = f"{uuid4().hex}{ext}"
                        path  = os.path.join(UPLOAD_DIR, fname)
                        slika.stream.seek(0)
                        slika.save(path)
                        public_url = (request.url_root.rstrip("/") + "/uploads/" + fname)
                        log.info("Falling back to /uploads URL: %s", public_url)
                        remember_last_image(public_url)
                        odgovor, used_path, used_model = route_image_flow_url(
                            public_url, razred, history, requested_tasks=requested
                        )

                if (not plot_expression_added) and should_plot(combined_text):
                    expression = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expression:
                        odgovor = add_plot_div_once(odgovor, expression); plot_expression_added = True

                history.append({"user": combined_text if combined_text else "[SLIKA]", "bot": odgovor.strip()})
                history = history[-8:]
                session["history"] = history

                try:
                    if sheet:
                        mod_str = f"{used_path}|{used_model}"
                        sheet.append_row([combined_text if combined_text else "[SLIKA]", odgovor, mod_str])
                except Exception as ee:
                    log.warning("Sheets append error: %s", ee)

                if is_ajax:
                    return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # --- PURE TEXT (UVJEK PREKO OPENAI) ---
            prompt_za_razred = PROMPTI_PO_RAZREDU[razred]
            requested = extract_requested_tasks(pitanje)
            only_clause = ""
            strict_geom_policy_text = (
                " Ako problem uključuje geometriju iz slike ili teksta: "
                "1) koristi samo eksplicitno date podatke; "
                "2) ne pretpostavljaj paralelnost/jednakokrakost bez oznake; "
                "3) navedi nazive teorema (unutrašnji naspramni, vanjski ugao, Thales, itd.)."
            )
            if requested:
                only_clause = (
                    " Riješi ISKLJUČIVO sljedeće zadatke: " + ", ".join(map(str, requested)) +
                    ". Sve ostale primjere u poruci ili slici ignoriraj."
                )

            system_message = {
                "role": "system",
                "content": (
                    prompt_za_razred +
                    " Odgovaraj na jeziku na kojem je pitanje postavljeno. Ako nisi siguran, koristi bosanski. "
                    "Ne miješaj jezike i ne koristi engleske riječi u objašnjenjima. "
                    "Uvijek koristi ijekavicu. Ako pitanje nije iz matematike, reci: 'Molim te, postavi matematičko pitanje.' "
                    "Ako ne znaš tačno rješenje, reci: 'Za ovaj zadatak se obrati instruktorima na info@matematicari.com'. "
                    " Ne prikazuj ASCII ili tekstualne dijagrame koordinatnog sistema u code blockovima (```...```) "
                    " osim ako korisnik eksplicitno traži ASCII dijagram. "
                    " Ako korisnik nije tražio graf, nemoj crtati ni spominjati grafički prikaz."
                    + only_clause + strict_geom_policy_text
                )
            }

            messages = [system_message]
            for msg in history[-5:]:
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["bot"]})
            messages.append({"role": "user", "content": pitanje})

            response = safe_llm_chat(MODEL_TEXT, messages, timeout=OPENAI_TIMEOUT, max_out=900, temperature=0.2)
            if response is None:
                odgovor = FALLBACK_HTML
                actual_model = MODEL_TEXT
            else:
                actual_model = getattr(response, "model", MODEL_TEXT)
                raw_odgovor = response.choices[0].message.content
                raw_odgovor = strip_ascii_graph_blocks(raw_odgovor)
                odgovor = f"<p>{latexify_fractions(safe_html(raw_odgovor))}</p>"

            if (not plot_expression_added) and should_plot(pitanje):
                expression = extract_plot_expression(pitanje, razred=razred, history=history)
                if expression:
                    odgovor = add_plot_div_once(odgovor, expression); plot_expression_added = True

            history.append({"user": pitanje, "bot": odgovor.strip()})
            history = history[-8:]
            session["history"] = history

            try:
                if sheet:
                    mod_str = f"text|{actual_model}"
                    sheet.append_row([pitanje, odgovor, mod_str])
            except Exception as ee:
                log.warning("Sheets append error: %s", ee)

        except Exception as e:
            log.error("FATAL index.POST: %r", e)
            err_html = FALLBACK_HTML
            history.append({"user": request.form.get('pitanje') or "[SLIKA]", "bot": err_html})
            history = history[-8:]
            session["history"] = history
            if request.form.get("ajax") == "1":
                return render_template("index.html", history=history, razred=razred)
            return redirect(url_for("index"))

    return render_template("index.html", history=history, razred=razred)

@app.errorhandler(413)
def too_large(e):
    msg = f"<p><b>Greška:</b> Fajl je prevelik (limit {MAX_MB} MB). Pokušaj ponovo (npr. fotografija bez Live/HEIC duplih snimaka), ili koristi GCS upload.</p>"
    return render_template("index.html", history=[{"user":"[SLIKA]", "bot": msg}], razred=session.get("razred")), 413

@app.errorhandler(500)
def handle_500(e):
    history = session.get("history", [])
    history.append({"user": "[SYSTEM]", "bot": FALLBACK_HTML})
    session["history"] = history[-8:]
    return render_template("index.html", history=session["history"], razred=session.get("razred")), 200

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

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["Vary"] = "Cookie"
    ancestors = os.getenv("FRAME_ANCESTORS", "").strip()
    if ancestors:
        resp.headers["Content-Security-Policy"] = f"frame-ancestors {ancestors}"
    resp.headers.pop("X-Frame-Options", None)
    return resp

# Ukloni ASCII grafove koji znaju biti ogromni u code fence-u
def strip_ascii_graph_blocks(text: str) -> str:
    fence_re = re.compile(r"```([\s\S]*?)```", flags=re.MULTILINE)
    def looks_like_ascii_graph(block: str) -> bool:
        sample = block.strip()
        if len(sample) == 0:
            return False
        allowed = set(" \t\r\n-_|*^><().,/\\0123456789xyXY+=")
        ratio_allowed = sum(c in allowed for c in sample) / max(len(sample), 1)
        lines = sample.splitlines()
        return (ratio_allowed > 0.9) and (3 <= len(lines) <= 60)
    def repl(m):
        block = m.group(1)
        return "" if looks_like_ascii_graph(block) else m.group(0)
    return fence_re.sub(repl, text)

@app.get("/app-health")
def app_health():
    problems = []
    llm_ok = False
    try:
        test = safe_llm_chat(MODEL_TEXT, [{"role":"user","content":"ping"}], timeout=15, max_out=10, temperature=0)
        llm_ok = True if (test and getattr(test, "choices", None)) else False
    except Exception as e:
        problems.append(f"OpenAI: {e}")
    base_url = None
    try:
        base_url = str(getattr(getattr(client, "_client", None), "base_url", None))
    except Exception:
        pass
    return {
        "llm_ok": llm_ok,
        "MODEL_TEXT": MODEL_TEXT,
        "MODEL_VISION": MODEL_VISION,
        "has_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "base_url": base_url,
        "problems": problems
    }, (200 if not problems else 500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
