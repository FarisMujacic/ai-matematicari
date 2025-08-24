from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify
from dotenv import load_dotenv
import os, re, base64, json, html, datetime, time
from datetime import timedelta
from io import BytesIO
from functools import wraps
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor

from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError, APITimeoutError
from flask_cors import CORS

# Google Sheets (opciono; neće srušiti ako credentials nema)
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Google Cloud Storage (za signed URL upload)
from google.cloud import storage

# ================== ENV ==================
load_dotenv(override=True)

# Admin kredencijali iz ENV-a (prima i lower i upper ključeve)
ADMIN_EMAIL = (os.getenv("ADMIN_EMAIL") or os.getenv("adminEmail") or "").strip().lower()
ADMIN_PASS  = (os.getenv("ADMIN_PASS")  or os.getenv("adminPass")  or "").strip()

# Access code (ENV ili fallback)
ACCESS_CODE = os.getenv("ACCESS_CODE", "MATH-2025")

# Ako nema DB, koristi se fallback skup (memorija)
ALLOWED_EMAILS_FALLBACK = {
    "ucenik1@example.com",
    # "ucenik2@example.com",
}

DB_URL = os.getenv("DATABASE_URL") or os.getenv("EXTERNAL_DATABASE_URL")

def _with_sslmode(url: str) -> str:
    if not url:
        return url
    if "sslmode=" in url:
        return url
    return url + ("&" if "?" in url else "?") + "sslmode=require"

# -------- Flask app / session --------
SECURE_COOKIES = os.getenv("COOKIE_SECURE", "0") == "1"  # lokalno False; na Render/Cloud Run postavi 1 po potrebi
app = Flask(__name__)
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=SECURE_COOKIES,
    SESSION_COOKIE_NAME="matbot_session_v2",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    SEND_FILE_MAX_AGE_DEFAULT=0,  # ne utiče na /uploads jer ga preskačemo niže
    ETAG_DISABLED=True,
)
CORS(app, supports_credentials=True)
app.secret_key = os.getenv("SECRET_KEY", "tajna_lozinka")

# Limit request body-a (sprječava tihi 500; prilagodi po potrebi)
MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# Upload folder (fallback kada ne koristiš GCS direktno iz browsera)
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- Guard-ovi ----------
def require_login(fn):
    @wraps(fn)
    def w(*a, **kw):
        if not session.get("user_email"):
            return redirect(url_for("prijava"))
        return fn(*a, **kw)
    return w

def require_admin(fn):
    @wraps(fn)
    def w(*a, **kw):
        if not session.get("is_admin"):
            return redirect(url_for("prijava"))
        return fn(*a, **kw)
    return w
# --------------------------------

# ---------- DB helper-i i whitelist ----------
def db():
    """Vrati konekciju ili None ako DB_URL nije postavljen."""
    if not DB_URL:
        return None
    return psycopg2.connect(_with_sslmode(DB_URL), cursor_factory=RealDictCursor)

def is_email_allowed(email: str) -> bool:
    """Provjera whiteliste iz baze (fallback na ALLOWED_EMAILS_FALLBACK)."""
    email = (email or "").strip().lower()
    conn = db()
    if conn:
        try:
            with conn, conn.cursor() as cur:
                cur.execute("select 1 from allowed_emails where email=%s", (email,))
                return cur.fetchone() is not None
        finally:
            conn.close()
    return email in ALLOWED_EMAILS_FALLBACK

def list_allowed_emails():
    conn = db()
    if conn:
        try:
            with conn, conn.cursor() as cur:
                cur.execute("select email from allowed_emails order by email asc;")
                rows = cur.fetchall()              # [{'email': '...'}, ...]
                return [r["email"] for r in rows]  # ['...', ...]
        finally:
            conn.close()
    return sorted(list(ALLOWED_EMAILS_FALLBACK))

def add_allowed_email(email: str) -> bool:
    email = (email or "").strip().lower()
    if not email or "@" not in email or len(email) > 200:
        return False
    conn = db()
    if conn:
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "insert into allowed_emails(email) values (%s) on conflict do nothing;",
                    (email,)
                )
            return True
        finally:
            conn.close()
    ALLOWED_EMAILS_FALLBACK.add(email)
    return True

def delete_allowed_email(email: str) -> None:
    email = (email or "").strip().lower()
    conn = db()
    if conn:
        try:
            with conn, conn.cursor() as cur:
                cur.execute("delete from allowed_emails where email=%s", (email,))
        finally:
            conn.close()
        return
    try:
        ALLOWED_EMAILS_FALLBACK.remove(email)
    except KeyError:
        pass
# ---------------------------------------------

# OpenAI klijent (veći timeout + više retry)
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "180"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=OPENAI_TIMEOUT,
    max_retries=OPENAI_MAX_RETRIES
)

# Modeli (možeš prepisati kroz .env)
MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-5-mini")
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-5")  # možeš probati "gpt-5-mini" za brže

# Google Sheets (opciono)
try:
    SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    CREDS_FILE = "credentials.json"
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    gs_client = gspread.authorize(creds)
    sheet = gs_client.open("matematika-bot").sheet1
except Exception:
    sheet = None

# GCS (za signed URL upload)
GCS_BUCKET = os.getenv("GCS_BUCKET", "").strip()
try:
    storage_client = storage.Client() if GCS_BUCKET else None
except Exception:
    storage_client = None

# ===== Globalni prompti po razredu (jedan izvor istine) =====
PROMPTI_PO_RAZREDU = {
    "5": "Ti si pomoćnik iz matematike za učenike 5. razreda osnovne škole. Objašnjavaj jednostavnim i razumljivim jezikom. Pomaži učenicima da razumiju zadatke iz prirodnih brojeva, osnovnih računskih operacija, jednostavne geometrije i tekstualnih zadataka. Svako rješenje objasni jasno, korak po korak.",
    "6": "Ti si pomoćnik iz matematike za učenike 6. razreda osnovne škole. Odgovaraj detaljno i pedagoški, koristeći primjere primjerene njihovom uzrastu. Pomaži im da razumiju razlomke, decimalne brojeve, procente, geometriju i tekstualne zadatke. Objasni rješenje jasno i korak po korak.",
    "7": "Ti si pomoćnik iz matematike za učenike 7. razreda osnovne škole. Pomaži im u razumijevanju složenijih zadataka iz algebre, geometrije i funkcija. Koristi jasan, primjeren jezik i objasni svaki korak logično i precizno.",
    "8": "Ti si pomoćnik iz matematike za učenike 8. razreda osnovne škole. Fokusiraj se na linearne izraze, sisteme jednačina, geometriju i statistiku. Pomaži učenicima da razumiju postupke i objasni svako rješenje detaljno, korak po korak.",
    "9": "Ti si pomoćnik iz matematike za učenike 9. razreda osnovne škole. Pomaži im u savladavanju zadataka iz algebre, funkcija, geometrije i statistike. Koristi jasan i stručan jezik, ali primjeren njihovom nivou. Objasni svaki korak rješenja jasno i precizno."
}
DOZVOLJENI_RAZREDI = set(PROMPTI_PO_RAZREDU.keys())

# --- parser brojeva zadataka ---
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

# --- helper: očisti HTML prije slanja modelu ---
tag_re = re.compile(r"<[^>]+>")

def _plain(txt: str, limit=800):
    t = tag_re.sub("", txt or "")
    return t[:limit]

def extract_plot_expression(text, razred=None, history=None):
    try:
        system_message = {
            "role": "system",
            "content": (
                "Tvoja uloga je da iz korisničkog pitanja detektuješ da li KORISNIK EKSPPLICITNO TRAŽI CRTEŽ GRAFA. "
                "Ako korisnik NE traži graf, odgovori tačno 'None'. "
                "Ako traži graf, i ako je prikladno nacrtati funkciju, odgovori isključivo u obliku 'y = ...'. "
                "Ako su data jednačina/nejednačina bez traženja grafa, odgovori 'None'. "
                "Ako je tražen graf nejednačine, takođe odgovori 'None'."
            )
        }
        messages = [system_message]
        if history:
            for msg in history[-5:]:
                messages.append({"role": "user", "content": _plain(msg["user"], 800)})
                messages.append({"role": "assistant", "content": _plain(msg["bot"], 800)})
        messages.append({"role": "user", "content": text})

        response = client.chat.completions.create(model=MODEL_TEXT, messages=messages)
        raw = response.choices[0].message.content.strip()
        if raw.lower() == "none":
            return None
        cleaned = raw.replace(" ", "")
        if cleaned.startswith("y="):
            return cleaned
        fx_match = re.match(r"f\s*\(\s*x\s*\)\s*=\s*(.+)", raw, flags=re.IGNORECASE)
        if fx_match:
            rhs = fx_match.group(1).strip()
            return "y=" + rhs.replace(" ", "")
    except Exception as e:
        print("GPT nije prepoznao funkciju za crtanje:", e, flush=True)
    return None

def get_history_from_request():
    history_json = request.form.get("history_json", "")
    if not history_json:
        return []
    try:
        data = json.loads(history_json)
        if not isinstance(data, list):
            return []
        trimmed = []
        for item in data[-5:]:
            u = str(item.get("user", ""))[:2000]
            b = str(item.get("bot", ""))[:4000]
            trimmed.append({"user": u, "bot": b})
        return trimmed
    except Exception as e:
        print("history_json parse fail:", e, flush=True)
        return []

# ---- Vision flow (URL) ----
def route_image_flow_url(image_url: str, razred: str, history, requested_tasks=None):
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])

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
        messages.append({"role": "user", "content": _plain(msg["user"], 800)})
        messages.append({"role": "assistant", "content": _plain(msg["bot"], 800)})

    user_content = [{"type": "text", "text": "Na slici je matematički zadatak."}]
    if requested_tasks:
        user_content[0]["text"] += f" Riješi isključivo zadatak(e): {', '.join(map(str, requested_tasks))}."
    else:
        user_content[0]["text"] += " Riješi samo ono što korisnik izričito traži."

    user_content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages.append({"role": "user", "content": user_content})

    try:
        resp = client.chat.completions.create(model=MODEL_VISION, messages=messages)
    except APIStatusError as e:
        print("OpenAI status error (vision URL):", getattr(e, "status_code", None), repr(e), flush=True)
        return "<p><b>Greška:</b> Servis sporo odgovara ili je zauzet. Pokušaj ponovo.</p>", "vision_url_error", "n/a"
    except (APIConnectionError, RateLimitError, APITimeoutError) as e:
        print("OpenAI vision error:", repr(e), flush=True)
        return "<p><b>Greška:</b> Servis sporo odgovara ili je zauzet. Pokušaj ponovo.</p>", "vision_url_error", "n/a"

    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{latexify_fractions(raw)}</p>", "vision_url", actual_model

# (stara base64 varijanta ostaje kao fallback, ali se ne koristi više u glavnom toku)
def route_image_flow(slika_bytes: bytes, razred: str, history, requested_tasks=None):
    image_b64 = base64.b64encode(slika_bytes).decode()
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])

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
        messages.append({"role": "user", "content": _plain(msg["user"], 800)})
        messages.append({"role": "assistant", "content": _plain(msg["bot"], 800)})

    user_content = [{"type": "text", "text": "Na slici je matematički zadatak."}]
    if requested_tasks:
        user_content[0]["text"] += f" Riješi isključivo zadatak(e): {', '.join(map(str, requested_tasks))}."
    else:
        user_content[0]["text"] += " Riješi samo ono što korisnik izričito traži."
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    messages.append({"role": "user", "content": user_content})

    try:
        resp = client.chat.completions.create(model=MODEL_VISION, messages=messages)
    except APIStatusError as e:
        print("OpenAI status error (vision base64):", getattr(e, "status_code", None), repr(e), flush=True)
        return "<p><b>Greška:</b> Servis sporo odgovara ili je zauzet. Pokušaj ponovo.</p>", "vision_direct_error", "n/a"
    except (APIConnectionError, RateLimitError, APITimeoutError) as e:
        print("OpenAI vision error (base64 path):", repr(e), flush=True)
        return "<p><b>Greška:</b> Servis sporo odgovara ili je zauzet. Pokušaj ponovo.</p>", "vision_direct_error", "n/a"

    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{latexify_fractions(raw)}</p>", "vision_direct", actual_model

# ----------------- LOGIN & ADMIN -----------------
@app.route("/prijava", methods=["GET", "POST"])
def prijava():
    if request.method == "GET":
        return render_template("prijava.html", error=None)

    email = (request.form.get("email") or "").strip().lower()
    code  = (request.form.get("code")  or "").strip()

    # 1) ADMIN login (ENV)
    if ADMIN_EMAIL and ADMIN_PASS and email == ADMIN_EMAIL and code == ADMIN_PASS:
        session["user_email"] = email
        session["is_admin"]  = True
        return redirect(url_for("admin_panel"))

    # 2) OBIČAN KORISNIK: whitelist + access code
    if is_email_allowed(email) and code == ACCESS_CODE:
        session["user_email"] = email
        session["is_admin"]  = False
        return redirect(url_for("index"))

    return render_template("prijava.html", error="Pogrešan email ili kod. Pokušaj ponovo.")

@app.get("/admin")
@require_login
@require_admin
def admin_panel():
    emails = list_allowed_emails()
    db_active = bool(DB_URL)
    return render_template("admin.html",
                           emails=emails, db_active=db_active,
                           message=request.args.get("m"))

@app.post("/admin/add")
@require_login
@require_admin
def admin_add():
    email = (request.form.get("email") or "").strip().lower()
    ok = add_allowed_email(email)
    msg = "OK" if ok else "Neispravan email."
    # PRG
    return redirect(url_for("admin_panel") + (f"?m={msg}" if msg else ""))

@app.post("/admin/delete")
@require_login
@require_admin
def admin_delete():
    email = (request.form.get("email") or "").strip().lower()
    delete_allowed_email(email)
    return redirect(url_for("admin_panel"))

# brzi JSON pregled liste (za debug)
@app.get("/admin/list.json")
@require_login
@require_admin
def admin_list_json():
    return {"db_active": bool(DB_URL), "emails": list_allowed_emails()}

@app.post("/odjava")
def odjava():
    session.clear()
    return redirect(url_for("prijava"))
# ----------------------------------------------

# ---- GCS signed upload endpoint (browser -> bucket PUT) ----
@app.post("/gcs/signed-upload")
@require_login
def gcs_signed_upload():
    if not (GCS_BUCKET and storage_client):
        return {"error": "GCS nije konfigurisan (GCS_BUCKET)."}, 500
    data = request.get_json(force=True, silent=True) or {}
    content_type = data.get("contentType") or "image/jpeg"
    ext = ".jpg" if "jpeg" in content_type else (".png" if "png" in content_type else ".webp")
    blob_name = f"uploads/{uuid4().hex}{ext}"

    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)

    upload_url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=10),
        method="PUT",
        content_type=content_type,
    )
    read_url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=45),
        method="GET",
    )
    return jsonify({"uploadUrl": upload_url, "readUrl": read_url})

# ---- Lokalni fallback serviranja uploadovane slike kao URL (/tmp/uploads) ----
@app.get("/uploads/<name>")
def uploads(name):
    # za lokalni/dev ili Cloud Run privremeni disk
    resp = send_from_directory(UPLOAD_DIR, name)
    # dozvoli caching da OpenAI bez problema povuče fajl
    resp.headers["Cache-Control"] = "public, max-age=900, immutable"
    return resp

# ---- Glavna ruta (MAT-BOT) ----
@app.route("/", methods=["GET", "POST"])
@require_login
def index():
    plot_expression_added = False
    history = get_history_from_request() or session.get("history", [])

    # Uvijek uzmi razred iz forme (ako je POST) ili iz sesije
    razred = (request.form.get("razred") or session.get("razred") or "").strip()

    if request.method == "POST":
        # ✅ Server-side validacija razreda (obavezno)
        if razred not in DOZVOLJENI_RAZREDI:
            return render_template("index.html",
                                   history=history, razred=razred,
                                   error="Molim odaberi razred."), 400
        # zapamti izbor
        session["razred"] = razred

        try:
            pitanje = (request.form.get("pitanje", "") or "").strip()
            slika = request.files.get("slika")
            image_url = (request.form.get("image_url") or "").strip()  # kad klijent koristi GCS signed upload
            is_ajax = request.form.get("ajax") == "1" or request.headers.get("X-Requested-With") == "XMLHttpRequest"

            # --- slika preko URL-a (preporučeno) ---
            if image_url:
                combined_text = pitanje
                requested = extract_requested_tasks(combined_text)

                try:
                    odgovor, used_path, used_model = route_image_flow_url(
                        image_url, razred, history, requested_tasks=requested
                    )
                except Exception as e:
                    print("route_image_flow_url error:", repr(e), flush=True)
                    odgovor = f"<p><b>Greška:</b> {html.escape(str(e))}</p>"
                    used_path = "error"; used_model = "n/a"

                # graf?
                will_plot = should_plot(combined_text)
                if (not plot_expression_added) and will_plot:
                    expression = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expression:
                        odgovor = add_plot_div_once(odgovor, expression); plot_expression_added = True

                history.append({"user": combined_text if combined_text else "[SLIKA-URL]", "bot": odgovor.strip()})
                # drži historiju razumno kratkom u cookie-u
                history = history[-8:]
                session["history"] = history

                try:
                    if sheet:
                        mod_str = f"{used_path}|{used_model}"
                        sheet.append_row([combined_text if combined_text else "[SLIKA-URL]", odgovor, mod_str])
                except Exception as ee:
                    print("Sheets append error:", ee, flush=True)

                if is_ajax:
                    return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # --- slika uploadovana direktno (fallback: snimi pa posluži kao URL) ---
            if slika and slika.filename:
                # sačuvaj u /tmp/uploads i napravi javni URL
                ext = os.path.splitext(slika.filename)[1].lower() or ".jpg"
                fname = f"{uuid4().hex}{ext}"
                path  = os.path.join(UPLOAD_DIR, fname)
                slika.save(path)
                public_url = (request.url_root.rstrip("/") + "/uploads/" + fname)

                combined_text = pitanje
                requested = extract_requested_tasks(combined_text)

                try:
                    odgovor, used_path, used_model = route_image_flow_url(
                        public_url, razred, history, requested_tasks=requested
                    )
                except Exception as e:
                    print("route_image_flow_url (local) error:", repr(e), flush=True)
                    odgovor = f"<p><b>Greška:</b> {html.escape(str(e))}</p>"
                    used_path = "error"; used_model = "n/a"

                # VAŽNO: ne briši odmah fajl (OpenAI može dohvatiti kasnije)
                # Ako želiš čišćenje, uradi poseban periodic job ili obriši nakon dužeg vremena.
                # try:
                #     time.sleep(120)
                #     os.remove(path)
                # except Exception:
                #     pass

                # graf?
                will_plot = should_plot(combined_text)
                if (not plot_expression_added) and will_plot:
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
                    print("Sheets append error:", ee, flush=True)

                if is_ajax:
                    return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # --- tekst ---
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
                messages.append({"role": "user", "content": _plain(msg["user"], 800)})
                messages.append({"role": "assistant", "content": _plain(msg["bot"], 800)})
            messages.append({"role": "user", "content": pitanje})

            response = client.chat.completions.create(model=MODEL_TEXT, messages=messages)
            actual_model = getattr(response, "model", MODEL_TEXT)
            raw_odgovor = response.choices[0].message.content
            raw_odgovor = strip_ascii_graph_blocks(raw_odgovor)
            odgovor = f"<p>{latexify_fractions(raw_odgovor)}</p>"

            will_plot = should_plot(pitanje)
            if (not plot_expression_added) and will_plot:
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
                print("Sheets append error:", ee, flush=True)

        except Exception as e:
            print("FATAL index.POST:", repr(e), flush=True)
            err_html = f"<p><b>Greška servera:</b> {html.escape(str(e))}</p>"
            history.append({"user": request.form.get('pitanje') or "[SLIKA]", "bot": err_html})
            history = history[-8:]
            session["history"] = history
            if request.form.get("ajax") == "1":
                return render_template("index.html", history=history, razred=razred)
            return redirect(url_for("index"))

    # GET
    return render_template("index.html", history=history, razred=razred)

# ---- Error handler za prevelik upload ----
@app.errorhandler(413)
def too_large(e):
    msg = f"<p><b>Greška:</b> Fajl je prevelik (limit {MAX_MB} MB). Pokušaj ponovo (npr. fotografija bez duplih snimaka/Live/HEIC), ili koristi GCS upload.</p>"
    return render_template("index.html", history=[{"user":"[SLIKA]", "bot": msg}], razred=session.get("razred")), 413

@app.route("/clear", methods=["POST"])
@require_login
def clear():
    if request.form.get("confirm_clear") == "1":
        session.pop("history", None)
        session.pop("razred", None)
    if request.form.get("ajax") == "1":
        return render_template("index.html", history=[], razred=None)
    return redirect("/")

@app.after_request
def add_no_cache_headers(resp):
    # Ne diraj cache header za uploadovane slike (moraju biti dohvatljive i kesirane)
    if request.path.startswith("/uploads/"):
        return resp
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
    fence_re = re.compile(r"```([\s\S]*?)```", flags=re.MULTILINE)
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
    text = re.sub(r"(Grafički prikaz.*?:\s*)?```[\s\S]*?```",
                  lambda m: "" if "```" in m.group(0) else m.group(0),
                  text, flags=re.IGNORECASE)
    return fence_re.sub(repl, text)

@app.get("/db-health")
def db_health():
    try:
        conn = db()
        if not conn:
            return "DB_URL not set", 500
        with conn, conn.cursor() as cur:
            cur.execute("select count(*) as n from allowed_emails;")
            row = cur.fetchone()
            n = row["n"] if row and "n" in row else 0
        return f"OK, allowed_emails={n}", 200
    except Exception as e:
        return f"DB error: {e}", 500

@app.get("/app-health")
def app_health():
    problems = []
    # DB
    db_ok = False
    try:
        conn = db()
        if conn:
            with conn, conn.cursor() as cur:
                cur.execute("select 1;")
            db_ok = True
    except Exception as e:
        problems.append(f"DB: {e}")
    # OpenAI
    llm_ok = False
    try:
        test = client.chat.completions.create(
            model=MODEL_TEXT,
            messages=[{"role":"user","content":"ping"}],
        )
        llm_ok = True if getattr(test, "choices", None) else False
    except Exception as e:
        problems.append(f"OpenAI: {e}")
    return {
        "db_ok": db_ok,
        "llm_ok": llm_ok,
        "MODEL_TEXT": MODEL_TEXT,
        "MODEL_VISION": MODEL_VISION,
        "has_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "problems": problems
    }, (200 if not problems else 500)

if __name__ == "__main__":
    # Za lokalni debug; u Cloud Runu koristiš gunicorn
    app.run(debug=True, port=5000)
