# app.py — slika+tekst u istoj poruci, reuse zadnje slike, Mathpix OCR kao pomoć

from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify
from dotenv import load_dotenv
import os, re, base64, json, html, datetime, time, logging
from datetime import timedelta
from io import BytesIO
from uuid import uuid4

from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from flask_cors import CORS
import requests  # Mathpix

import gspread
from oauth2client.service_account import ServiceAccountCredentials

try:
    from google.cloud import storage
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
    ETAG_DISABLED=True,
)
CORS(app, supports_credentials=True)
app.secret_key = os.getenv("SECRET_KEY", "tajna_lozinka")

MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "240"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)

MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-5-mini")
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-5")

# ---------------- Mathpix config ----------------
MATHPIX_API_ID  = os.getenv("MATHPIX_API_ID")
MATHPIX_API_KEY = os.getenv("MATHPIX_API_KEY")
MATHPIX_MODE    = os.getenv("MATHPIX_MODE", "fallback").lower().strip()  # fallback|prefer|off
MATHPIX_ENABLED = bool(MATHPIX_API_ID and MATHPIX_API_KEY and MATHPIX_MODE != "off")
MATHPIX_TIMEOUT = float(os.getenv("MATHPIX_TIMEOUT", "30"))

# ---------------- Google Sheets -----------------
try:
    SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    CREDS_FILE = "credentials.json"
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    gs_client = gspread.authorize(creds)
    sheet = gs_client.open("matematika-bot").sheet1
except Exception as e:
    log.warning("Sheets disabled: %s", e)
    sheet = None

# ---------------- GCS ---------------------------
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

# ===================== Biz logika =====================
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
def _openai_chat(model: str, messages: list, timeout: float = None):
    try:
        cli = client if timeout is None else client.with_options(timeout=timeout)
        return cli.chat.completions.create(model=model, messages=messages)
    except (APIConnectionError, APIStatusError, RateLimitError) as e:
        raise e
    except Exception as e:
        raise e

# ===================== TEXT pipeline =====================
def answer_with_text_pipeline(pure_text: str, razred: str, history, requested):
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])
    only_clause = ""
    strict_geom_policy = (
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
            + only_clause + strict_geom_policy
        )
    }

    messages = [system_message]
    for msg in history[-5:]:
        messages.append({"role":"user","content": msg["user"]})
        messages.append({"role":"assistant","content": msg["bot"]})
    messages.append({"role":"user","content": pure_text})

    response = _openai_chat(MODEL_TEXT, messages, timeout=OPENAI_TIMEOUT)
    actual_model = getattr(response, "model", MODEL_TEXT)
    raw = response.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    html_out = f"<p>{latexify_fractions(raw)}</p>"
    return html_out, actual_model

# ===================== Mathpix helpers =====================
def _mathpix_headers():
    return {
        "app_id": MATHPIX_API_ID,
        "app_key": MATHPIX_API_KEY,
        "Content-Type": "application/json"
    }

def mathpix_from_src(src: str):
    if not MATHPIX_ENABLED:
        return {"ok": False, "text": "", "latex": "", "raw": None}
    payload = {
        "src": src,
        "formats": ["text"],   # plain text je dovoljno za pomoć
        "rm_spaces": True,
        "confidence_threshold": 0.1,
    }
    try:
        r = requests.post("https://api.mathpix.com/v3/text",
                          headers=_mathpix_headers(),
                          json=payload,
                          timeout=MATHPIX_TIMEOUT)
        j = r.json()
        text  = (j.get("text") or "").strip()
        ok = bool(text)
        return {"ok": ok, "text": text, "latex": "", "raw": j}
    except Exception as e:
        log.warning("Mathpix OCR error: %s", e)
        return {"ok": False, "text": "", "latex": "", "raw": None}

def mathpix_from_bytes(image_bytes: bytes, mime="image/jpeg"):
    b64 = base64.b64encode(image_bytes).decode()
    return mathpix_from_src(f"data:{mime};base64,{b64}")

# ===================== Vision poruke i pomoć =====================
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
            "Ako je korisnik koristio riječi tipa 'prvi/drugi/treći', tumači ih relativno na redoslijed vidljivih brojeva na stranici."
        )
    strict_geom_policy = (
        " Radi tačno i oprezno:\n"
        "1) PRVO, jasno prepiši koje brojeve zadataka i oznake podzadataka vidiš (npr. 497; a), b), c)).\n"
        "2) Nemoj pretpostavljati paralelnost, jednakokrakost, jednake uglove ili sličnost trokuta ako to nije eksplicitno označeno.\n"
        "3) Ako korisnik uz sliku da tekstualne podatke, ONI SU ISTINITI i imaju prioritet nad onim što vidiš.\n"
        "4) Daj konačan odgovor ako je moguće; inače navedi šta još treba da bi se zadatak zaključio."
    )
    return only_clause, strict_geom_policy

# Heuristika: izvući glavne brojeve zadataka sa slike
def _detect_exercise_numbers_from_image(user_content):
    try:
        sys = {
            "role": "system",
            "content": (
                "Pročitaj sliku i izdvoji sve VIDljive glavne brojeve zadataka "
                "(npr. 497, 498, 499...), odozgo prema dolje. "
                "Odgovori SAMO brojevima razdvojenim zarezom (npr. '497, 498, 499'). "
                "Ako ne vidiš, odgovori 'NONE'."
            )
        }
        msgs = [sys, {"role": "user", "content": user_content}]
        resp = _openai_chat(MODEL_VISION, msgs, timeout=min(OPENAI_TIMEOUT, 30))
        txt = (resp.choices[0].message.content or "").strip()
        if txt.upper() == "NONE":
            return []
        nums = [int(n) for n in re.findall(r"\d{2,5}", txt)]
        return nums
    except Exception as e:
        log.warning("Detect numbers fail: %s", e)
        return []

# ===================== Vision flows (GENERIČKI) =====================
def route_image_with_src(image_src: str, razred: str, history, requested_tasks=None, user_text: str | None = None):
    """
    image_src: https://... ili data:image/...;base64,...
    """
    only_clause, strict_geom_policy = _vision_clauses(requested_tasks)
    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)

    user_content = []

    # 1) korisnički tekst (ako postoji)
    if user_text:
        user_content.append({"type": "text", "text": f"Korisnički tekst: {user_text}"})

    # 2) okvirna instrukcija
    base_text = "Na slici je matematički zadatak."
    if requested_tasks:
        base_text += f" Riješi isključivo zadatak(e): {', '.join(map(str, requested_tasks))}."
    else:
        base_text += " Riješi samo ono što korisnik izričito traži."
    user_content.append({"type": "text", "text": base_text})

    # 3) (opcionalno) Mathpix OCR kao pomoćni tekst
    if MATHPIX_ENABLED:
        try:
            mp = mathpix_from_src(image_src)
            if mp.get("ok") and (MATHPIX_MODE in ("prefer", "fallback")) and mp.get("text"):
                user_content.append({"type": "text", "text": f"OCR (Mathpix) — prepoznati tekst sa slike:\n{mp['text'][:4000]}"})
        except Exception as e:
            log.warning("Mathpix help skipped: %s", e)

    # 4) sama slika
    user_content.append({"type": "image_url", "image_url": {"url": image_src}})

    messages.append({"role": "user", "content": user_content})

    # sačuvaj prepoznate brojeve (za kasniju poruku bez slike)
    try:
        detected = _detect_exercise_numbers_from_image(user_content)
        if detected:
            session["last_task_numbers"] = detected
    except Exception as e:
        log.debug("detect store skip: %s", e)

    try:
        resp = _openai_chat(MODEL_VISION, messages, timeout=OPENAI_TIMEOUT)
    except (APIConnectionError, APIStatusError, RateLimitError) as e:
        log.warning("OpenAI vision error: %r", e)
        msg = str(e)
        if "Timeout while downloading" in msg or "timed out while downloading" in msg:
            return (
                "<p><b>Greška:</b> Slika se nije mogla preuzeti dovoljno brzo. "
                "Pokušaj ponovo ili koristi GCS upload (brže), ili pošalji manju sliku.</p>",
                "vision_error", "n/a"
            )
        return "<p><b>Greška:</b> Servis sporo odgovara ili je zauzet. Pokušaj ponovo.</p>", "vision_error", "n/a"
    except Exception as e:
        log.error("OpenAI vision fatal: %s", e)
        return "<p><b>Greška:</b> Neočekivan problem pri analizi slike.</p>", "vision_error", "n/a"

    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{latexify_fractions(raw)}</p>", "vision", actual_model

# kompatibilni wrapperi
def route_image_flow_url(image_url: str, razred: str, history, requested_tasks=None, user_text=None):
    return route_image_with_src(image_url, razred, history, requested_tasks, user_text)

def route_image_flow(slika_bytes: bytes, razred: str, history, requested_tasks=None, user_text=None, mime="image/jpeg"):
    image_b64 = base64.b64encode(slika_bytes).decode()
    data_url = f"data:{mime};base64,{image_b64}"
    return route_image_with_src(data_url, razred, history, requested_tasks, user_text)

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
                version="v4",
                expiration=datetime.timedelta(minutes=45),
                method="GET",
            )
        else:
            try:
                blob.make_public()
                url = blob.public_url
            except Exception:
                url = blob.generate_signed_url(
                    version="v4",
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
            version="v4",
            expiration=datetime.timedelta(minutes=15),
            method="PUT",
            content_type=content_type,
        )

        if GCS_SIGNED_GET:
            read_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(minutes=45),
                method="GET",
            )
        else:
            try:
                blob.make_public()
                read_url = blob.public_url
            except Exception:
                read_url = blob.generate_signed_url(
                    version="v4",
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

# ===================== GLAVNA RUTA =====================
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

            # --- IMAGE via URL ---
            if image_url:
                combined_text = pitanje
                requested = extract_requested_tasks(combined_text)

                odgovor, used_path, used_model = route_image_flow_url(
                    image_url, razred, history, requested_tasks=requested, user_text=combined_text
                )

                # sačuvaj posljednju sliku (za naredne tekst-only poruke)
                session["last_image_src"] = image_url

                will_plot = should_plot(combined_text)
                if (not plot_expression_added) and will_plot:
                    expression = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expression:
                        odgovor = add_plot_div_once(odgovor, expression); plot_expression_added = True

                display_user = (combined_text + " [slika]") if combined_text else "[slika]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]
                session["history"] = history

                try:
                    if sheet:
                        mod_str = f"{used_path}|{used_model}"
                        sheet.append_row([display_user, odgovor, mod_str])
                except Exception as ee:
                    log.warning("Sheets append error: %s", ee)

                if is_ajax:
                    return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # --- IMAGE via FILE UPLOAD ---
            if slika and slika.filename:
                slika.stream.seek(0, os.SEEK_END)
                size_bytes = slika.stream.tell()
                slika.stream.seek(0)

                combined_text = pitanje
                requested = extract_requested_tasks(combined_text)

                used_src = None
                if size_bytes <= 1_500_000:
                    body = slika.read()
                    odgovor, used_path, used_model = route_image_flow(
                        body, razred, history, requested_tasks=requested, user_text=combined_text, mime=slika.mimetype or "image/jpeg"
                    )
                    # spremi data: URL
                    b64 = base64.b64encode(body).decode()
                    used_src = f"data:{slika.mimetype or 'image/jpeg'};base64,{b64}"
                else:
                    if not (storage_client and GCS_BUCKET) and (GCS_REQUIRED or os.getenv("K_SERVICE")):
                        return render_template("index.html", history=history, razred=razred,
                                               error="GCS nije konfigurisan – upload velikih slika nije moguć."), 400
                    gcs_url = gcs_upload_filestorage(slika)
                    if gcs_url:
                        odgovor, used_path, used_model = route_image_flow_url(
                            gcs_url, razred, history, requested_tasks=requested, user_text=combined_text
                        )
                        used_src = gcs_url
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
                        odgovor, used_path, used_model = route_image_flow_url(
                            public_url, razred, history, requested_tasks=requested, user_text=combined_text
                        )
                        used_src = public_url

                if used_src:
                    session["last_image_src"] = used_src

                will_plot = should_plot(combined_text)
                if (not plot_expression_added) and will_plot:
                    expression = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expression:
                        odgovor = add_plot_div_once(odgovor, expression); plot_expression_added = True

                display_user = (combined_text + " [slika]") if combined_text else "[slika]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]
                session["history"] = history

                try:
                    if sheet:
                        mod_str = f"{used_path}|{used_model}"
                        sheet.append_row([display_user, odgovor, mod_str])
                except Exception as ee:
                    log.warning("Sheets append error: %s", ee)

                if is_ajax:
                    return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # --- TEKST (ali možda se referiše na zadnju sliku) ---
            requested = extract_requested_tasks(pitanje)
            last_src = session.get("last_image_src")

            if last_src and requested:
                # Ako ima zadnja slika i korisnik spominje brojeve/ordinale -> koristi sliku
                odgovor, used_path, used_model = route_image_with_src(
                    last_src, razred, history, requested_tasks=requested, user_text=pitanje
                )
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
                        mod_str = f"vision-reuse|{used_model}"
                        sheet.append_row([pitanje, odgovor, mod_str])
                except Exception as ee:
                    log.warning("Sheets append error: %s", ee)

                if is_ajax:
                    return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            # inače — čisti tekst pipeline
            odgovor, actual_model = answer_with_text_pipeline(pitanje, razred, history, requested)

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
                log.warning("Sheets append error: %s", ee)

        except Exception as e:
            log.error("FATAL index.POST: %r", e)
            err_html = f"<p><b>Greška servera:</b> {html.escape(str(e))}</p>"
            history.append({"user": request.form.get('pitanje') or "[slika]", "bot": err_html})
            history = history[-8:]
            session["history"] = history
            if request.form.get("ajax") == "1":
                return render_template("index.html", history=history, razred=razred)
            return redirect(url_for("index"))

    return render_template("index.html", history=history, razred=razred)

# ===================== Ostalo =====================
@app.errorhandler(413)
def too_large(e):
    msg = f"<p><b>Greška:</b> Fajl je prevelik (limit {MAX_MB} MB). Pokušaj ponovo (npr. fotografija bez Live/HEIC duplih snimaka), ili koristi GCS upload.</p>"
    return render_template("index.html", history=[{"user":"[slika]", "bot": msg}], razred=session.get("razred")), 413

@app.route("/clear", methods=["POST"])
def clear():
    if request.form.get("confirm_clear") == "1":
        session.pop("history", None)
        session.pop("razred", None)
        session.pop("last_image_src", None)
        session.pop("last_task_numbers", None)
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

@app.get("/app-health")
def app_health():
    problems = []
    llm_ok = False
    try:
        test = _openai_chat(MODEL_TEXT, [{"role":"user","content":"ping"}], timeout=15)
        llm_ok = True if getattr(test, "choices", None) else False
    except Exception as e:
        problems.append(f"OpenAI: {e}")
    return {
        "llm_ok": llm_ok,
        "MODEL_TEXT": MODEL_TEXT,
        "MODEL_VISION": MODEL_VISION,
        "has_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "MATHPIX_ENABLED": MATHPIX_ENABLED,
        "MATHPIX_MODE": MATHPIX_MODE,
        "problems": problems
    }, (200 if not problems else 500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
