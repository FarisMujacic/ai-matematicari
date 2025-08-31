# app.py  -- MAT-BOT (history-aware)
# Flask backend koji pamti kontekst (history) i follow-upove:
# - "ne kontam / objasni / zasto" objasnjava prethodni odgovor
# - "uradi 3. zadatak sa slike" radi na zadnjoj slici ako nova nije poslata
# - "prvi / drugi / treci zadatak" mapira se u brojeve
#
# Minimalne vanjske ovisnosti: Flask, flask-cors, python-dotenv, openai, requests

from __future__ import annotations
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import os, re, base64, json, html, datetime, logging, traceback
from datetime import timedelta
from uuid import uuid4
import requests
from urllib.parse import urlparse

from openai import OpenAI

# ---------------- Bootstrapping ----------------
load_dotenv(override=False)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("matbot")

SECURE_COOKIES = os.getenv("COOKIE_SECURE", "0") == "1"

app = Flask(__name__)
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=SECURE_COOKIES,
    SESSION_COOKIE_NAME="matbot_session_v3",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    SEND_FILE_MAX_AGE_DEFAULT=0,
    ETAG_DISABLED=True,
    MAX_CONTENT_LENGTH=int(os.getenv("MAX_CONTENT_LENGTH_MB", "50")) * 1024 * 1024,
)
CORS(app, supports_credentials=True)
app.secret_key = os.getenv("SECRET_KEY", "tajna_lozinka")

# Health (jedini)
@app.get("/healthz")
def healthz():
    return {"ok": True}, 200

# ---------------- OpenAI ----------------
_OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not _OPENAI_API_KEY:
    log.error("OPENAI_API_KEY nije postavljen.")
client = OpenAI(api_key=_OPENAI_API_KEY, timeout=float(os.getenv("OPENAI_TIMEOUT", "120")), max_retries=2)

MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-5-mini")
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-5")

def _openai_chat(model: str, messages: list, timeout: float | None = None, max_tokens: int | None = None):
    cli = client if timeout is None else client.with_options(timeout=timeout)
    params = {"model": model, "messages": messages}
    if max_tokens is not None:
        # probaj preferirani parametar; fallback na max_tokens ako treba
        try:
            params["max_completion_tokens"] = max_tokens
        except Exception:
            params["max_tokens"] = max_tokens
    try:
        return cli.chat.completions.create(**params)
    except Exception as e:
        # fallback za API koji ne prima max_completion_tokens
        msg = str(e)
        if "max_completion_tokens" in msg or "Unsupported parameter" in msg:
            params.pop("max_completion_tokens", None)
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            return cli.chat.completions.create(**params)
        raise

# ---------------- Domene / prompti ----------------
PROMPTI_PO_RAZREDU = {
    "5": "Ti si pomoćnik iz matematike za učenike 5. razreda. Objašnjavaj jednostavno i korak po korak.",
    "6": "Ti si pomoćnik iz matematike za 6. razred. Budi pedagoški i precizan, sve korak po korak.",
    "7": "Ti si pomoćnik iz matematike za 7. razred. Objašnjavaj jasno, uz provjere koraka.",
    "8": "Ti si pomoćnik iz matematike za 8. razred. Linearni izrazi, geometrija; rješenja korak po korak.",
    "9": "Ti si pomoćnik iz matematike za 9. razred. Algebra, funkcije, geometrija; objasni svaku transformaciju.",
}
DOZVOLJENI_RAZREDI = set(PROMPTI_PO_RAZREDU.keys())

# ---- fraze za follow-up (objasni / ne kontam / zasto / kako si dobio) ----
FOLLOWUP_HINTS = re.compile(
    r"\b(ne\s*konta?m|ne\s*razumijem|objasni|pojasni|zasto|za[sš]to|kako\s+si\s+to\s+dobio|ponovi|detaljnije)\b",
    flags=re.IGNORECASE
)

# ---- izdvajanje brojeva zadataka i ordinala ----
ORDINAL_WORDS = {
    "prvi": 1, "drugi": 2, "treci": 3, "treći": 3, "cetvrti": 4, "četvrti": 4,
    "peti": 5, "sesti": 6, "šesti": 6, "sedmi": 7, "osmi": 8, "deveti": 9, "deseti": 10,
    "zadnji": -1, "posljednji": -1
}
_task_num_re = re.compile(
    r"(?:zadatak\s*(?:broj\s*)?(\d{1,4}))|(?:\b(\d{1,4})\s*\.)|(?:\b(" + "|".join(ORDINAL_WORDS.keys()) + r")\b)",
    flags=re.IGNORECASE
)

def extract_requested_tasks(text: str) -> list[int]:
    if not text:
        return []
    tasks, seen = [], set()
    for m in _task_num_re.finditer(text):
        if m.group(1):
            n = int(m.group(1))
        elif m.group(2):
            n = int(m.group(2))
        else:
            n = ORDINAL_WORDS.get(m.group(3).lower(), None)
        if n is None:
            continue
        if n not in seen:
            tasks.append(n); seen.add(n)
    return tasks

# ---------------- History helpers ----------------
def get_history_from_request() -> list[dict] | None:
    """Cita history iz form field-a 'history_json' (string JSON liste)"""
    try:
        hx = request.form.get("history_json")
        if hx:
            data = json.loads(hx)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return None

def get_history_any() -> list[dict]:
    """Najprije form, pa JSON body, pa session."""
    hx = get_history_from_request()
    if hx:
        return hx
    try:
        data = request.get_json(silent=True) or {}
        hx = data.get("history") or data.get("history_json")
        if isinstance(hx, str):
            hx = json.loads(hx)
        if isinstance(hx, list):
            return hx
    except Exception:
        pass
    return session.get("history", [])

def push_history_pair(user_text: str, bot_html: str):
    arr = session.get("history", [])
    arr.append({"user": user_text, "bot": (bot_html or "").strip()})
    session["history"] = arr[-8:]

# ---------------- Image helpers ----------------
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _short_name_for_display(name: str, maxlen: int = 60) -> str:
    n = os.path.basename(name or "").strip() or "nepoznato"
    if len(n) > maxlen:
        n = n[:maxlen - 3] + "..."
    return html.escape(n)

def _name_from_url(u: str) -> str:
    try:
        p = urlparse(u)
        base = os.path.basename(p.path) or ""
        return _short_name_for_display(base if base else u.split("?")[0].split("/")[-1] or u)
    except Exception:
        return _short_name_for_display(u)

def remember_image(url: str):
    imgs = session.get("images", [])
    imgs.append({"url": url, "ts": datetime.datetime.utcnow().isoformat() + "Z"})
    session["images"] = imgs[-6:]
    session["last_image_url"] = url

def last_image_url() -> str | None:
    return session.get("last_image_url")

@app.get("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

# ---------------- ASCII-plot filter (da se ne pojavljuju code-block grafovi) ----------------
def strip_ascii_graph_blocks(text: str) -> str:
    fence = re.compile(r"```([\s\S]*?)```", flags=re.MULTILINE)
    def looks_like_ascii_graph(block: str) -> bool:
        sample = block.strip()
        if not sample:
            return False
        allowed = set(" \t\r\n-_|*^><().,/\\0123456789xyXY")
        ratio = sum(c in allowed for c in sample) / max(1, len(sample))
        lines = sample.splitlines()
        return (ratio > 0.9) and (3 <= len(lines) <= 40)
    def repl(m):
        block = m.group(1)
        return "" if looks_like_ascii_graph(block) else m.group(0)
    return fence.sub(repl, text)

# ---------------- Vision helpers ----------------
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

def _vision_messages_base(razred: str, history: list[dict], only_clause: str, strict_geom_policy: str):
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])
    system_message = {
        "role": "system",
        "content": (
            prompt_za_razred
            + " Odgovaraj jezikom pitanja (bosanski/ijekavica). "
            + "Ne prikazuj ASCII grafove osim ako je izricito trazeno. "
            + only_clause + " " + strict_geom_policy
        )
    }
    messages = [system_message]
    # dodaj dosadasnju historiju zadnjih 5 parova
    for msg in (history or [])[-5:]:
        if msg.get("user"):
            messages.append({"role": "user", "content": msg["user"]})
        if msg.get("bot"):
            messages.append({"role": "assistant", "content": msg["bot"]})
    return messages

def _vision_clauses():
    return "", " Radi tacno i oprezno. Ako nesto nedostaje, reci sta nedostaje i stani."

def route_image_flow_url(image_url: str, razred: str, history: list[dict], user_text: str | None = None, timeout_override: float | None = None):
    only_clause, strict_geom_policy = _vision_clauses()

    # probaj skinuti bytes pa dati kao inline (najpouzdanije)
    try:
        r = requests.get(image_url, timeout=15)
        r.raise_for_status()
        mime_hint = r.headers.get("Content-Type") or None
        return route_image_flow(r.content, razred, history, user_text=user_text, timeout_override=timeout_override, mime_hint=mime_hint)
    except Exception as e:
        log.warning("download image_url failed (%s), fallback direct URL to model", e)

    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)
    user_content = []
    if user_text:
        user_content.append({"type": "text", "text": f"Korisnicki tekst: {user_text}"})
    user_content.append({"type": "text", "text": "Na slici je matematicki zadatak."})
    user_content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages.append({"role": "user", "content": user_content})
    resp = _openai_chat(MODEL_VISION, messages, timeout=timeout_override or float(os.getenv("HARD_TIMEOUT_S", "120")))
    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{raw}</p>", "vision_url", actual_model

def route_image_flow(slika_bytes: bytes, razred: str, history: list[dict], user_text: str | None = None, timeout_override: float | None = None, mime_hint: str | None = None):
    only_clause, strict_geom_policy = _vision_clauses()
    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)
    data_url = _bytes_to_data_url(slika_bytes, mime_hint=mime_hint)
    user_content = []
    if user_text:
        user_content.append({"type": "text", "text": f"Korisnicki tekst: {user_text}"})
    user_content.append({"type": "text", "text": "Na slici je matematic ki zadatak."})
    user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    messages.append({"role": "user", "content": user_content})
    resp = _openai_chat(MODEL_VISION, messages, timeout=timeout_override or float(os.getenv("HARD_TIMEOUT_S", "120")))
    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{raw}</p>", "vision_direct", actual_model

# ---------------- Tekst pipeline ----------------
def answer_with_text_pipeline(pure_text: str, razred: str, history: list[dict], requested: list[int], timeout_override: float | None = None):
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])
    strict_geom_policy = (
        " Ako problem ukljucuje geometriju: 1) koristi samo eksplicitno date podatke; "
        "2) ne pretpostavljaj nista bez oznake; 3) navedi nazive teorema kad ih koristis."
    )
    only_clause = ""
    if requested:
        only_clause = " Rijesi ISKLJUCIVO sljedece zadatke: " + ", ".join(map(str, requested)) + "."

    # follow-up heuristika: ako je poruka "ne kontam..." fokusiraj se na prethodni isti problem
    followup_note = (
        " Ako je ova poruka follow-up (npr. 'ne kontam', 'objasni', 'zasto'), referiraj se na prethodni problem i "
        "objasni isti zadatak korak po korak. Ako korisnik trazi drugi zadatak s iste slike, koristi kontekst."
    )

    system_message = {
        "role": "system",
        "content": (
            prompt_za_razred
            + " Odgovaraj jezikom pitanja (bosanski ijekavica). "
            + "Ako pitanje nije iz matematike, reci da treba postaviti matematicko pitanje. "
            + "Ne crtaj ASCII grafove osim ako je trazeno."
            + only_clause + strict_geom_policy + followup_note
        )
    }

    messages = [system_message]
    # zadnjih 5 parova iz historije
    for msg in (history or [])[-5:]:
        if msg.get("user"):
            messages.append({"role": "user", "content": msg["user"]})
        if msg.get("bot"):
            messages.append({"role": "assistant", "content": msg["bot"]})
    messages.append({"role": "user", "content": pure_text})

    response = _openai_chat(MODEL_TEXT, messages, timeout=timeout_override or float(os.getenv("HARD_TIMEOUT_S", "120")))
    actual_model = getattr(response, "model", MODEL_TEXT)
    raw = response.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    html_out = f"<p>{html.escape(raw).replace('\\n', '<br>')}</p>"
    return html_out, actual_model

# ---------------- Index (opcionalno templating) ----------------
@app.route("/", methods=["GET"])
def index():
    # Ako imas templates/index.html, ovo ce ga renderati; u suprotnom vrati plain tekst
    try:
        return render_template("index.html", history=session.get("history", []), razred=session.get("razred"))
    except Exception:
        return "MAT-BOT backend running", 200

# ---------------- SUBMIT (history-aware) ----------------
@app.route("/submit", methods=["POST", "OPTIONS"])
def submit():
    if request.method == "OPTIONS":
        return ("", 204)

    # ulazi iz forme / JSON-a
    razred = (request.form.get("razred") or request.args.get("razred") or "").strip()
    user_text = (request.form.get("user_text") or request.form.get("pitanje") or "").strip()
    image_url = (request.form.get("image_url") or request.args.get("image_url") or "").strip()

    data = request.get_json(silent=True) or {}
    if data:
        razred    = (data.get("razred")    or razred).strip()
        user_text = (data.get("pitanje")   or data.get("user_text") or user_text).strip()
        image_url = (data.get("image_url") or image_url).strip()

    if razred not in DOZVOLJENI_RAZREDI:
        razred = "5"
    session["razred"] = razred

    history = get_history_any()  # kljucno: uzmi zadnjih ~5 iz forme/JS ili sesije
    requested = extract_requested_tasks(user_text)

    # Heuristike: follow-up poruka bez nove slike, a postoje slike
    is_followup = bool(FOLLOWUP_HINTS.search(user_text))
    wants_from_image = ("sa slike" in user_text.lower()) or ("slike" in user_text.lower()) or bool(requested)

    # fajl iz forme (slika)
    file_storage = request.files.get("file")
    file_bytes = None
    file_mime = None
    file_name = None
    if file_storage and file_storage.filename:
        try:
            file_bytes = file_storage.read()
            file_mime = file_storage.mimetype or "application/octet-stream"
            file_name = file_storage.filename
        except Exception as e:
            log.warning("file read failed: %s", e)

    # 1) Ako je dosla nova slika (file ili image_url) -> vision
    try:
        if file_bytes:
            html_out, used_path, used_model = route_image_flow(
                file_bytes, razred, history, user_text=user_text, timeout_override=float(os.getenv("HARD_TIMEOUT_S", "120")), mime_hint=file_mime
            )
            # sacuvaj mali lokalni preview da se moze referencirati
            try:
                ext = os.path.splitext(file_name or "")[1].lower() or ".img"
                fname = f"{uuid4().hex}{ext}"
                with open(os.path.join(UPLOAD_DIR, fname), "wb") as fp:
                    fp.write(file_bytes)
                public_url = (request.url_root.rstrip("/") + "/uploads/" + fname)
                remember_image(public_url)
                session["last_image_url"] = public_url
            except Exception as _e:
                log.warning("could not persist local upload copy: %s", _e)

            display_user = user_text or "[SLIKA]"
            push_history_pair(display_user, html_out)
            return jsonify({"result": {"html": html_out, "path": used_path, "model": used_model}}), 200

        if image_url:
            remember_image(image_url)
            html_out, used_path, used_model = route_image_flow_url(
                image_url, razred, history, user_text=user_text, timeout_override=float(os.getenv("HARD_TIMEOUT_S", "120"))
            )
            display_user = (user_text + f" [slika: {_name_from_url(image_url)}]") if user_text else f"[slika: {_name_from_url(image_url)}]"
            push_history_pair(display_user, html_out)
            return jsonify({"result": {"html": html_out, "path": used_path, "model": used_model}}), 200
    except Exception as e:
        log.error("vision path failed: %s\n%s", e, traceback.format_exc())
        err_html = f"<div class='alert alert-danger'>Greska pri obradi slike: {html.escape(str(e))}</div>"
        push_history_pair(user_text or "[SLIKA]", err_html)
        return jsonify({"result": {"html": err_html, "path": "error", "model": "n/a"}}), 200

    # 2) Nema nove slike: ako trazimo zadatke ili follow-up sa slike -> koristi zadnju sliku
    last_url = last_image_url()
    if last_url and (wants_from_image or requested or is_followup):
        try:
            html_out, used_path, used_model = route_image_flow_url(
                last_url, razred, history, user_text=user_text, timeout_override=float(os.getenv("HARD_TIMEOUT_S", "120"))
            )
            display_user = (user_text + f" [slika: {_name_from_url(last_url)}]") if user_text else f"[slika: {_name_from_url(last_url)}]"
            push_history_pair(display_user, html_out)
            return jsonify({"result": {"html": html_out, "path": used_path, "model": used_model}}), 200
        except Exception as e:
            log.error("follow-up vision failed: %s", e)

    # 3) Cisti tekst (sa historijom u porukama)
    try:
        html_out, used_model = answer_with_text_pipeline(user_text, razred, history, requested, timeout_override=float(os.getenv("HARD_TIMEOUT_S", "120")))
        push_history_pair(user_text, html_out)
        return jsonify({"result": {"html": html_out, "path": "text", "model": used_model}}), 200
    except Exception as e:
        log.error("text pipeline failed: %s\n%s", e, traceback.format_exc())
        err_html = f"<div class='alert alert-danger'>Greska: {html.escape(str(e))}</div>"
        push_history_pair(user_text or "(prazno)", err_html)
        return jsonify({"result": {"html": err_html, "path": "error", "model": "n/a"}}), 200

# ---------------- CLEAR ----------------
@app.post("/clear")
def clear():
    if request.form.get("confirm_clear") == "1":
        session.pop("history", None)
        session.pop("razred", None)
        session.pop("last_image_url", None)
        session.pop("images", None)
    if request.form.get("ajax") == "1":
        try:
            return render_template("index.html", history=[], razred=None)
        except Exception:
            return "", 204
    return redirect(url_for("index"))

# ---------------- After-request: no-cache + CSP (opcionalno) ----------------
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

# ---------------- Main ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    log.info("Starting app on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
