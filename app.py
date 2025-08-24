from flask import Flask, render_template, request, session, redirect, url_for
from dotenv import load_dotenv
import os, re, base64, json, html
from datetime import timedelta
from io import BytesIO
from functools import wraps

from openai import OpenAI
from flask_cors import CORS

# Google Sheets (opciono; neće srušiti ako credentials nema)
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ================== ENV ==================
load_dotenv(override=True)
# ========================================

# ---------- JEDNOSTAVNI LOGIN (lokalno) ----------
# Dozvoljeni emailovi i pristupni kod – ZA TESTIRANJE
ALLOWED_EMAILS = {
    "ucenik1@example.com",
    # "ucenik2@example.com",
}
ACCESS_CODE = "MATH-2025"  # promijeni po želji

def require_login(fn):
    @wraps(fn)
    def w(*a, **kw):
        if not session.get("user_email"):
            return redirect(url_for("prijava"))
        return fn(*a, **kw)
    return w
# --------------------------------------------------

# OpenAI klijent
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Modeli (možeš prepisati kroz .env)
MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-5-mini")
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-5")

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

# -------- Flask app / session --------
SECURE_COOKIES = os.getenv("COOKIE_SECURE", "0") == "1"  # lokalno False; na Render postavi 1
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

# Limit request body-a (sprječava tihi 500; prilagodi po potrebi)
MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# Google Sheets (opciono)
try:
    SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    CREDS_FILE = "credentials.json"
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    gs_client = gspread.authorize(creds)
    sheet = gs_client.open("matematika-bot").sheet1
except Exception as _:
    sheet = None

# Prompti po razredu
prompti_po_razredu = {
    "5": "Ti si pomoćnik iz matematike za učenike 5. razreda osnovne škole. Objašnjavaj jednostavnim i razumljivim jezikom. Pomaži učenicima da razumiju zadatke iz prirodnih brojeva, osnovnih računskih operacija, jednostavne geometrije i tekstualnih zadataka. Svako rješenje objasni jasno, korak po korak.",
    "6": "Ti si pomoćnik iz matematike za učenike 6. razreda osnovne škole. Odgovaraj detaljno i pedagoški, koristeći primjere primjerene njihovom uzrastu. Pomaži im da razumiju razlomke, decimalne brojeve, procente, geometriju i tekstualne zadatke. Objasni rješenje jasno i korak po korak.",
    "7": "Ti si pomoćnik iz matematike za učenike 7. razreda osnovne škole. Pomaži im u razumijevanju složenijih zadataka iz algebre, geometrije i funkcija. Koristi jasan, primjeren jezik i objasni svaki korak logično i precizno.",
    "8": "Ti si pomoćnik iz matematike za učenike 8. razreda osnovne škole. Fokusiraj se na linearne izraze, sisteme jednačina, geometriju i statistiku. Pomaži učenicima da razumiju postupke i objasni svako rješenje detaljno, korak po korak.",
    "9": "Ti si pomoćnik iz matematike za učenike 9. razreda osnovne škole. Pomaži im u savladavanju zadataka iz algebre, funkcija, geometrije i statistike. Koristi jasan i stručan jezik, ali primjeren njihovom nivou. Objasni svaki korak rješenja jasno i precizno."
}

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
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["bot"]})
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
        print("GPT nije prepoznao funkciju za crtanje:", e)
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
        print("history_json parse fail:", e)
        return []

# ---- Vision flow (slika) ----
def route_image_flow(slika_bytes: bytes, razred: str, history, requested_tasks=None):
    image_b64 = base64.b64encode(slika_bytes).decode()

    prompt_za_razred = prompti_po_razredu.get(razred, prompti_po_razredu["5"])

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
        messages.append({"role": "user", "content": msg["user"]})
        messages.append({"role": "assistant", "content": msg["bot"]})

    user_content = [{"type": "text", "text": "Na slici je matematički zadatak."}]
    if requested_tasks:
        user_content[0]["text"] += f" Riješi isključivo zadatak(e): {', '.join(map(str, requested_tasks))}."
    else:
        user_content[0]["text"] += " Riješi samo ono što korisnik izričito traži."
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    messages.append({"role": "user", "content": user_content})

    resp = client.chat.completions.create(model=MODEL_VISION, messages=messages)
    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{latexify_fractions(raw)}</p>", "vision_direct", actual_model

# ----------------- LOGIN RUTE -----------------
@app.route("/prijava", methods=["GET", "POST"])
def prijava():
    if request.method == "GET":
        # Ako je već ulogovan, nema smisla prikazivati login
        if session.get("user_email"):
            return redirect(url_for("index"))
        return render_template("prijava.html", error=None)

    # ⇩⇩ KLJUČNO: svaku novu prijavu počni čistom sesijom
    session.pop("user_email", None)

    email = (request.form.get("email") or "").strip().lower()
    code  = (request.form.get("code")  or "").strip()

    allowed = (email in ALLOWED_EMAILS)
    code_ok = (code == ACCESS_CODE)

    # mali debug u logu (ne ispisuj stvarni kod!)
    print(f"[login] email={email} allowed={allowed} code_ok={code_ok}")

    if allowed and code_ok:
        session["user_email"] = email
        return redirect(url_for("index"))

    # ne postavljamo session, ostaje izlogovan
    return render_template("prijava.html", error="Pogrešan email ili kod. Pokušaj ponovo."), 401


@app.post("/odjava")
def odjava():
    session.clear()
    return redirect(url_for("prijava"))
# ----------------------------------------------

# ---- Glavna ruta ----
@app.route("/", methods=["GET", "POST"])
@require_login
def index():
    plot_expression_added = False
    razred = session.get("razred") or request.form.get("razred")
    print("Content-Length:", request.content_length, "bytes")

    history = get_history_from_request() or session.get("history", [])

    if request.method == "POST":
        pitanje = (request.form.get("pitanje", "") or "").strip()
        slika = request.files.get("slika")
        is_ajax = request.form.get("ajax") == "1" or request.headers.get("X-Requested-With") == "XMLHttpRequest"

        # --- slika ---
        if slika and slika.filename:
            slika_bytes = BytesIO(slika.read())
            slika.seek(0)

            combined_text = pitanje
            requested = extract_requested_tasks(combined_text)

            try:
                odgovor, used_path, used_model = route_image_flow(
                    slika_bytes.getvalue(), razred, history, requested_tasks=requested
                )
            except Exception as e:
                print("route_image_flow error:", e)
                odgovor = f"<p><b>Greška:</b> {html.escape(str(e))}</p>"
                used_path = "error"; used_model = "n/a"

            # graf?
            will_plot = should_plot(combined_text)
            if (not plot_expression_added) and will_plot:
                expression = extract_plot_expression(combined_text, razred=razred, history=history)
                if expression:
                    odgovor = add_plot_div_once(odgovor, expression); plot_expression_added = True

            history.append({"user": combined_text if combined_text else "[SLIKA]", "bot": odgovor.strip()})
            session["history"] = history
            session["razred"]  = razred

            try:
                if sheet:
                    mod_str = f"{used_path}|{used_model}"
                    sheet.append_row([combined_text if combined_text else "[SLIKA]", odgovor, mod_str])
            except Exception as ee:
                print("Sheets append error:", ee)

            if is_ajax:
                return render_template("index.html", history=history, razred=razred)
            return redirect(url_for("index"))

        # --- tekst ---
        prompt_za_razred = prompti_po_razredu.get(razred, prompti_po_razredu["5"])

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

        try:
            messages = [system_message]
            for msg in history[-5:]:
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["bot"]})
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
            session["history"] = history
            session["razred"]  = razred

            try:
                if sheet:
                    mod_str = f"text|{actual_model}"
                    sheet.append_row([pitanje, odgovor, mod_str])
            except Exception as ee:
                print("Sheets append error:", ee)

        except Exception as e:
            err_html = f"<p><b>Greška:</b> {html.escape(str(e))}</p>"
            history.append({"user": pitanje, "bot": err_html})
            session["history"] = history; session["razred"] = razred
            try:
                if sheet:
                    mod_str = f"text_error|{MODEL_TEXT}"
                    sheet.append_row([pitanje, err_html, mod_str])
            except Exception as ee:
                print("Sheets append error:", ee)
            if is_ajax:
                return render_template("index.html", history=history, razred=razred)
            return redirect(url_for("index"))

        if is_ajax:
            return render_template("index.html", history=history, razred=razred)
        return redirect(url_for("index"))

    # GET
    return render_template("index.html", history=history, razred=razred)

# ---- Error handler za prevelik upload ----
@app.errorhandler(413)
def too_large(e):
    msg = f"<p><b>Greška:</b> Fajl je prevelik (limit {MAX_MB} MB). Smanji rezoluciju slike i pokušaj ponovo.</p>"
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
