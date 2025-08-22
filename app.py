from flask import Flask, render_template, request, session, redirect, url_for
from openai import OpenAI
from dotenv import load_dotenv
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
import requests
import base64
from flask_cors import CORS
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import html

# Učitaj .env varijable
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MATHPIX_API_ID = os.getenv("MATHPIX_API_ID")
MATHPIX_API_KEY = os.getenv("MATHPIX_API_KEY")

# ===================== Model konstante (sigurni defaulti + ENV override) =====================
MODEL_TEXT = os.getenv("OPENAI_MODEL_TEXT", "gpt-4o-mini")   # tekstualni zadaci i OCR -> tekst
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-4o")    # direktan vision za geometriju/slike-u-slici
MODEL_IMAGE_CLASSIFIER = os.getenv("OPENAI_MODEL_IMAGE_CLASSIFIER", MODEL_VISION)
# =================================================================================================

app = Flask(__name__)
app.config.update(
    SESSION_COOKIE_SAMESITE=None,
    SESSION_COOKIE_SECURE=True
)
# Limit request body-a (sprječava tihi 500; prilagodi po potrebi)
MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "8"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

CORS(app, supports_credentials=True)
app.secret_key = os.getenv("SECRET_KEY", "tajna_lozinka")

# Google Sheets
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "credentials.json"
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
gs_client = gspread.authorize(creds)
sheet = gs_client.open("matematika-bot").sheet1

# Prompti po razredu
prompti_po_razredu = {
    "5": "Ti si pomoćnik iz matematike za učenike 5. razreda osnovne škole. Objašnjavaj jednostavnim i razumljivim jezikom. Pomaži učenicima da razumiju zadatke iz prirodnih brojeva, osnovnih računskih operacija, jednostavne geometrije i tekstualnih zadataka. Svako rješenje objasni jasno, korak po korak.",
    "6": "Ti si pomoćnik iz matematike za učenike 6. razreda osnovne škole. Odgovaraj detaljno i pedagoški, koristeći primjere primjerene njihovom uzrastu. Pomaži im da razumiju razlomke, decimalne brojeve, procente, geometriju i tekstualne zadatke. Objasni rješenje jasno i korak po korak.",
    "7": "Ti si pomoćnik iz matematike za učenike 7. razreda osnovne škole. Pomaži im u razumijevanju složenijih zadataka iz algebre, geometrije i funkcija. Koristi jasan, primjeren jezik i objasni svaki korak logično i precizno.",
    "8": "Ti si pomoćnik iz matematike za učenike 8. razreda osnovne škole. Fokusiraj se na linearne izraze, sisteme jednačina, geometriju i statistiku. Pomaži učenicima da razumiju postupke i objasni svako rješenje detaljno, korak po korak.",
    "9": "Ti si pomoćnik iz matematike za učenike 9. razreda osnovne škole. Pomaži im u savladavanju zadataka iz algebre, funkcija, geometrije i statistike. Koristi jasan i stručan jezik, ali primjeren njihovom nivou. Objasni svaki korak rješenja jasno i precizno."
}

def extract_text_from_image(file):
    image_data_b64 = base64.b64encode(file.read()).decode()
    headers = {
        "app_id": MATHPIX_API_ID,
        "app_key": MATHPIX_API_KEY,
        "Content-type": "application/json"
    }
    data = {
        "src": f"data:image/jpg;base64,{image_data_b64}",
        "formats": ["text"],
        "ocr": ["math", "text"]
    }
    response = requests.post("https://api.mathpix.com/v3/text", headers=headers, json=data)
    if response.ok:
        text = (response.json().get("text") or "").strip()
        confidence_hint = len(text) >= 20
        return text, confidence_hint
    else:
        return "", False

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
_trigger_re = re.compile("|".join(TRIGGER_PHRASES), flags=re.IGNORECASE)
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
    """
    Sa Thinkific/iframe ograničenjima: parsiramo samo zadnjih 5 poruka i ograničimo dužinu.
    """
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

# ===================== KLASIFIKACIJA I ROUTING ZA SLIKE =====================
def classify_image_for_flow(image_bytes: bytes) -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    messages = [
        {
            "role": "system",
            "content": (
                "Task: Determine routing for a math-helper app.\n"
                "Answer ONLY as compact JSON with keys: has_geometry (true/false), "
                "has_embedded_images (true/false), reason (short string)."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this image for routing."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }
    ]
    try:
        resp = client.chat.completions.create(model=MODEL_IMAGE_CLASSIFIER, messages=messages)
        raw = resp.choices[0].message.content.strip()
        try:
            result = json.loads(raw)
        except Exception:
            raw = re.sub(r"^```json|```$", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
            result = json.loads(raw)
        return {
            "has_geometry": bool(result.get("has_geometry", False)),
            "has_embedded_images": bool(result.get("has_embedded_images", False)),
            "reason": result.get("reason", "")
        }
    except Exception as e:
        print("Image classification failed:", e)
        return {"has_geometry": True, "has_embedded_images": True, "reason": "fallback_on_error"}

def route_image_flow(slika_bytes: bytes, razred: str, history):
    """
    Vraća: (odgovor_html, used_path, used_model)
    """
    klass = classify_image_for_flow(slika_bytes)
    print("Image classification:", klass)

    if klass["has_geometry"] or klass["has_embedded_images"]:
        image_b64 = base64.b64encode(slika_bytes).decode()
        image_prompt = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Na slici je matematički zadatak. Objasni i riješi ga korak po korak."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }

        prompt_za_razred = prompti_po_razredu.get(razred, prompti_po_razredu["5"])
        system_message = {
            "role": "system",
            "content": (
                prompt_za_razred +
                " Odgovaraj na jeziku pitanja; ako nisi siguran, koristi bosanski (ijekavica). "
                "Ne miješaj jezike i ne koristi engleske riječi u objašnjenjima. "
                "Ako nije matematika, reci: 'Molim te, postavi matematičko pitanje.' "
                "Ako ne znaš tačno rješenje, reci: 'Za ovaj zadatak se obrati instruktorima na info@matematicari.com'."
            )
        }
        messages = [system_message]
        for msg in history[-5:]:
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["bot"]})
        messages.append(image_prompt)

        resp = client.chat.completions.create(model=MODEL_VISION, messages=messages)
        raw = resp.choices[0].message.content
        raw = strip_ascii_graph_blocks(raw)
        return f"<p>{latexify_fractions(raw)}</p>", "vision_direct", MODEL_VISION

    # OCR put
    ocr_text, good = extract_text_from_image(BytesIO(slika_bytes))
    if good and ocr_text:
        prompt_za_razred = prompti_po_razredu.get(razred, prompti_po_razredu["5"])
        system_message = {
            "role": "system",
            "content": (
                prompt_za_razred +
                " Odgovaraj na jeziku pitanja; ako nisi siguran, koristi bosanski (ijekavica). "
                "Ne miješaj jezike i ne koristi engleske riječi u objašnjenjima. "
                "Ako nije matematika, reci: 'Molim te, postavi matematičko pitanje.' "
                "Ako ne znaš tačno rješenje, reci: 'Za ovaj zadatak se obrati instruktorima na info@matematicari.com'. "
                "Ne prikazuj ASCII grafove osim ako su izričito traženi."
            )
        }
        messages = [system_message]
        for msg in history[-5:]:
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["bot"]})
        messages.append({"role": "user", "content": ocr_text})

        resp = client.chat.completions.create(model=MODEL_TEXT, messages=messages)
        raw = resp.choices[0].message.content
        raw = strip_ascii_graph_blocks(raw)
        return f"<p>{latexify_fractions(raw)}</p>", "ocr_to_text", MODEL_TEXT

    # fallback na vision
    image_b64 = base64.b64encode(slika_bytes).decode()
    resp = client.chat.completions.create(
        model=MODEL_VISION,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Na slici je matematički zadatak. Objasni i riješi ga korak po korak."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }]
    )
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{latexify_fractions(raw)}</p>", "vision_direct", MODEL_VISION
# =============================================================================

@app.route("/", methods=["GET", "POST"])
def index():
    plot_expression_added = False
    razred = session.get("razred") or request.form.get("razred")
    print("Content-Length:", request.content_length, "bytes")
    history = get_history_from_request()

    if request.method == "POST":
        pitanje = request.form.get("pitanje", "")
        slika = request.files.get("slika")
        pitanje_iz_slike = ""

        # ---------- IMAGE BRANCH ----------
        if slika and slika.filename:
            slika_bytes = BytesIO(slika.read())
            slika.seek(0)

            try:
                odgovor, used_path, used_model = route_image_flow(slika_bytes.getvalue(), razred, history)
            except Exception as e:
                print("route_image_flow error:", e)
                odgovor = f"<p><b>Greška:</b> {html.escape(str(e))}</p>"
                used_path = "error"
                used_model = "n/a"

            combined_text = (pitanje or "").strip()
            will_plot = should_plot(combined_text)
            if (not plot_expression_added) and will_plot:
                expression = extract_plot_expression(combined_text, razred=razred, history=history)
                if expression:
                    odgovor = add_plot_div_once(odgovor, expression)
                    plot_expression_added = True

            history.append({
                "user": pitanje.strip() if pitanje else "[SLIKA]",
                "bot": odgovor.strip(),
            })
            session["history"] = history
            session["razred"] = razred

            mod_str = f"{used_path}|{used_model}"
            try:
                sheet.append_row([pitanje if pitanje else "[SLIKA]", odgovor, mod_str])
            except Exception as ee:
                print("Sheets append error:", ee)

            return render_template("index.html", history=history, razred=razred)

        # ---------- TEXT BRANCH ----------
        prompt_za_razred = prompti_po_razredu.get(razred, prompti_po_razredu["5"])
        system_message = {
            "role": "system",
            "content": (
                prompt_za_razred +
                " Odgovaraj na jeziku na kojem je pitanje postavljeno. Ako nisi siguran, koristi bosanski. "
                "Ne miješaj jezike i ne koristi engleske riječi u objašnjenjima. "
                "Uvijek koristi ijekavicu. Ako pitanje nije iz matematike, reci: 'Molim te, postavi matematičko pitanje.' "
                "Ako ne znaš tačno rješenje, reci: 'Za ovaj zadatak se obrati instruktorima na info@matematicari.com'."
                " Ne prikazuj ASCII ili tekstualne dijagrame koordinatnog sistema u code blockovima (```...```) "
                " osim ako korisnik eksplicitno traži ASCII dijagram. "
                " Ako korisnik nije tražio graf, nemoj crtati ni spominjati grafički prikaz."
            )
        }

        try:
            messages = [system_message]
            for msg in history[-5:]:
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["bot"]})
            messages.append({"role": "user", "content": pitanje})

            response = client.chat.completions.create(model=MODEL_TEXT, messages=messages)
            raw_odgovor = response.choices[0].message.content
            raw_odgovor = strip_ascii_graph_blocks(raw_odgovor)
            odgovor = f"<p>{latexify_fractions(raw_odgovor)}</p>"

            will_plot = should_plot(pitanje)
            if (not plot_expression_added) and will_plot:
                expression = extract_plot_expression(pitanje, razred=razred, history=history)
                if expression:
                    odgovor = add_plot_div_once(odgovor, expression)
                    plot_expression_added = True

            history.append({"user": pitanje.strip(), "bot": odgovor.strip()})
            session["history"] = history
            session["razred"] = razred

            mod_str = f"text|{MODEL_TEXT}"
            sheet.append_row([pitanje, odgovor, mod_str])

        except Exception as e:
            err_html = f"<p><b>Greška:</b> {html.escape(str(e))}</p>"
            history.append({"user": (pitanje or "").strip(), "bot": err_html})
            session["history"] = history
            session["razred"] = razred
            try:
                mod_str = f"text_error|{MODEL_TEXT}"
                sheet.append_row([pitanje, err_html, mod_str])
            except Exception as ee:
                print("Sheets append error:", ee)
            return render_template("index.html", history=history, razred=razred)

        return render_template("index.html", history=history, razred=razred)

    return render_template("index.html", history=history, razred=razred)

# ---- Error handler za prevelik upload (npr. kada Thinkific/proxy odbije) ----
@app.errorhandler(413)
def too_large(e):
    msg = f"<p><b>Greška:</b> Fajl je prevelik (limit {MAX_MB} MB). Smanji rezoluciju slike i pokušaj ponovo.</p>"
    return render_template("index.html", history=[{"user":"[SLIKA]", "bot": msg}], razred=session.get("razred")), 413

@app.route("/clear", methods=["POST"])
def clear():
    if request.form.get("confirm_clear") == "1":
        session.pop("history", None)
        session.pop("razred", None)
    return redirect("/")

@app.route("/promijeni-razred", methods=["POST"])
def promijeni_razred():
    session.pop("razred", None)
    session.pop("history", None)
    novi_razred = request.form.get("razred")
    session["razred"] = novi_razred
    return redirect(url_for("index"))

from datetime import timedelta
app.config.update(
    SESSION_COOKIE_NAME="matbot_session_v2",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    SEND_FILE_MAX_AGE_DEFAULT=0,
    ETAG_DISABLED=True,
)

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["Vary"] = "Cookie"
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

# (drugi after_request izbačen – jedan je dovoljan)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
