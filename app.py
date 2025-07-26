from flask import Flask, render_template, request, session
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
import html  # za HTML-escape u data-latex



# Učitaj .env varijable
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MATHPIX_API_ID = os.getenv("MATHPIX_API_ID")
MATHPIX_API_KEY = os.getenv("MATHPIX_API_KEY")

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("SECRET_KEY", "tajna_lozinka")

# Google Sheets autorizacija
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
    image_data = base64.b64encode(file.read()).decode()
    headers = {
        "app_id": MATHPIX_API_ID,
        "app_key": MATHPIX_API_KEY,
        "Content-type": "application/json"
    }
    data = {
        "src": f"data:image/jpg;base64,{image_data}",
        "formats": ["text"],
        "ocr": ["math", "text"]
    }
    response = requests.post("https://api.mathpix.com/v3/text", headers=headers, json=data)
    if response.ok:
        text = response.json().get("text", "")
        return text.strip(), len(text.strip()) > 20
    else:
        return "", False

def latexify_fractions(text):
    def zamijeni(match):
        brojilac, imenilac = match.groups()
        return f"\\(\\frac{{{brojilac}}}{{{imenilac}}}\\)"
    return re.sub(r'\b(\d{1,4})/(\d{1,4})\b', zamijeni, text)
# --- Desmos helperi ---

# --- Desmos helperi (CIJELI BLOK ZAMIJENJEN) ---

GRAPH_BOUNDS_DEFAULT = {"left": -10, "right": 10, "top": 10, "bottom": -10}

# mapiranje funkcija u Desmos LaTeX
FUNC_MAP = {
    'sin': r'\sin', 'cos': r'\cos', 'tan': r'\tan', 'tg': r'\tan',
    'cot': r'\cot', 'ctg': r'\cot', 'sec': r'\sec', 'csc': r'\csc',
    'arcsin': r'\arcsin', 'arccos': r'\arccos', 'arctan': r'\arctan',
    'ln': r'\ln', 'log': r'\log', 'exp': r'\exp'
}
FUNC_RX = r'(?<!\\)\b(arcsin|arccos|arctan|sin|cos|tan|tg|cot|ctg|sec|csc|ln|log|exp)\b'

def normalize_for_desmos(latex: str) -> str:
    """Pretvori 'y=sinx', 'y=sin x', 'y=lnx', 'y=sqrt x' u validan Desmos LaTeX."""
    s = (latex or '').strip()

    # osiguraj "y=" na početku
    if not re.match(r'^\s*y\s*=', s, flags=re.I):
        s = 'y=' + s

    # sqrt(...) ili sqrt x -> \sqrt{...}
    s = re.sub(r'(?<!\\)\bsqrt\s*\(\s*([^)]+)\s*\)', r'\\sqrt{\1}', s, flags=re.I)
    s = re.sub(r'(?<!\\)\bsqrt\s+([A-Za-z0-9x^+\-*/]+)', r'\\sqrt{\1}', s, flags=re.I)
    s = re.sub(r'√\s*([A-Za-z0-9x^+\-*/]+)', r'\\sqrt{\1}', s)

    # 1) ako je već sin(x) bez backslasha -> dodaj backslash
    s = re.sub(FUNC_RX + r'(?=\()', lambda m: FUNC_MAP[m.group(1).lower()], s, flags=re.I)

    # 2) slučajevi bez zagrada: "sin x", "sinx", "lnx", "log x", "cos2x" (osnovno)
    def _fn_arg(m):
        name = m.group(1).lower()
        arg = (m.group(2) or 'x').strip()
        return f"{FUNC_MAP[name]}({arg})"

    # sin x   / ln x
    s = re.sub(FUNC_RX + r'\s+([A-Za-z0-9]*x(?:\^\d+)?)\b', _fn_arg, s, flags=re.I)
    # sinx    / lnx
    s = re.sub(FUNC_RX + r'([A-Za-z0-9]*x(?:\^\d+)?)\b', _fn_arg, s, flags=re.I)

    # zamijeni tg/ctg koji su možda već backslashovani
    s = s.replace(r'\tg', r'\tan').replace(r'\ctg', r'\cot')

    # počisti višak razmaka
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _extract_latex_for_graph(text):
    """Iz teksta nađi najjednostavniji oblik: y=..., f(x)=..., ili izraz sa x nakon 'nacrtaj/graf'."""
    if not text:
        return None
    t = text.strip()

    m = re.search(r'\b(y\s*=\s*[^;\n\r]+)', t, flags=re.I)
    if m:
        return m.group(1)  # NE briši razmake

    m = re.search(r'\bf\s*\(\s*x\s*\)\s*=\s*([^;\n\r]+)', t, flags=re.I)
    if m:
        return 'y=' + m.group(1)

    if re.search(r'\b(graf|grafik|nacrtaj|nacrtati|plot)\b', t, flags=re.I):
        m = re.search(r'([\-+*/\d.\s]*x(?:\^\d+)?[^\n\r;]*)', t)
        if m:
            expr = m.group(1).strip()
            return f'y={expr}' if not expr.startswith('y=') else expr

    return None

def _strip_tags(html_text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', html_text or '')

def maybe_add_desmos_graph(bot_odgovor, korisnikov_upit, bounds=None):
    """Ako prepoznamo funkciju, dodaj <div class="desmos-calculator" ...> u odgovor."""
    # izvuci kandidat iz upita ili iz "plain" varijante bot_odgovor-a
    latex = _extract_latex_for_graph(korisnikov_upit) or _extract_latex_for_graph(_strip_tags(bot_odgovor))
    if not latex:
        return bot_odgovor

    # normalizuj u validan Desmos LaTeX (npr. 'y=sinx' -> 'y=\sin(x)')
    latex = normalize_for_desmos(latex)

    b = bounds or GRAPH_BOUNDS_DEFAULT
    div = (
        f'<div class="desmos-calculator" '
        f'data-latex="{html.escape(latex, quote=True)}" '
        f"data-bounds='{json.dumps(b)}' "
        f'style=\"width:100%;height:420px;margin-top:10px;\"></div>'
    )
    return bot_odgovor + '<br><strong>Graf funkcije:</strong> ' + div


@app.route("/", methods=["GET", "POST"])
def index():
    razred = session.get("razred") or request.form.get("razred")
    print("razred u session:", session.get("razred"))

    history = get_history_from_request() or session.get("history", [])


    if request.method == "POST":
        pitanje = request.form.get("pitanje", "")
        slika = request.files.get("slika")
        pitanje_iz_slike = ""

        if slika and slika.filename:
            slika_bytes = BytesIO(slika.read())
            slika.seek(0)

            tekst_iz_slike, validan_tekst = extract_text_from_image(slika_bytes)

            if validan_tekst:
                pitanje_iz_slike = tekst_iz_slike
            else:
                image_data = base64.b64encode(slika_bytes.getvalue()).decode()
                image_prompt = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Na slici je matematički zadatak. Molim te objasni i riješi ga korak po korak."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }

                prompt_za_razred = prompti_po_razredu.get(razred, prompti_po_razredu["5"])
                system_message = {
                    "role": "system",
                    "content": (
                        prompt_za_razred +
                        " Odgovaraj na jeziku na kojem je pitanje postavljeno. Ako nisi siguran, koristi bosanski. "
                        "Uvijek koristi ijekavicu. Ako pitanje nije iz matematike, reci: 'Molim te, postavi matematičko pitanje.' "
                        "Ako ne znaš tačno rješenje, reci: 'Za ovaj zadatak se obrati instruktorima na info@matematicari.com'."
                                            )
                }
                messages = [system_message]
                for msg in history[-5:]:
                    messages.append({"role": "user", "content": msg["user"]})
                    messages.append({"role": "assistant", "content": msg["bot"]})
                messages.append(image_prompt)

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages
                    )
                    raw_odgovor = response.choices[0].message.content
                    odgovor = f"<p>{latexify_fractions(raw_odgovor)}</p>"
                    odgovor = maybe_add_desmos_graph(odgovor, (pitanje or "") + "\n" + (pitanje_iz_slike or ""))
# ⬆️ DODAJ OVU LINIJU


                    history.append({"user": "[SLIKA]", "bot": odgovor.strip()})
                    session["history"] = history
                    session["razred"] = razred
                    sheet.append_row(["[SLIKA]", odgovor])

                except Exception as e:
                    odgovor = f"<p><b>Greška:</b> {str(e)}</p>"

                return render_template("index.html", history=history, razred=razred)

        if pitanje_iz_slike:
            pitanje += "\n" + pitanje_iz_slike

        prompt_za_razred = prompti_po_razredu.get(razred, prompti_po_razredu["5"])
        system_message = {
            "role": "system",
            "content": (
                prompt_za_razred +
                " Odgovaraj na jeziku na kojem je pitanje postavljeno. Ako nisi siguran, koristi bosanski. "
                "Uvijek koristi ijekavicu. Ako pitanje nije iz matematike, reci: 'Molim te, postavi matematičko pitanje.' "
                "Ako ne znaš tačno rješenje, reci: 'Za ovaj zadatak se obrati instruktorima na info@matematicari.com'."
            )
        }

        try:
            messages = [system_message]
            for msg in history[-5:]:
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["bot"]})

            messages.append({"role": "user", "content": pitanje})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            raw_odgovor = response.choices[0].message.content
            odgovor = f"<p>{latexify_fractions(raw_odgovor)}</p>"
            odgovor = maybe_add_desmos_graph(odgovor, pitanje)
# ⬆️ DODAJ OVU LINIJU


            history.append({"user": pitanje.strip(), "bot": odgovor.strip()})
            session["history"] = history
            session["razred"] = razred
            sheet.append_row([pitanje, odgovor])

        except Exception as e:
            odgovor = f"<p><b>Greška:</b> {str(e)}</p>"

        return render_template("index.html", history=history, razred=razred)

    return render_template("index.html", history=history, razred=razred)

@app.route("/clear", methods=["POST"])
def clear():
    # OBRIŠI SAMO HISTORIJU — ZADRŽI IZABRANI RAZRED
    session.pop("history", None)
    return redirect(url_for("index"))



from flask import redirect, url_for

@app.route("/promijeni-razred", methods=["POST"])
def promijeni_razred():
    session.pop("razred", None)
    session.pop("history", None)
    novi_razred = request.form.get("razred")
    session["razred"] = novi_razred
    return redirect(url_for("index"))  

def get_history_from_request():
    history_json = request.form.get("history_json", "")
    if history_json:
        try:
            return json.loads(history_json)
        except Exception:
            return []
    return []




if __name__ == "__main__":
    app.run(debug=True, port=5000)
