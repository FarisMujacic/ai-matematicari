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

app = Flask(__name__)
app.config.update(
    SESSION_COOKIE_SAMESITE=None,
    SESSION_COOKIE_SECURE=True
)
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
def add_plot_div_once(odgovor_html: str, expression: str) -> str:
    """
    Ubacuje <div class="plot-request" ...> samo ako već ne postoji
    za isti izraz. Time sprječavamo dupli insert iz backenda.
    """
    marker = f'class="plot-request"'
    expr_attr = f'data-expression="{html.escape(expression)}"'
    if (marker in odgovor_html) and (expr_attr in odgovor_html):
        return odgovor_html
    return odgovor_html + f'<div class="plot-request" data-expression="{html.escape(expression)}"></div>'


# ---------- NOVO: Odlučivanje da li uopće crtati graf ----------
TRIGGER_PHRASES = [
    r"\bnacrtaj\b", r"\bnacrtati\b", r"\bcrtaj\b", r"\biscrtaj\b", r"\bskiciraj\b",
    r"\bgraf\b", r"\bgrafik\b", r"\bprika[žz]i\s+graf\b", r"\bplot\b", r"\bvizualizuj\b",
    r"\bnasrtaj\b"  # tipične slovne greške
]
NEGATION_PHRASES = [
    r"\bbez\s+grafa\b", r"\bne\s+crt(a|aj)\b", r"\bnemoj\s+crtati\b", r"\bne\s+treba\s+graf\b"
]

_trigger_re = re.compile("|".join(TRIGGER_PHRASES), flags=re.IGNORECASE)
_negation_re = re.compile("|".join(NEGATION_PHRASES), flags=re.IGNORECASE)

def should_plot(text: str) -> bool:
    """
    Crtamo SAMO ako korisnik eksplicitno traži graf i pritom nije naveo negaciju (bez grafa, ne crtati...).
    """
    if not text:
        return False
    if _negation_re.search(text):
        return False
    return _trigger_re.search(text) is not None
# --------------------------------------------------------------


def extract_plot_expression(text, razred=None, history=None):
    """
    Vraća 'y=...' samo ako u tekstu postoji EKSPPLICITAN zahtjev za grafom.
    Inače vraća None. LLM prompt je pooštren da ne vraća funkcije ako graf nije tražen.
    """
    try:
        system_message = {
            "role": "system",
            "content": (
                "Tvoja uloga je da iz korisničkog pitanja detektuješ da li KORISNIK EKSPPLICITNO TRAŽI CRTEŽ GRAFA. "
                "Ako korisnik NE traži graf, odgovori tačno 'None'. "
                "Ako traži graf, i ako je prikladno nacrtati funkciju, odgovori isključivo u obliku 'y = ...'. "
                "Ako su data jednačina/nejednačina bez traženja grafa, odgovori 'None'. "
                "Ako je tražen graf nejednačine, takođe odgovori 'None' (grafiramo samo obične funkcije kada je to eksplicitno traženo)."
                
            )
        }
        messages = [system_message]

        # (opcionalno) kratka historija konteksta
        if history:
            for msg in history[-5:]:
                messages.append({"role": "user", "content": msg["user"]})
                messages.append({"role": "assistant", "content": msg["bot"]})

        messages.append({"role": "user", "content": text})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        raw = response.choices[0].message.content.strip()
        if raw.lower() == "none":
            return None

        # Prihvati tipične forme
        cleaned = raw.replace(" ", "")
        if cleaned.startswith("y="):
            return cleaned

        # Dozvoli i f(x)=... -> prevede u y=...
        fx_match = re.match(r"f\s*\(\s*x\s*\)\s*=\s*(.+)", raw, flags=re.IGNORECASE)
        if fx_match:
            rhs = fx_match.group(1).strip()
            return "y=" + rhs.replace(" ", "")

    except Exception as e:
        print("GPT nije prepoznao funkciju za crtanje:", e)
    return None


def get_history_from_request():
    history_json = request.form.get("history_json", "")
    if history_json:
        try:
            return json.loads(history_json)
        except Exception:
            return []
    return []



@app.route("/", methods=["GET", "POST"])
def index():
 
    plot_expression_added = False
    razred = session.get("razred") or request.form.get("razred")
    print("razred u session:", session.get("razred"))

    history = get_history_from_request()

    if request.method == "POST":
        pitanje = request.form.get("pitanje", "")
        slika = request.files.get("slika")
        pitanje_iz_slike = ""

        # ---------- IMAGE BRANCH ----------
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
                    raw_odgovor = strip_ascii_graph_blocks(raw_odgovor)      # <--- NOVO
                    odgovor = f"<p>{latexify_fractions(raw_odgovor)}</p>"


                    # ---- SAMO AKO JE KORISNIK TRAŽIO GRAF ----
                    combined_text = (pitanje or "").strip()
                    will_plot = should_plot(combined_text)
                    if (not plot_expression_added) and will_plot:
                        expression = extract_plot_expression(combined_text, razred=razred, history=history)
                        if expression:
                            odgovor = add_plot_div_once(odgovor, expression)
                            plot_expression_added = True


                    # Historija
                    history.append({
                        "user": pitanje.strip() if pitanje else "[SLIKA]",
                        "bot": odgovor.strip(),
                    })
                    session["history"] = history
                    session["razred"] = razred
                    sheet.append_row([pitanje if pitanje else "[SLIKA]", odgovor])

                except Exception as e:
                    odgovor = f"<p><b>Greška:</b> {str(e)}</p>"

                return render_template("index.html", history=history, razred=razred)

        # Ako je OCR iz slike dao čitljiv tekst, pridruži ga pitanju
        if pitanje_iz_slike:
            pitanje = (pitanje + "\n" + pitanje_iz_slike).strip()

        # ---------- TEXT BRANCH ----------
        prompt_za_razred = prompti_po_razredu.get(razred, prompti_po_razredu["5"])
        system_message = {
            "role": "system",
            "content": (
                prompt_za_razred +
                " Odgovaraj na jeziku na kojem je pitanje postavljeno. Ako nisi siguran, koristi bosanski. "
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

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            raw_odgovor = response.choices[0].message.content
            raw_odgovor = strip_ascii_graph_blocks(raw_odgovor)      # <--- NOVO
            odgovor = f"<p>{latexify_fractions(raw_odgovor)}</p>"


            # ---- SAMO AKO JE KORISNIK TRAŽIO GRAF ----
            will_plot = should_plot(pitanje)
            if (not plot_expression_added) and will_plot:
                expression = extract_plot_expression(pitanje, razred=razred, history=history)
                if expression:
                    odgovor = add_plot_div_once(odgovor, expression)
                    plot_expression_added = True


            history.append({
                "user": pitanje.strip(),
                "bot": odgovor.strip(),
            })

            session["history"] = history
            session["razred"] = razred
            sheet.append_row([pitanje, odgovor])

        except Exception as e:
            odgovor = f"<p><b>Greška:</b> {str(e)}</p>"

        return render_template("index.html", history=history, razred=razred)

    return render_template("index.html", history=history, razred=razred)


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

# konfiguracija kolačića/sesije (opcionalno, ali korisno)
app.config.update(
    SESSION_COOKIE_NAME="matbot_session_v2",  # izbjegni kolizije
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    SEND_FILE_MAX_AGE_DEFAULT=0,
    ETAG_DISABLED=True,
)

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    # vrlo bitno za CDN/proxy: sadržaj zavisi od session cookie-a
    resp.headers["Vary"] = "Cookie"
    return resp


def strip_ascii_graph_blocks(text: str) -> str:
    """
    Uklanja code-blockove koji izgledaju kao ASCII graf (osi, zvjezdice, crtice…),
    npr. ono što model nekad generiše umjesto pravog grafa.
    """
    fence_re = re.compile(r"```([\s\S]*?)```", flags=re.MULTILINE)

    def looks_like_ascii_graph(block: str) -> bool:
        sample = block.strip()
        if len(sample) == 0:
            return False
        # Dozvoljeni znakovi u ASCII 'grafovima'
        allowed = set(" \t\r\n-_|*^><().,/\\0123456789xyXY")
        ratio_allowed = sum(c in allowed for c in sample) / len(sample)
        # heuristika: kratak do srednje dugačak blok, ~sastavljen od 'crtanja'
        lines = sample.splitlines()
        return (ratio_allowed > 0.9) and (3 <= len(lines) <= 40)

    def repl(m):
        block = m.group(1)
        return "" if looks_like_ascii_graph(block) else m.group(0)

    # Ukloni uvodne fraze tipa "Grafički prikaz izgleda ovako:" prije bloka
    text = re.sub(r"(Grafički prikaz.*?:\s*)?```[\s\S]*?```", 
                  lambda m: "" if "```" in m.group(0) else m.group(0),
                  text, flags=re.IGNORECASE)
    # Dodatno: prođi sve fence-ove i filtriraj heuristikom
    return fence_re.sub(repl, text)
@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp



if __name__ == "__main__":
    app.run(debug=True, port=5000)
