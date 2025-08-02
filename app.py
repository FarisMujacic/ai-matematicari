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
app.config.update(
    SESSION_COOKIE_SAMESITE=None,
    SESSION_COOKIE_SECURE=True
)
CORS(app, supports_credentials=True)
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

def extract_plot_expression(text, razred=None, history=None):
    try:
        system_message = {
            "role": "system",
            "content": "Tvoja uloga je da iz korisničkog pitanja detektuješ da li sadrži funkciju koju treba nacrtati. Ako postoji funkcija, odgovori samo u obliku 'y = ...'. Ako ne postoji funkcija, odgovori 'None'."
        }
        messages = [system_message]
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
        if raw.startswith("y=") or raw.startswith("y ="):
            return raw.replace(" ", "")
    except Exception as e:
        print("GPT nije prepoznao funkciju za crtanje:", e)
    return None





@app.route("/", methods=["GET", "POST"])
def index():
    plot_expression_added = False
    razred = session.get("razred") or request.form.get("razred")
    print("razred u session:", session.get("razred"))

    history = session.get("history", [])


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

                    # Dodaj plot-request ako korisnik traži crtanje
                    if not plot_expression_added:
                        expression = extract_plot_expression(pitanje, razred=razred, history=history)
                        if expression:
                            odgovor += f'<div class="plot-request" data-expression="{html.escape(expression)}"></div>'
                            print("DETJEKTOVAN IZRAZ ZA CRTANJE:", expression)
                            plot_expression_added = True


                    # Dodaj u historiju
                    history.append({
                        "user": pitanje.strip(),
                        "bot": odgovor.strip(),
                    })

                    session["history"] = history
                    session["razred"] = razred
                    sheet.append_row([pitanje, odgovor])


                    



                    
                
                    
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
           # Dodaj plot-request ako korisnik traži crtanje
            if not plot_expression_added:
                expression = extract_plot_expression(pitanje, razred=razred, history=history)
                if expression:
                    odgovor += f'<div class="plot-request" data-expression="{html.escape(expression)}"></div>'
                    print("DETJEKTOVAN IZRAZ ZA CRTANJE:", expression)
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
        session.pop("razred", None)  # ako koristiš razred u sesiji
    return redirect("/")





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
