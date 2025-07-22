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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    "5": "Ti si pomoćnik iz matematike za učenike 5. razreda osnovne škole...",
    "6": "Ti si pomoćnik iz matematike za učenike 6. razreda osnovne škole...",
    "7": "Ti si pomoćnik iz matematike za učenike 7. razreda osnovne škole...",
    "8": "Ti si pomoćnik iz matematike za učenike 8. razreda osnovne škole...",
    "9": "Ti si pomoćnik iz matematike za učenike 9. razreda osnovne škole..."
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
        return response.json().get("text", "")
    else:
        return f"Mathpix greška: {response.text}"

def latexify_fractions(text):
    def zamijeni(match):
        brojilac, imenilac = match.groups()
        return f"\\(\\frac{{{brojilac}}}{{{imenilac}}}\\)"
    return re.sub(r'\b(\d{1,4})/(\d{1,4})\b', zamijeni, text)

@app.route("/", methods=["GET", "POST"])
def index():
    razred = session.get("razred")
    if not razred and request.method == "POST":
        razred = request.form.get("razred", "5")
        session["razred"] = razred

    if not razred:
        return render_template("index.html", history=[], razred=None)

    if request.method == "POST":
        pitanje = request.form.get("pitanje", "")
        slika = request.files.get("slika")

        if slika and slika.filename:
            tekst_iz_slike = extract_text_from_image(slika)
            pitanje += "\n" + tekst_iz_slike

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
            # Ne koristimo prošlu historiju jer session ne radi
            messages.append({"role": "user", "content": pitanje})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            raw_odgovor = response.choices[0].message.content
            odgovor = f"<h1>Odgovor:</h1><p>{latexify_fractions(raw_odgovor)}</p>"

            sheet.append_row([pitanje, odgovor])
            return render_template("index.html", history=[{"user": pitanje, "bot": odgovor}], razred=razred)

        except Exception as e:
            odgovor = f"<p><b>Greška:</b> {str(e)}</p>"
            return render_template("index.html", history=[{"user": pitanje, "bot": odgovor}], razred=razred)

    return render_template("index.html", history=[], razred=razred)



@app.route("/clear", methods=["POST"])
def clear():
    session.clear()
    return render_template("index.html", history=[], razred="5")

@app.route("/promijeni-razred", methods=["POST"])
def promijeni_razred():
    session.pop("razred", None)
    session.pop("history", None)
    return render_template("index.html", history=[], razred="5")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
