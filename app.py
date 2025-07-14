from flask import Flask, render_template, request, redirect, url_for, session
from openai import OpenAI
from dotenv import load_dotenv
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
import base64
from flask_cors import CORS

# Učitaj .env varijable
load_dotenv()

# OpenAI klijent
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Flask aplikacija
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("SECRET_KEY", "tajna_lozinka")

# Mathpix API
MATHPIX_API_ID = os.getenv("MATHPIX_API_ID")
MATHPIX_API_KEY = os.getenv("MATHPIX_API_KEY")

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

# Google Sheets konekcija
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "credentials.json"
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
gs_client = gspread.authorize(creds)
sheet = gs_client.open("matematika-bot").sheet1

# Prompti po razredima
prompti_po_razredu = {
    "5": "Ti si pomoćnik iz matematike za učenike 5. razreda osnovne škole. Odgovaraj jasno, koristeći jednostavan jezik i objašnjavaj svaki korak rješenja.",
    "6": "Ti si pomoćnik iz matematike za učenike 6. razreda osnovne škole. Odgovaraj detaljno i pedagoški, koristeći primjere koji su primjereni tom uzrastu.",
    "7": "Ti si pomoćnik iz matematike za učenike 7. razreda osnovne škole. Pomozi im u razumijevanju složenijih zadataka iz algebre, geometrije i funkcija.",
    "8": "Ti si pomoćnik iz matematike za učenike 8. razreda osnovne škole. Usredotoči se na rješavanje zadataka iz linearnog izraza, sistema jednačina, geometrije i statistike.",
    "9": "Ti si pomoćnik iz matematike za učenike 9. razreda osnovne škole. Pomozi im u složenijim zadacima iz algebre, geometrije, funkcija i statistike."
}

number_map = {
    "1": "jedan", "2": "dva", "3": "tri", "4": "četiri", "5": "pet",
    "6": "šest", "7": "sedam", "8": "osam", "9": "devet", "0": "nula",
    "1/2": "jedna polovina", "1/3": "jedna trećina", "2/3": "dvije trećine",
    "3/4": "tri četvrtine", "1/4": "jedna četvrtina"
}

def normalize_text(text):
    text = text.lower().strip()
    for fraction, word in number_map.items():
        text = re.sub(rf"\b{re.escape(fraction)}\b", word, text)
    for number, word in number_map.items():
        text = re.sub(rf"\b{word}\b", number, text)
    text = re.sub(r"\s+", " ", text)
    return text

def latexify_fractions(text):
    def zamijeni(match):
        brojilac, imenilac = match.groups()
        return f"\\(\\frac{{{brojilac}}}{{{imenilac}}}\\)"
    return re.sub(r'\b(\d{1,4})/(\d{1,4})\b', zamijeni, text)

def find_similar_question(user_question, sheet, threshold=0.85):
    user_question_norm = normalize_text(user_question)
    existing_rows = sheet.get_all_values()[1:]

    if not existing_rows:
        return None, None

    existing_questions = [row[0] for row in existing_rows if row]
    normalized_questions = [normalize_text(q) for q in existing_questions]

    # 1. 🔍 TAČNA PODUDARNOST (nema potrošnje CPU-a)
    for i, norm_q in enumerate(normalized_questions):
        if user_question_norm == norm_q:
            return existing_questions[i], existing_rows[i][1]

    # 2. 🔍 SLIČNA PITANJA (ako nema tačne podudarnosti)
    vectorizer = TfidfVectorizer().fit_transform([user_question_norm] + normalized_questions)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    max_index = similarities.argmax()
    max_score = similarities[max_index]

    if max_score >= threshold:
        return existing_questions[max_index], existing_rows[max_index][1]

    return None, None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        razred = request.form.get("razred", "5")
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
            slicno_pitanje, prethodni_odgovor = find_similar_question(pitanje, sheet)
            if prethodni_odgovor:
                odgovor = latexify_fractions(prethodni_odgovor)
            else:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[system_message, {"role": "user", "content": pitanje}]
                )
                odgovor = response.choices[0].message.content
                odgovor = latexify_fractions(odgovor)
                sheet.append_row([pitanje, odgovor])
        except Exception as e:
            odgovor = f"Greška: {str(e)}"

        session["odgovor"] = odgovor
        return redirect(url_for("index"))

    odgovor = session.pop("odgovor", "")
    return render_template("index.html", odgovor=odgovor)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
