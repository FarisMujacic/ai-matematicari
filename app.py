from flask import Flask, render_template, request, redirect, url_for, session
from openai import OpenAI
from dotenv import load_dotenv
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Učitaj varijable iz .env
load_dotenv()

# Inicijalizuj OpenAI klijent
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicijalizuj Flask aplikaciju
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "tajna_lozinka_za_sesiju")  # za sesije

# Uputstvo za bota
system_message = {
    "role": "system",
    "content": (
        "Ti si pomoćnik za matematiku za osnovnu školu (5. do 9. razred). "
        "Učenici ti postavljaju pitanja na srpskom ili bosanskom jeziku. "
        "Odgovori na jeziku na kojem je pitanje postavljeno. Ako nisi siguran, koristi bosanski. "
        "Uvijek koristi ijekavicu, a ne ekavicu. "
        "Uvijek objasni rješenje jasno, korak po korak. "
        "Ako je pitanje iz matematike, bez obzira na to koliko je jednostavno (npr. 'jedan plus jedan'), uvijek odgovori jasno i precizno. Ako pitanje nije iz matematike, reci: 'Molim te, postavi matematičko pitanje.' "
        "Ako nisi siguran u rješenje, reci: 'Za ovaj zadatak se obrati instruktorima na info@matematicari.com' "
        "Koristi prijateljski i jednostavan ton."
    )
}

# Poveži se na Google Sheets
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "credentials.json"
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
gs_client = gspread.authorize(creds)
sheet = gs_client.open("matematika-bot").sheet1  # zamijeni ako je drugi naziv

# Zamjene brojeva i razlomaka
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
    existing_rows = sheet.get_all_values()[1:]  # preskoči header
    if not existing_rows:
        return None, None

    existing_questions = [row[0] for row in existing_rows if row]
    normalized_questions = [normalize_text(q) for q in existing_questions]

    vectorizer = TfidfVectorizer().fit_transform([user_question_norm] + normalized_questions)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

    max_index = similarities.argmax()
    max_score = similarities[max_index]

    if max_score >= threshold:
        return existing_questions[max_index], existing_rows[max_index][1]
    return None, None

# Flask ruta
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pitanje = request.form["pitanje"]
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

# Pokreni aplikaciju
if __name__ == "__main__":
    app.run(debug=True, port=5000)
