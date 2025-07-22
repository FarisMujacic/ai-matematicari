from flask import Flask, render_template, request, redirect, url_for, session
from openai import OpenAI
from dotenv import load_dotenv
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
import requests
import base64
from flask_cors import CORS

# Učitaj .env varijable
load_dotenv()

# Inicijalizacija
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

# Funkcija za OCR sa slike
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

# Početna ruta
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Sigurno dohvati razred iz forme ili sesije
        razred = request.form.get('razred') or session.get('razred', '5')
        session['razred'] = razred

        pitanje = request.form.get('pitanje', '')
        slika = request.files.get('slika')

        if slika and slika.filename:
            tekst_iz_slike = extract_text_from_image(slika)
            pitanje += '\n' + tekst_iz_slike

        prompt = prompti_po_razredu.get(razred, prompti_po_razredu["5"])
        system_msg = {
            'role': 'system',
            'content': (
                prompt +
                " Odgovaraj matematički precizno. "
                "Ako pitanje nije iz matematike, reci: 'Postavi mi matematičko pitanje.'"
            )
        }

        messages = [system_msg, {'role': 'user', 'content': pitanje}]
        try:
            resp = client.chat.completions.create(
                model='gpt-4o',  # promijeni u 'gpt-3.5-turbo' ako je potrebno
                messages=messages
            )
            odgovor = resp.choices[0].message.content.strip()
        except Exception as e:
            odgovor = f"Greška pri odgovoru: {str(e)}"

        history = session.get('history', [])
        history.append({'user': pitanje, 'bot': odgovor})
        session['history'] = history
        sheet.append_row([pitanje, odgovor])

        return redirect(url_for('index'))

    return render_template('index.html',
                           history=session.get('history', []),
                           razred=session.get('razred', ''))

# Očisti konverzaciju
@app.route('/clear', methods=['POST'])
def clear():
    session.clear()
    return redirect(url_for('index'))

# Promijeni razred
@app.route('/promijeni-razred', methods=['POST'])
def promijeni_razred():
    session.pop('history', None)
    session.pop('razred', None)
    return redirect(url_for('index'))

# Pokreni aplikaciju
if __name__ == '__main__':
    app.run(debug=True, port=5000)
