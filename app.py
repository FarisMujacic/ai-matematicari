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
import plotly.graph_objs as go
import numpy as np
from fractions import Fraction

# Konfiguracija
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

# Prompti po razredu (primjer)
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

def extract_fractions_and_expr(text):
    frs = re.findall(r'(\d+)/(\d+)', text)
    return [Fraction(int(a), int(b)) for a, b in frs], text.replace(' ', '')

def evaluate_fraction_expression(expr_text: str) -> Fraction:
    expr = sympify(expr_text, locals={"sqrt": sqrt, "sin": sin, "cos": cos, "tan": tan, "log": log, "exp": exp})
    num, den = expr.as_numer_denom()
    return Fraction(int(num), int(den))



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'razred' not in session:
            session['razred'] = request.form.get('razred', '5')
        pitanje = request.form.get('pitanje', '')
        slika = request.files.get('slika')

        # Dodaj tekst iz slike ako postoji
        if slika and slika.filename:
            tekst_iz_slike = extract_text_from_image(slika)
            pitanje += '\n' + tekst_iz_slike

        prompt = prompti_po_razredu.get(session['razred'], prompti_po_razredu['5'])
        system_msg = {'role': 'system', 'content': prompt + ' Odgovaraj matematički precizno.'}
        messages = [system_msg, {'role': 'user', 'content': pitanje}]
        resp = client.chat.completions.create(model='gpt-4o', messages=messages)
        odgovor = resp.choices[0].message.content.strip()

        

        history = session.get('history', [])
        history.append({'user': pitanje, 'bot': odgovor})
        session['history'] = history
        sheet.append_row([pitanje, odgovor])
        return redirect(url_for('index'))

    return render_template('index.html', history=session.get('history', []), razred=session.get('razred', ''))

@app.route('/clear', methods=['POST'])
def clear():
    session.clear()
    return redirect(url_for('index'))

@app.route('/promijeni-razred', methods=['POST'])
def promijeni_razred():
    session.pop('history', None)
    session['razred'] = request.form.get('razred', '')
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True, port=5000)
