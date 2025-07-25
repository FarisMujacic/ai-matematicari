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



@app.route("/", methods=["GET", "POST"])
def index():
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
    session.clear()
    return render_template("index.html", history=[], razred="5")


from flask import redirect, url_for

@app.route("/promijeni-razred", methods=["POST"])
def promijeni_razred():
    session.pop("razred", None)
    session.pop("history", None)
    novi_razred = request.form.get("razred")
    session["razred"] = novi_razred
    return redirect(url_for("index"))  





import re
import io
import math
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template, request
from sympy import gcd

@app.route("/proba", methods=["GET", "POST"])
def proba():
    rezultat = ""
    img_data = ""

    if request.method == "POST":
        unos = request.form.get("razlomak", "")
        match = re.match(r"\s*(\d+)\s*/\s*(\d+)\s*", unos)

        if match:
            a = int(match.group(1))
            b = int(match.group(2))

            if b == 0:
                rezultat = "<p><b>Greška:</b> Imenilac ne može biti 0.</p>"
            else:
                nzd = int(gcd(a, b))
                a_s = a // nzd
                b_s = b // nzd

                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                fig.subplots_adjust(wspace=0.5)

                # 🔶 Lijeva strana (a/b)
                axs[0].set_title(f"{a}/{b}", fontsize=16, color="red")
                if b <= 10:
                    for i in range(b):
                        face = "yellow" if i < a else "white"
                        rect = plt.Rectangle((i, 0), 1, 1,
                                             facecolor=face,
                                             edgecolor="red", linewidth=2)
                        axs[0].add_patch(rect)
                    axs[0].set_xlim(0, b)
                    axs[0].set_ylim(0, 1)
                else:
                    cols_b = math.ceil(math.sqrt(b))
                rows_b = math.ceil(b / cols_b)

                cols_b = math.ceil(math.sqrt(b))
                rows_b = math.ceil(b / cols_b)

                for i in range(b):
                    col = i % cols_b
                    row = i // cols_b
                    face = "yellow" if i < a else "white"
                    rect = plt.Rectangle((col, -row), 1, 1,
                                        facecolor=face,
                                        edgecolor="red", linewidth=2)
                    axs[0].add_patch(rect)

                axs[0].set_xlim(0, cols_b)
                axs[0].set_ylim(-rows_b, 0)
                axs[0].set_aspect("equal")
                axs[0].axis("off")



                # 🔷 Desna strana (a_s/b_s)
                axs[1].set_title(f"{a_s}/{b_s}", fontsize=16, color="red")
                if b_s <= 10:
                    for i in range(b_s):
                        face = "yellow" if i < a_s else "white"
                        rect = plt.Rectangle((i, 0), 1, 1,
                                             facecolor=face,
                                             edgecolor="red", linewidth=2)
                        axs[1].add_patch(rect)
                    axs[1].set_xlim(0, b_s)
                    axs[1].set_ylim(0, 1)
                else:
                    cols_s = math.ceil(math.sqrt(b_s))
                    rows_s = math.ceil(b_s / cols_s)
                    for i in range(b_s):
                        col = i % cols_s
                        row = i // cols_s
                        face = "yellow" if i < a_s else "white"
                        rect = plt.Rectangle((col, -row), 1, 1,
                                             facecolor=face,
                                             edgecolor="red", linewidth=2)
                        axs[1].add_patch(rect)
                    axs[1].set_xlim(0, cols_s)
                    axs[1].set_ylim(-rows_s, 0)
                axs[1].set_aspect("equal")
                axs[1].axis("off")

                # 📷 Konverzija u base64 sliku
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                img_data = base64.b64encode(buffer.read()).decode('utf-8')
                buffer.close()
                plt.close(fig)

                rezultat = (
                    f"<h3>✂️ Skraćivanje razlomka {a}/{b}:</h3>"
                    f"<p>✅ NZD: {nzd}<br>"
                    f"{a} ÷ {nzd} = {a_s}, {b} ÷ {nzd} = {b_s}<br>"
                    f"<b>👉 Rezultat: {a_s}/{b_s}</b></p>"
                )
        else:
            rezultat = "<p><b>Greška:</b> Unesi razlomak u formatu npr. 9/15.</p>"

    return render_template("proba.html", rezultat=rezultat, img_data=img_data)




 



if __name__ == "__main__":
    app.run(debug=True, port=5000)
