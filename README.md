# 🤖 MAT-BOT – AI pomoćnik za matematiku

MAT-BOT je web aplikacija koja koristi OpenAI, Mathpix i druge AI servise za pomoć učenicima osnovnih i srednjih škola u rješavanju matematičkih problema. Učenici mogu postaviti pitanje u tekstualnom obliku ili putem slike, a bot vraća detaljno objašnjenje sa mogućim grafovima, LaTeX prikazom i savjetima.

---

## ✨ Glavne funkcionalnosti

- ✅ Razumijevanje matematičkih zadataka putem OpenAI modela (GPT-4)
- ✅ Prepoznavanje zadataka sa slika pomoću Mathpix API-ja
- ✅ Prikaz rješenja u LaTeX formatu
- ✅ Prikaz grafa funkcija (JSXGraph ili Plotly)
- ✅ Interaktivno sučelje sa podrškom za više razreda i jezika

---

## 📁 Struktura projekta

.
├── app.py # Flask backend
├── list_models.py # Lista dostupnih modela
├── requirements.txt # Python zavisnosti
├── templates/
│ └── index.html # Frontend (chat UI)
├── test_env.py # Test okruženja
├── .gitignore
└── README.md # Ova datoteka


---

## ⚙️ Instalacija i pokretanje lokalno

> ⚠️ **Napomena:** Pokretanje aplikacije **nije moguće** bez validnih API ključeva za OpenAI i Mathpix. Te ključeve je potrebno dodati lokalno u `.env` fajl ili ih postaviti kao okruženjske varijable na online server (npr. [Render.com](https://render.com/)).

### 1. Kloniraj repozitorij
### 2. Instaliraj Requirements
### 3. Kreiraj .env file i unesi sljedeće podatke
OPENAI_API_KEY=ovdje_unesi_svoj_openai_kljuc
MATHPIX_API_ID=ovdje_unesi_svoj_mathpix_id
MATHPIX_API_KEY=ovdje_unesi_svoj_mathpix_kljuc
SECRET_KEY=tajna_lozinka
### 4. Pokreni aplikaciju

## 🌐 Korištenje
1)Otvori aplikaciju u browseru.
2)Unesi matematički zadatak tekstualno ili kao sliku.
3)MAT-BOT će analizirati zadatak i prikazati rješenje sa objašnjenjem, grafom i latex prikazom.

## 🛠️ Tehnologije
Python + Flask
OpenAI API (GPT-4)
Mathpix OCR API
Plotly / JSXGraph za grafove 
HTML + JavaScript + MathJax


## 👤 Autor:
Faris Mujacić


