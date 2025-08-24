# init_db.py  — najjednostavnije: cuva ciste e-mailove
import os, sys, psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get("EXTERNAL_DATABASE_URL") or os.environ.get("DATABASE_URL")
if not DB_URL:
    DB_URL = input("Zalijepi External ili Database URL (Render): ").strip()

def add_sslmode(url: str) -> str:
    if "sslmode=" in url: 
        return url
    return url + ("&" if "?" in url else "?") + "sslmode=require"

SQL_SCHEMA = """
create table if not exists allowed_emails (
  email text primary key
);
"""

def load_csv(path: str):
    if not os.path.exists(path): 
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            e = line.strip().lower()
            if e and "@" in e:
                out.append(e)
    return out

def seed_emails(conn, emails):
    if not emails: 
        return
    with conn.cursor() as cur:
        cur.executemany(
            "insert into allowed_emails(email) values (%s) on conflict do nothing;",
            [(e,) for e in emails]
        )

def main():
    url = add_sslmode(DB_URL)
    print("🔗 Spajam se na bazu...")
    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL_SCHEMA)
        # (opciono) ucitaj seed iz CSV-a
        csv_emails = load_csv("allowed_emails.csv")
        seed_emails(conn, csv_emails)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("select count(*) from allowed_emails;")
            n = cur.fetchone()[0]
    print(f"✅ OK: tabela allowed_emails postoji. Trenutno {n} zapisa.")
    if csv_emails:
        print("ℹ️ Ucitao sam 'allowed_emails.csv' (jedan e-mail po liniji).")

if __name__ == "__main__":
    main()
