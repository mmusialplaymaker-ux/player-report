#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_all.py — eksportuje wynik kohorta_rocznik.sql do CSV, streamując wiersz po wierszu.
Kursor serwerowy => 4 mln wierszy nie ląduje w RAM, nie ma limitu jak w Excelu.

Dane połączenia bierze (w tej kolejności):
  1) .streamlit/secrets.toml  (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD)
  2) zmienne środowiskowe o tych samych nazwach
  3) pyta interaktywnie

Wymaga tunelu SSH do bazy (localhost:5433) aktywnego w trakcie.

Użycie (terminal VS Code):
    pip install psycopg2-binary
    python export_all.py
    python export_all.py --sql kohorta_rocznik.sql --out kohorta_ALL.csv
"""
import argparse
import csv
import os
import sys
import time

try:
    import psycopg2
except ImportError:
    sys.exit("Brak psycopg2. Zainstaluj:  pip install psycopg2-binary")


def _load_secrets():
    cfg = {}
    path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(path):
        try:
            import tomllib
            with open(path, "rb") as f:
                data = tomllib.load(f)
            cfg = {k: str(v) for k, v in data.items() if isinstance(v, (str, int))}
            print(f"  wczytano dane połączenia z {path}")
        except Exception as e:
            print(f"  (nie udało się wczytać {path}: {e})")
    return cfg


def _param(cfg, key, prompt, default=None, secret=False):
    val = cfg.get(key) or os.environ.get(key)
    if val:
        return str(val)
    import getpass
    if secret:
        return getpass.getpass(f"{prompt}: ")
    suffix = f" [{default}]" if default else ""
    got = input(f"{prompt}{suffix}: ").strip()
    return got or (default or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql", default="kohorta_rocznik.sql")
    ap.add_argument("--out", default="kohorta_ALL.csv")
    ap.add_argument("--encoding", default="utf-8-sig",
                    help="kodowanie pliku wyjściowego (domyślnie utf-8-sig — poprawne polskie i ukraińskie znaki)")
    a = ap.parse_args()

    if not os.path.exists(a.sql):
        sys.exit(f"Brak pliku {a.sql}.")
    query = open(a.sql, encoding="utf-8").read().strip().rstrip(";")

    cfg = _load_secrets()
    conn_kw = dict(
        host=_param(cfg, "PGHOST", "Host", "localhost"),
        port=_param(cfg, "PGPORT", "Port", "5433"),
        dbname=_param(cfg, "PGDATABASE", "Baza"),
        user=_param(cfg, "PGUSER", "Użytkownik"),
        password=_param(cfg, "PGPASSWORD", "Hasło", secret=True),
    )
    print(f"\nŁączę z {conn_kw['user']}@{conn_kw['host']}:{conn_kw['port']}/{conn_kw['dbname']} ...")
    try:
        conn = psycopg2.connect(**conn_kw)
    except Exception as e:
        sys.exit(f"Nie udało się połączyć: {e}\n"
                 "Sprawdź, czy tunel SSH do bazy (localhost:5433) jest aktywny.")

    # kursor SERWEROWY (nazwany) => streaming, wiersze schodzą partiami, nie wszystkie naraz
    cur = conn.cursor(name="export_kohorta")
    cur.itersize = 50_000
    print("Wykonuję zapytanie (to może chwilę potrwać przy dużym zakresie) ...")
    cur.execute(query)

    n, t0 = 0, time.time()
    with open(a.out, "w", newline="", encoding=a.encoding) as f:
        w = csv.writer(f)
        header_written = False
        while True:
            rows = cur.fetchmany(cur.itersize)
            if not rows:
                break
            if not header_written:
                w.writerow([d[0] for d in cur.description])
                header_written = True
            w.writerows(rows)
            n += len(rows)
            print(f"\r  zapisano {n:,} wierszy ...".replace(",", " "), end="", flush=True)

    cur.close()
    conn.close()
    dt = time.time() - t0
    size = os.path.getsize(a.out) / 1e6
    print(f"\n✓ {a.out}  {size:.0f} MB, {n:,} wierszy, {dt:.0f}s".replace(",", " "))
    print("Teraz:  python precompute.py " + a.out)


if __name__ == "__main__":
    main()