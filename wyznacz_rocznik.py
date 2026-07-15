#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wyznacz_rocznik.py - weryfikuje ROCZNIK zawodnika po FLOORZE lig z historii.
SAMODZIELNY (nie wymaga app.py).

Idea: w danej lidze (sezonie) nie zagra nikt starszy niz rocznik graniczny Y.
Wiec grajac w lidze o graniczniku Y => rocznik >= Y. Bierzemy MAX(Y) po wszystkich
ligach i sezonach zawodnika = "rocznik_z_lig" (floor). To nie da sie oszukac polem
"wiek", wiec lapie bug (2011 pokazany jako 2010): jesli data mowi STARSZY niz
pozwalaja ligi (rocznik_z_daty < floor) -> sprzecznosc -> korekta w gore do floora.

Przesuniecie sezonowe: granicznik przesuwa sie o rok co sezon
   Y(liga, sezon) = baza(liga) - (2026 - rok_konca_sezonu)

Wejscie: CSV z rocznik_historia.sql (player_id, firstname, lastname, est_birth_year,
         season_id, league_name, matches, minutes).
Wyjscie: rocznik_status.csv - 1 wiersz na zawodnika.

Uzycie:
  python wyznacz_rocznik.py --hist rocznik_historia.csv
  python wyznacz_rocznik.py --hist rocznik_historia.csv --min-mecze 2
"""
import argparse
import os
import re
import sys

import pandas as pd

ROK_BAZOWY = 2026  # sezon 25/26 = mapowanie "surowe"

# dywizja -> najstarszy dopuszczalny rocznik (sezon 25/26, wg regulaminu MZPN). CLJ U-1x wg numeru.
_CAT_MAXYEAR_PATS = [
    (r'(^A1$|U-?19)', 2007), (r'(^A2$|U-?18)', 2008),
    (r'(^B1$|U-?17)', 2009), (r'(^B2$|U-?16)', 2010),
    (r'(^C1$|U-?15)', 2011), (r'(^C2$|U-?14)', 2012),
    (r'(^D1$|U-?13)', 2013), (r'(^D2$|U-?12)', 2014),
    (r'(^E1$|U-?11)', 2015), (r'(^E2$|U-?10)', 2016),
    (r'(^F1$|U-?9)',  2017), (r'(^F2$|U-?8)',  2018),
]

# prefiks season_id -> rok konca sezonu
SEASON_END_YEAR = {
    "e9d66181": 2026,  # 25/26 (biezacy)
    "4be7b40c": 2025,  # 24/25
    "29d748c8": 2024,  # 23/24
    "b004c86c": 2023,  # 22/23
    "b682af6d": 2022,  # 21/22  <- 4 sezony wstecz (musi byc zgodne z rocznik_historia.sql)
}
ENCODINGS = ("utf-8", "cp1250", "latin-1")


def _cat_max_year(name):
    """Najstarszy dopuszczalny rocznik dla dywizji (25/26). Senior/nieznane -> None."""
    n = str(name)
    for pat, y in _CAT_MAXYEAR_PATS:
        if re.search(pat, n, re.I):
            return y
    return None


def rd(path):
    for e in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=e, dtype=str, keep_default_na=False)
        except Exception:
            continue
    raise RuntimeError(f"Nie udalo sie wczytac {path}")


def _year_bound(league_name, season_id):
    """Rocznik graniczny ligi w danym sezonie (z przesunieciem), albo None."""
    base = _cat_max_year(league_name)
    if base is None:
        return None
    endy = SEASON_END_YEAR.get(str(season_id)[:8])
    if endy is None:
        return None
    return base - (ROK_BAZOWY - endy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="rocznik_historia.csv", help="CSV z rocznik_historia.sql")
    ap.add_argument("--out", default="rocznik_status.csv")
    ap.add_argument("--min-mecze", dest="min_mecze", type=int, default=2,
                    help="min. meczow w lidze, by liczyla sie do floora (chroni przed 1 blednym wpisem)")
    a = ap.parse_args()

    if not os.path.exists(a.hist):
        print(f"BLAD: brak {a.hist}. Uruchom rocznik_historia.sql i zapisz wynik jako {a.hist}.")
        sys.exit(1)

    h = rd(a.hist)
    for c in ("matches", "minutes", "est_birth_year"):
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    h["_ybound"] = [_year_bound(ln, sid) for ln, sid in zip(h["league_name"], h["season_id"])]

    rows = []
    for pid, g in h.groupby("player_id"):
        nm = f"{g['firstname'].iloc[0]} {g['lastname'].iloc[0]}".strip()
        date_z = g["est_birth_year"].dropna()
        date_z = int(date_z.iloc[0]) if len(date_z) else None

        q = g[g["_ybound"].notna() & (g["matches"].fillna(0) >= a.min_mecze)]
        if len(q):
            i = q["_ybound"].astype(int).idxmax()
            floor = int(q.loc[i, "_ybound"])
            dowod = f"{q.loc[i, 'league_name']} ({str(q.loc[i,'season_id'])[:8]}, {int(q.loc[i,'matches'])} mecz.)"
        else:
            floor, dowod = None, ""

        if floor is None:
            status, final = "BRAK_HISTORII", date_z
        elif date_z is None:
            status, final = "BRAK_DATY", floor
        elif date_z == floor:
            status, final = "POTWIERDZONY", floor          # gra wlasna kategorie - rocznik pewny
        elif abs(date_z - floor) >= 4:
            status, final = "SPRAWDZ", date_z              # ogromna roznica -> zepsuty rekord, do wgladu
        elif date_z < floor:
            status, final = "KOREKTA", floor               # za stary (niemozliwe wg lig) -> floor wygrywa
        elif date_z - floor == 1:
            status, final = "KOREKTA", floor               # o rok za mlody (czesty bug zrodla) -> floor
        else:
            status, final = "SPRAWDZ", date_z              # 2-3 lata za mlody: bug ALBO gra w gore -> feedback

        roznica = (date_z - floor) if (floor is not None and date_z is not None) else None
        if status == "SPRAWDZ" and floor is not None and date_z is not None:
            widelki = f"{min(floor, date_z)}-{max(floor, date_z)}"
        else:
            widelki = str(final) if final is not None else ""
        pewnosc = {
            "POTWIERDZONY": "pewny",
            "KOREKTA": "pewny (z historii)",
            "SPRAWDZ": f"do weryfikacji ({widelki})",
            "BRAK_HISTORII": "tylko źródło",
            "BRAK_DATY": "z historii",
        }.get(status, "?")
        rows.append({
            "player_id": pid, "zawodnik": nm,
            "rocznik_z_daty": date_z, "rocznik_z_lig": floor,
            "rocznik_final": final, "roznica_lat": roznica,
            "status": status, "pewnosc": pewnosc, "widelki": widelki,
            "dowod_floor": dowod,
        })

    out = pd.DataFrame(rows).sort_values(["status", "roznica_lat", "zawodnik"],
                                         na_position="last")
    try:
        out.to_csv(a.out, index=False, encoding="utf-8-sig")
    except PermissionError:
        print(f"\nBLAD: nie moge zapisac {a.out} - plik jest OTWARTY w Excelu/LibreOffice.")
        print("Zamknij go i uruchom ponownie.")
        sys.exit(1)
    print(f"Zapisano {a.out}: {len(out)} zawodnikow")
    print("Rozklad statusow:")
    for k, v in out["status"].value_counts().items():
        print(f"  {k}: {v}")
    kor = out[out["status"] == "KOREKTA"]
    if len(kor):
        print(f"KOREKTY (floor wygrywa, o rok obok) wg kierunku (data-floor): "
              f"{kor['roznica_lat'].value_counts().sort_index().to_dict()}")
    rec = out[out["status"] == "SPRAWDZ"]
    if len(rec):
        print(f"SPRAWDZ (>=2 lata roznicy, do wgladu): {len(rec)}")


if __name__ == "__main__":
    main()