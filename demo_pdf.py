# -*- coding: utf-8 -*-
"""demo_pdf.py — generuje przykładowy raport PDF (dane fikcyjne, realistyczne).
Uruchamiać z katalogu, w którym leży logo.png i raport_rocznikowy.py."""
import io
import math
import os
import textwrap
import datetime as _dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "raport_rocznikowy.py"
src = io.open(SRC, encoding="utf-8").read()

hlp = src[src.index("def _round_nice("):src.index("PZPN_CAT =")]
rek = src[src.index("# ── SILNIK WSKAZÓWEK"):
          src.index("# ─────────────────────────────────────────────────────────────────────────────\n# WYKRESY")]
blk = src[src.index("def wystepy_per_play("):src.index("def check_password():")]

ns = {"plt": plt, "np": np, "pd": pd, "io": io, "textwrap": textwrap, "_dt": _dt,
      "os": os, "math": math, "_secret": lambda k, d=None: d}
exec(hlp, ns)
exec(rek, ns)
exec(blk, ns)
build_pdf = ns["build_pdf"]

rng = np.random.default_rng(11)

IMIONA = ["Antoni", "Jakub", "Szymon", "Filip", "Franciszek", "Aleksander", "Mikołaj",
          "Wojciech", "Marcel", "Tymon", "Kacper", "Oliwier", "Adam", "Michał", "Bartosz",
          "Przemysław", "Krzysztof", "Mateusz", "Dawid", "Igor", "Nikodem", "Maksymilian"]
NAZWISKA = ["Nowak", "Kowalski", "Wiśniewski", "Wójcik", "Kowalczyk", "Kamiński", "Lewandowski",
            "Zieliński", "Szymański", "Woźniak", "Dąbrowski", "Kozłowski", "Jankowski",
            "Mazur", "Kwiatkowski", "Krawczyk", "Piotrowski", "Grabowski", "Pawłowski",
            "Michalski", "Adamczyk", "Dudek", "Zając", "Wieczorek", "Jabłoński"]

# przeciwnicy — realne kluby (CLJ U-17 to rozgrywki krajowe akademii)
RYWALE_CLJ = ["Legia Warszawa", "Lech Poznań", "Wisła Kraków", "Pogoń Szczecin",
              "Raków Częstochowa", "Górnik Zabrze", "Jagiellonia Białystok", "Zagłębie Lubin",
              "Śląsk Wrocław", "Cracovia", "Piast Gliwice", "Widzew Łódź", "Korona Kielce",
              "Miedź Legnica", "Lechia Gdańsk", "Motor Lublin"]
RYWALE_A2 = ["Escola Varsovia", "Polonia Warszawa", "Znicz Pruszków", "Legionovia Legionowo",
             "Świt Nowy Dwór", "Ursus Warszawa", "Mazur Karczew", "Wisła Płock"]
RYWALE_IV = ["Broń Radom", "Pilica Białobrzegi", "Orzeł Wierzbica", "Mazowsze Grójec",
             "Józefovia Józefów", "Naprzód Skórzec"]


def nazwiska(n, seed):
    r = np.random.default_rng(seed)
    out, used = [], set()
    while len(out) < n:
        s = f"{IMIONA[r.integers(len(IMIONA))]} {NAZWISKA[r.integers(len(NAZWISKA))]}"
        if s not in used:
            used.add(s)
            out.append(s)
    return out


ZAWODNIK = "Przemysław Nowak"
KLUB = "FC PlayMaker.pro"
ROK = 2010
MIN_MIN = 1000

# ── zawodnik: gra CLJ U-17 (2 lata w górę), A2 i seniorów w IV lidze ─────────
r = pd.Series({
    "player_id": "DEMO", "zawodnik": ZAWODNIK, "club_name": KLUB,
    "region_name": "mazowieckie", "rocznik_final": ROK, "rocznik_pewnosc": "potwierdzony",
    "eligible": True, "pctl": 0.982, "rank_nat": 310, "cohort_n": 16563,
    "pm_score": 0.641, "pm_quality": 0.32, "min_total": 1616, "mecze": 25,
    "gole": 7, "kartki": 3, "gole_per90": 0.39, "kartki_per90": 0.17,
    "forma": 0.041, "kons": 0.73, "gra_ze_starszymi": True, "roczniki_w_gore": 2,
    "clj_minutes": 1111, "senior_minutes": 84, "szczebel": 5,
    "szczebel_nazwa": "CLJ / Makroregionalna", "kategoria_glowna": "CLJ U-17",
    "play_glowna": 'CLJ U-17 "Grupa A"', "pctl_lvl": 0.88,
})

# ── mecze: CLJ U-17 (15), A2 (6), IV liga seniorów (4) ──────────────────────
def mecze():
    rows, d = [], _dt.date(2025, 8, 17)
    plan = ([('CLJ U-17 "Grupa A"', "CLJ U-17", 15, (60, 90), RYWALE_CLJ)]
            + [('I liga wojewódzka A2 Junior Starszy', "A2", 6, (45, 90), RYWALE_A2)]
            + [("Czwarta liga", "IV liga", 4, (8, 30), RYWALE_IV)])
    for play, liga, n, (lo, hi), rywale in plan:
        pula = list(rywale)
        rng.shuffle(pula)
        for i in range(n):
            d += _dt.timedelta(days=int(rng.integers(5, 11)))
            mn = int(rng.integers(lo, hi + 1))
            wynik = str(rng.choice(["wygrana", "remis", "porażka"], p=[0.56, 0.24, 0.20]))
            rows.append({
                "player_id": "DEMO", "match_date": pd.Timestamp(d), "play_name": play,
                "league_name": liga, "opponent_name": pula[i % len(pula)],
                "minutes": mn, "goals": int(rng.integers(0, 2)) if mn > 40 else 0,
                "yellow_cards": int(rng.random() < 0.12), "red_cards": 0,
                "match_result": wynik, "team_side": str(rng.choice(["gospodarz", "gość"])),
                "_sc": float(np.clip(rng.normal(0.62, 0.04), 0, 1)),
            })
    g = pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
    g["_sc"] = np.clip(np.linspace(0.60, 0.66, len(g)) + rng.normal(0, 0.02, len(g)), 0, 1)
    return g


pm_rows = mecze()

# ── Top 10 rocznika: duża pula, realne nazwiska ─────────────────────────────
nn = nazwiska(10, 5)
top_pdf = [[i + 1, nn[i], round(0.73 - i * 0.008, 3)] for i in range(10)]

# ── rozkład całego rocznika (do paska „gdzie jesteś”) ───────────────────────
dist = np.clip(np.random.default_rng(9).beta(2.4, 4.6, 16563) * 0.95, 0, 1)

# ── Twoje rozgrywki (CLJ U-17) i Twoja liga (grupa A) ───────────────────────
kn = nazwiska(10, 21)
KLUBY = ["Legia Warszawa", "Lech Poznań", "Wisła Kraków", "Pogoń Szczecin", "Raków Częstochowa",
         "Górnik Zabrze", "Jagiellonia Białystok", "Zagłębie Lubin", "Śląsk Wrocław", "Cracovia"]
kat = {"nazwa": "CLJ U-17", "pctl": 0.86, "rank": 63, "n": 1280,   # 4 grupy x 16 zespołów x 20
       "top": [[i + 1, kn[i], round(0.71 - i * 0.009, 3), KLUBY[i]] for i in range(10)]}
pn = nazwiska(9, 33)
play_top = [[i + 1, pn[i], round(0.69 - i * 0.010, 3), KLUBY[i]] for i in range(9)]
play_top.insert(3, [4, ZAWODNIK, 0.641, KLUB])
for i, row in enumerate(play_top):
    row[0] = i + 1
play = {"nazwa": 'CLJ U-17 "Grupa A"', "pctl": 0.94, "rank": 4, "n": 320,   # 16 zespołów x 20
        "top": play_top[:10]}

out = build_pdf(r, top_pdf, pm_rows, dist, ROK, MIN_MIN, kat, play)
open("raport_PRZYKLAD_Przemyslaw_Nowak.pdf", "wb").write(out)
print(f"OK: raport_PRZYKLAD_Przemyslaw_Nowak.pdf ({len(out) / 1024:.0f} KB)")
print(f"   zawodnik: {ZAWODNIK} · {KLUB} · rocznik {ROK}")
print(f"   rocznik : {r['rank_nat']}. z {r['cohort_n']}  (percentyl {r['pctl']:.3f})")
print(f"   rozgrywki: {kat['nazwa']} — {kat['rank']}. z {kat['n']}")
print(f"   liga    : {play['nazwa']} — {play['rank']}. z {play['n']}")
print(f"   mecze   : {len(pm_rows)} (CLJ U-17 15, A2 6, IV liga 4)")