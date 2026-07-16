"""
raport_rocznikowy.py
────────────────────
Raport rocznikowy PlayMaker: gdzie zawodnik jest na tle CAŁEGO rocznika w Polsce.
Odbiorca: zawodnik / rodzic (rodzic płaci). Werdykt + kierunek rozwoju, nie tabela metryk.

DWA TRYBY DANYCH:
  1) PRECOMPUTED (domyślny, pod chmurę): czyta małe pliki wygenerowane przez precompute.py
       • kohorta_agg.csv.gz    — 1 wiersz/zawodnik (cała kohorta → realne N)
       • kohorta_trend.csv.gz  — minimalne wiersze meczowe do wykresu formy
     Apka nie ładuje całego match-level — percentyle z agregatu, trend tylko dla wybranego.
  2) FALLBACK match-level (lokalnie): gdy nie ma kohorta_agg.csv.gz, liczy z kohorta.csv / DB
     tym samym wzorem (compute_pm_score z app.py).

URUCHOMIENIE:
    pip install streamlit pandas numpy plotly
    python precompute.py           # raz, lokalnie — robi male pliki z duzego kohorta.csv
    streamlit run raport_rocznikowy.py

DEEP-LINK: ?player=<player_id> otwiera od razu danego zawodnika.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import io
import math
import textwrap
import datetime as _dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# scoring i helpery z istniejącej apki (UI app.py jest pod main() → import bezpieczny)
from app import compute_pm_score, _coerce, _cat_maxyear_series, _secret

CURRENT_SEASON = _secret("PM_SEASON_ID", "e9d66181-d03e-4bb3-b889-4da848f4831d")
DATA_MODE = (_secret("PM_DATA_MODE", "csv") or "csv").lower()
MIN_MIN_DEFAULT = int(float(_secret("PM_MIN_MINUTES", "1000") or "1000"))
# Najmłodszy rocznik w ofercie (sekret PM_MAX_ROCZNIK — da się zmienić bez deployu).
# Im młodszy rocznik, tym mniej rozgrywek centralnych, więc ranking jest bardziej
# wrażliwy na dzieci grające w górę w klubach bez odpowiedniej kategorii.
MAX_ROCZNIK = int(float(_secret("PM_MAX_ROCZNIK", "2012") or "2012"))

AGG_PATH = _secret("PM_AGG", "data/kohorta_agg.parquet")
MATCHES_DIR = _secret("PM_MATCHES", "data/matches")


def _round_nice(n):
    """Liczebność puli: tysiące → do tysiąca, setki → do setki. 16 600 → 17 000."""
    n = int(n)
    if n >= 1000:
        return int(round(n / 1000.0)) * 1000
    if n >= 100:
        return int(round(n / 100.0)) * 100
    return n


def _rank_bucket(rank):
    """TOP-N dla pozycji > 10. Zaokrąglamy W GÓRĘ — inaczej 'TOP 500' przy 550. miejscu
    byłoby po prostu nieprawdą. Zwraca None dla miejsc 1–10 (pokazujemy dokładne)."""
    r = int(rank)
    if r <= 10:
        return None
    if r <= 100:
        return 100
    if r < 1000:
        return int(math.ceil(r / 100.0)) * 100
    return int(math.ceil(r / 1000.0)) * 1000


def _fmt_pl(n):
    return f"{int(n):,}".replace(",", " ")


def pozycja_opis(rank, cohort_n, year):
    """Jedno zdanie o pozycji — używane w apce i w PDF (żeby nie rozjechały się wzajemnie)."""
    pula = _fmt_pl(_round_nice(cohort_n))
    b = _rank_bucket(rank)
    gdzie = f"{int(rank)}. miejsce" if b is None else f"TOP {_fmt_pl(b)}"
    return f"{gdzie} na ~{pula} zawodników rocznika {int(year)}"


def pozycja_krotko(rank, cohort_n):
    """Krótka forma do metryki w apce."""
    b = _rank_bucket(rank)
    gdzie = f"{int(rank)}. miejsce" if b is None else f"TOP {_fmt_pl(b)}"
    return f"{gdzie} / ~{_fmt_pl(_round_nice(cohort_n))}"


def _num(v, default=0.0):
    """Bezpieczna liczba: NaN/None/tekst → default."""
    try:
        f = float(v)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default


def pozycja_w_grupie_opis(rank, n):
    """Pozycja w lidze/kategorii — ZAWSZE dokładnie. W przeciwieństwie do rocznika (gdzie pula
    jest szacowana) tu wiemy dokładnie, kto gra w tych rozgrywkach, więc nie ma czego zaokrąglać."""
    return f"{int(rank)}. miejsce na {_fmt_pl(n)}"


def pozycja_w_grupie(agg, r, col, min_min, min_n=15):
    """Pozycja zawodnika wśród WSZYSTKICH (każdy rocznik), dla których ta sama kategoria
    (league_name) albo liga (play_name) jest główna. Miejsce liczymy wśród grających
    regularnie (min_min), ale mianownik to CAŁA grupa — bo tak to komunikujemy."""
    if col not in agg.columns:
        return None
    key = r.get(col)
    if key is None or (isinstance(key, float) and pd.isna(key)) or not str(key).strip():
        return None
    g_all = agg[agg[col].astype(str) == str(key)].copy()
    g_all["pm_score"] = pd.to_numeric(g_all["pm_score"], errors="coerce")
    g_all["min_total"] = pd.to_numeric(g_all["min_total"], errors="coerce")
    n_all = int(len(g_all))
    g = g_all[(g_all["min_total"].fillna(0) >= min_min) & g_all["pm_score"].notna()]
    if len(g) < min_n:
        return None
    if str(r["player_id"]) not in set(g["player_id"].astype(str)):
        rr = r.copy()
        g = pd.concat([g, rr.to_frame().T], ignore_index=True)
        g["pm_score"] = pd.to_numeric(g["pm_score"], errors="coerce")
        n_all += 1
    g["_rank"] = g["pm_score"].rank(ascending=False, method="min")
    me = g[g["player_id"].astype(str) == str(r["player_id"])]
    if not len(me):
        return None
    rank = int(me["_rank"].iloc[0])
    top = g.sort_values("pm_score", ascending=False).head(10)
    return {"nazwa": str(key), "pctl": float(max(0.0, min(1.0, 1.0 - (rank - 1) / max(n_all, 1)))),
            "rank": rank, "n": n_all,
            "top": [[i + 1, str(t.zawodnik), float(t.pm_score)]
                    for i, t in enumerate(top.itertuples())]}


def _norm_txt(v):
    return " ".join(str(v).strip().lower().split())


def _txt(v, default="—"):
    """Bezpieczny tekst do wyświetlenia. UWAGA: NaN jest w Pythonie 'truthy',
    więc `x or '—'` przepuszcza NaN i renderuje literalne 'nan' — stąd ten helper."""
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "<na>") else default

PZPN_CAT = {2006: "A1 / U-19", 2007: "A2 / U-18", 2008: "B1 / U-17", 2009: "B2 / U-16",
            2010: "C1 / U-15", 2011: "C2 / U-14", 2012: "D1 / U-13", 2013: "D2 / U-12",
            2014: "E1 / U-11", 2015: "E2 / U-10", 2016: "F1 / U-9", 2017: "F2 / U-8"}

DIMS = ["Jakość gry", "Skuteczność", "Regularność gry", "Równość formy", "Dyscyplina"]

_AGG_NUMS = ["est_birth_year", "min_total", "mecze", "gole", "kartki", "pm_score", "pm_quality",
             "gole_per90", "kartki_per90", "forma", "kons", "roczniki_w_gore",
             "clj_minutes", "senior_minutes", "szczebel"]

st.set_page_config(page_title="Raport rocznikowy · PlayMaker", layout="wide")

# wykresy bez zoomu / paska narzędzi / pełnego ekranu
PLOTLY_CFG = {"displayModeBar": False, "scrollZoom": False, "staticPlot": False}


# ─────────────────────────────────────────────────────────────────────────────
# WCZYTYWANIE
# ─────────────────────────────────────────────────────────────────────────────
def _read_csv(path):
    # pandas sam wykryje gzip po rozszerzeniu .gz
    for enc in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1")


_AGG_NUMS2 = ["rocznik_final", "min_total", "mecze", "gole", "kartki", "pm_score", "pm_quality",
              "gole_per90", "kartki_per90", "forma", "kons", "roczniki_w_gore",
              "clj_minutes", "senior_minutes", "szczebel"]


@st.cache_data(show_spinner=False)
def load_agg():
    """Warstwa rankingowa: 1 wiersz na zawodnika, wszystkie roczniki. Ładowana w całości."""
    df = pd.read_parquet(AGG_PATH)
    for c in _AGG_NUMS2:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "gra_ze_starszymi" in df.columns:
        df["gra_ze_starszymi"] = (df["gra_ze_starszymi"].astype(str).str.strip().str.lower()
                                  .isin(["true", "1", "t", "yes"]))
    df["player_id"] = df["player_id"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_player_matches(pid, rocznik_final):
    """Warstwa szczegółowa: mecze JEDNEGO zawodnika, czytane z partycji (bez ładowania reszty)."""
    empty = pd.DataFrame(columns=["player_id", "match_date", "minutes", "_sc"])
    if not os.path.exists(MATCHES_DIR):
        return empty
    flt = [("player_id", "==", str(pid))]
    try:
        if pd.notna(rocznik_final):
            flt = [("rocznik_final", "==", int(rocznik_final)), ("player_id", "==", str(pid))]
        df = pd.read_parquet(MATCHES_DIR, filters=flt)
    except Exception:
        try:
            df = pd.read_parquet(MATCHES_DIR, filters=[("player_id", "==", str(pid))])
        except Exception:
            return empty
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    for c in ("minutes", "goals", "yellow_cards", "red_cards", "_sc"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_pool(agg, declared_Y, requester):
    """Pula rówieśników = kubełek rocznik_final == declared_Y + wstrzyknięty zgłaszający.
    Zgłaszający ZAWSZE w mianowniku swojego zadeklarowanego rocznika (priorytet nad floorem)."""
    pool = agg[agg["rocznik_final"] == declared_Y].copy()
    if str(requester["player_id"]) not in set(pool["player_id"]):
        rr = requester.copy()
        rr["rocznik_final"] = declared_Y
        pool = pd.concat([pool, rr.to_frame().T], ignore_index=True)
    for c in ("pm_score", "pm_quality", "gole_per90", "kartki_per90", "min_total", "kons"):
        if c in pool.columns:
            pool[c] = pd.to_numeric(pool[c], errors="coerce")
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# AGREGACJA match-level → per zawodnik (używane TYLKO w trybie fallback)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_cohort(m):
    m = m.copy()
    m["zawodnik"] = (m["firstname"].fillna("") + " " + m["lastname"].fillna("")).str.strip()
    comp = compute_pm_score(m)
    m["_sc"] = comp["score"].values
    m["_sp"] = comp["stats_part"].values
    mn = pd.to_numeric(m["minutes"], errors="coerce").fillna(0)
    m["_mn"] = mn
    m["_maxy"] = _cat_maxyear_series(m)

    gp = m.groupby("player_id")
    den = mn.groupby(m["player_id"]).sum().replace(0, np.nan)

    def wmean(col):
        return (m[col] * mn).groupby(m["player_id"]).sum() / den

    out = pd.DataFrame(index=den.index)
    out.index.name = "player_id"
    out["zawodnik"] = gp["zawodnik"].first()
    out["est_birth_year"] = gp["est_birth_year"].max()
    out["min_total"] = mn.groupby(m["player_id"]).sum()
    out["mecze"] = gp["match_id"].nunique()
    out["gole"] = gp["goals"].sum()
    out["kartki"] = gp["yellow_cards"].sum() + gp["red_cards"].sum()
    out["pm_score"] = wmean("_sc")
    out["pm_quality"] = wmean("_sp")
    out["gole_per90"] = (out["gole"] / out["min_total"] * 90).replace([np.inf, -np.inf], np.nan)
    out["kartki_per90"] = (out["kartki"] / out["min_total"] * 90).replace([np.inf, -np.inf], np.nan)
    lead = (m.sort_values("_mn", ascending=False)
            .groupby("player_id")[["club_name", "region_name", "league_name"]].first())
    out = out.join(lead)

    def _form(g):
        x = g.sort_values("match_date")["_sp"].dropna()
        mm = x.mean()
        return pd.Series({"forma": ((x.tail(5).mean() - mm) / mm) if len(x) >= 3 and mm else np.nan,
                          "kons": (1 / (1 + x.std(ddof=0))) if len(x) >= 2 else np.nan})
    out = out.join(gp.apply(_form))

    py = m["est_birth_year"]
    jun_older = (mn > 0) & m["_maxy"].notna() & py.notna() & (py > m["_maxy"])
    out["roczniki_w_gore"] = (py - m["_maxy"]).where(jun_older).groupby(m["player_id"]).max()
    out["gra_ze_starszymi"] = jun_older.groupby(m["player_id"]).any().reindex(out.index).fillna(False)
    is_clj = m["league_name"].astype(str).str.contains(r"\bCLJ\b|Centralna Liga Junior", case=False, regex=True, na=False)
    out["clj_minutes"] = (mn * is_clj).groupby(m["player_id"]).sum()
    is_senior = (~m["is_junior_comp"].fillna(False)) & (m["age_at_match"].between(12, 19))
    out["senior_minutes"] = (mn * is_senior).groupby(m["player_id"]).sum()
    out["kategorie"] = gp["league_name"].agg(lambda s: "; ".join(sorted(set(s.dropna().astype(str)))))

    scored = m[["player_id", "match_date", "_sc", "minutes"]].copy()
    return out.reset_index(), scored


def apply_percentiles(base, min_min):
    """Pozycję liczymy wśród zawodników z wiarygodną próbą (min_min), ale ODNOSIMY ją do
    CAŁEGO rocznika — bo tak ją komunikujemy. Zawodnicy poniżej progu nie mają dość minut,
    by przypisać im miejsce, więc w tej skali są poniżej sklasyfikowanych."""
    df = base.copy()
    elig = df["min_total"].fillna(0) >= min_min
    df["eligible"] = elig
    sub = df[elig]
    n_all = max(int(len(df)), 1)
    df["cohort_n"] = n_all
    rank = sub["pm_score"].rank(ascending=False, method="min")
    df.loc[elig, "rank_nat"] = rank
    df.loc[elig, "pctl"] = (1.0 - (rank - 1) / n_all).clip(lower=0.0, upper=1.0)
    src = {"Jakość gry": "pm_quality", "Skuteczność": "gole_per90",
           "Regularność gry": "min_total", "Równość formy": "kons"}
    for lbl, col in src.items():
        df.loc[elig, lbl] = sub[col].rank(pct=True)
    df.loc[elig, "Dyscyplina"] = (-sub["kartki_per90"]).rank(pct=True)
    # percentyl w obrębie własnego szczebla (do wskazówek „czy przerastasz poziom”)
    if "szczebel" in df.columns:
        for sz, idx in df[elig & (df["szczebel"] > 0)].groupby("szczebel").groups.items():
            if len(idx) >= 20:                       # za mała grupa → brak wiarygodnego percentyla
                df.loc[idx, "pctl_lvl"] = df.loc[idx, "pm_score"].rank(pct=True)
    return df


# ── SILNIK WSKAZÓWEK ─────────────────────────────────────────────────────────
NASTEPNY_SZCZEBEL = {1: "II liga wojewódzka", 2: "II liga wojewódzka",
                     3: "I liga wojewódzka", 4: "CLJ (Centralna Liga Juniorów)"}


def rekomendacja(r, min_min):
    """(nagłówek, [kroki]) — konkretny następny krok na podstawie wyniku, szczebla i minut."""
    pctl = r.get("pctl")
    lvl = r.get("pctl_lvl")
    sz = int(r.get("szczebel") or 0)
    mins = float(r.get("min_total") or 0)
    sen = float(r.get("senior_minutes") or 0)
    clj = float(r.get("clj_minutes") or 0)
    kat = str(r.get("kategoria_glowna") or "")
    starszy = bool(r.get("gra_ze_starszymi"))

    # 1) za mało gry — najpierw minuty, dopiero potem myślenie o zmianie klubu
    if mins < min_min:
        return ("Priorytet: regularna gra", [
            f"Masz {int(mins)} min w sezonie — poniżej progu {min_min} min, od którego oceniamy w skali kraju.",
            "Zanim myślisz o zmianie klubu, powalcz o miejsce w składzie tam, gdzie jesteś.",
            "Regularność sama w sobie podnosi PM Score.",
        ])

    # 2) przerastasz swój szczebel → konkretny awans
    if sz and pd.notna(lvl) and lvl >= 0.85 and sz < 5:
        cel = NASTEPNY_SZCZEBEL.get(sz, "wyższy szczebel")
        return (f"Twój poziom jest dla Ciebie za łatwy — celuj w {cel}", [
            f"Jesteś w czołowych {(1 - lvl) * 100:.0f}% zawodników swojego szczebla "
            f"({_txt(r.get('szczebel_nazwa'))}).",
            f"Konkretny krok: testy w klubie grającym w {cel}.",
            "Jeśli nie grasz w starszym roczniku — poproś trenera o próbę." if not starszy
            else "Grasz już w starszej kategorii — trzymaj tak dalej.",
        ])

    # 3) klasa krajowa bez minut w seniorach → seniorzy
    if pd.notna(pctl) and pctl >= 0.90 and sen == 0 and kat in ("A1", "A2", "B1"):
        return ("Czas na pierwsze minuty w seniorach", [
            f"Jesteś w czołowych {(1 - pctl) * 100:.0f}% rocznika w Polsce.",
            "Poproś trenera o powołanie do kadry seniorów — nawet kilka meczów robi różnicę.",
            "Jeśli grasz poniżej CLJ, rozważ testy w akademii z CLJ." if clj == 0 else
            "Masz już minuty w CLJ — utrzymaj poziom i zbieraj minuty.",
        ])

    # 3b) krajowa czołówka, która już gra wyżej — utrzymanie kierunku, nie „dołóż minut”
    if pd.notna(pctl) and pctl >= 0.95 and (sen > 0 or clj > 0):
        gdzie = []
        if clj > 0:
            gdzie.append("grasz w CLJ")
        if sen > 0:
            gdzie.append("masz minuty w seniorach")
        return ("Jesteś w krajowej czołówce rocznika — utrzymaj kierunek", [
            f"Wyprzedzasz {pctl * 100:.0f}% rocznika w Polsce ({', '.join(gdzie)}).",
            "Priorytet: regularne minuty na najwyższym dostępnym poziomie, nie zmiana klubu.",
            "Kolejny krok to mocniejszy zespół seniorów / kadra wojewódzka." if sen > 0
            else "Kolejny krok to pierwsze minuty w seniorach.",
        ])

    # 4) mocny, ale jeszcze nie dominuje szczebla
    if pd.notna(lvl) and lvl >= 0.60:
        return ("Jesteś blisko — dołóż minut i stabilności", [
            f"W swoim szczeblu ({_txt(r.get('szczebel_nazwa'))}) wyprzedzasz "
            f"{lvl * 100:.0f}% zawodników.",
            "Celuj w pełne 90 minut i równą formę mecz po meczu.",
            "Spróbuj gry w starszym roczniku — to najszybszy sposób na wzrost PM Score." if not starszy
            else "Utrzymaj grę w starszej kategorii.",
        ])

    # 5) mocny w skali kraju, ale bez wiarygodnych danych o szczeblu
    if pd.notna(pctl) and pctl >= 0.75:
        return ("Mocny wynik w skali kraju — czas celować wyżej", [
            f"Wyprzedzasz {pctl * 100:.0f}% rocznika w Polsce.",
            "Rozważ testy w klubie z wyższego szczebla (wojewódzki, docelowo CLJ)."
            if sz == 0 else f"Rozważ testy o szczebel wyżej niż {_txt(r.get('szczebel_nazwa'))}.",
            "Poproś o minuty w starszym roczniku lub w seniorach." if sen == 0 else
            "Masz już minuty w seniorach — zbieraj ich więcej.",
        ])

    # 6) niżej w stawce — uczciwie: fundamenty, nie testy
    return ("Fundamenty: minuty, regularność, forma", [
        "Na tym etapie zmiana klubu na mocniejszy raczej nie pomoże — najpierw zbuduj przewagę tam, gdzie grasz.",
        "Cel na najbliższe tygodnie: więcej minut i równiejsze występy (mniej słabych meczów).",
        "PM Score rośnie też z poziomem rozgrywek — ale dopiero, gdy realnie na nim grasz.",
    ])


def pozycja_txt(pctl):
    """Uczciwy opis pozycji: bez mylącego 'TOP 88%' i bez 'TOP 0%' dla lidera."""
    przed = pctl * 100
    top_pct = 100 - przed
    if przed >= 50 and top_pct >= 1:
        return f"Wyprzedzasz {przed:.0f}% rocznika  ·  TOP {top_pct:.0f}%"
    return f"Wyprzedzasz {przed:.0f}% rocznika w Polsce"


# ─────────────────────────────────────────────────────────────────────────────
# WYKRESY
# ─────────────────────────────────────────────────────────────────────────────
def fig_gauge(pctl):
    top = (1 - pctl) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pctl * 100,
        number={"suffix": " pct", "font": {"size": 40}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#5db0ff"},
               "steps": [{"range": [0, 50], "color": "#20262f"},
                         {"range": [50, 80], "color": "#26303c"},
                         {"range": [80, 100], "color": "#1c3a4a"}],
               "threshold": {"line": {"color": "#f5c451", "width": 3}, "value": pctl * 100}}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#e8edf4")
    return fig, top


def fig_distribution(sub_scores, player_score):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sub_scores, nbinsx=40, marker_color="#2b3b4d", name="rocznik"))
    fig.add_vline(x=player_score, line_color="#5db0ff", line_width=3,
                  annotation_text="tu jesteś", annotation_position="top",
                  annotation_font_color="#5db0ff")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#cdd6e0", showlegend=False,
                      xaxis_title="PM Score (krajowy, league-aware)", yaxis_title="liczba zawodników")
    return fig


def fig_radar(row):
    vals = [(row.get(d) or 0) * 100 for d in DIMS]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=DIMS + [DIMS[0]],
                                    fill="toself", line_color="#5db0ff",
                                    fillcolor="rgba(93,176,255,0.25)"))
    fig.update_layout(height=340, margin=dict(l=40, r=40, t=30, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd6e0",
                      polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 100], showticklabels=True,
                                                 tickvals=[25, 50, 75, 100])))
    return fig


def fig_trend(pm_rows, cohort_median):
    g = pm_rows.sort_values("match_date")
    roll = g["_sc"].rolling(3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["match_date"], y=g["_sc"], mode="markers",
                             marker=dict(size=6, color="#3b4b5d"), name="mecz"))
    fig.add_trace(go.Scatter(x=g["match_date"], y=roll, mode="lines",
                             line=dict(color="#5db0ff", width=3), name="forma (3 mecze)"))
    if pd.notna(cohort_median):
        fig.add_hline(y=cohort_median, line_dash="dash", line_color="#f5c451",
                      annotation_text="mediana rocznika", annotation_font_color="#f5c451")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#cdd6e0", legend=dict(orientation="h", y=1.15),
                      xaxis_title="", yaxis_title="PM Score / mecz")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DOSTĘP (opcjonalne hasło — secret APP_PASSWORD)
# ─────────────────────────────────────────────────────────────────────────────
def wystepy_per_play(g):
    """Agregat występów per rozgrywki (play) + kategoria. Używane w apce i w PDF."""
    if g is None or not len(g):
        return pd.DataFrame()
    gg = g.copy()
    gg["_play"] = gg.get("play_name")
    if "_play" not in gg.columns or gg["_play"].isna().all():
        gg["_play"] = gg.get("league_name")
    mn = pd.to_numeric(gg["minutes"], errors="coerce").fillna(0)
    gl = pd.to_numeric(gg["goals"], errors="coerce").fillna(0)
    kk = (pd.to_numeric(gg.get("yellow_cards"), errors="coerce").fillna(0)
          + pd.to_numeric(gg.get("red_cards"), errors="coerce").fillna(0))
    gg = gg.assign(_mn=mn, _gl=gl, _kk=kk)
    per = (gg.groupby(["_play", "league_name"], dropna=False)
           .agg(mecze=("match_date", "count"), minuty=("_mn", "sum"), gole=("_gl", "sum"),
                kartki=("_kk", "sum"), pm=("_sc", lambda s: pd.to_numeric(s, errors="coerce").mean()))
           .reset_index().sort_values("minuty", ascending=False))
    for c in ("mecze", "minuty", "gole", "kartki"):
        per[c] = per[c].astype(int)
    return per


def build_pdf(r, top_pdf, pm_rows, dist_scores, year, min_min, kat=None, play=None):
    """PDF: str.1 = rocznik, str.2 = kategoria + liga, str.3 = sezon + mecze, str.4+ = mecze."""
    figs = [_pdf_page1(r, top_pdf, dist_scores, pm_rows, year, min_min)]
    if kat or play:
        figs.append(_pdf_page_ligi(r, kat, play, year, min_min))
    figs.append(_pdf_page2(r, pm_rows, year, min_min))
    if pm_rows is not None and len(pm_rows) > MATCH_ROWS_P2:
        rest = pm_rows.sort_values("match_date").iloc[MATCH_ROWS_P2:]
        part = 1
        while len(rest):
            part += 1
            figs.append(_pdf_matches_page(r, rest.iloc[:MATCH_ROWS_PN], year, min_min, part))
            rest = rest.iloc[MATCH_ROWS_PN:]
    buf = io.BytesIO()
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(buf) as pp:
        for f in figs:
            if f is not None:
                pp.savefig(f, facecolor=f.get_facecolor())
    for f in figs:
        if f is not None:
            plt.close(f)
    buf.seek(0)
    return buf.getvalue()


# ── paleta (jak w projekcie) ─────────────────────────────────────────────────
BG      = "#0A0A0B"
CARD    = "#161619"
CARD2   = "#1E1E22"
EDGE    = "#26262B"
TXT     = "#F5F5F7"
MUTED   = "#8B8B93"
RED     = "#E8232A"
RED_DIM = "#3A1416"
AMBER   = "#F0B429"
AMBER_D = "#3A2D12"
GREEN   = "#22A06B"
GREEN_D = "#12291F"


def _dark_fig():
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
    fig.patch.set_facecolor(BG)
    return fig


def _card(fig, x, y, w, h, fc=CARD, ec=EDGE, lw=0.8, r=0.018):
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       transform=fig.transFigure, fc=fc, ec=ec, lw=lw, zorder=0)
    fig.patches.append(p)
    return p


def _chip(fig, x, y, text, fc=CARD2, tc=TXT, ec=EDGE, fs=8.5, weight="normal", pad=0.010):
    t = fig.text(x + pad, y, text, fontsize=fs, color=tc, va="center", weight=weight, zorder=3)
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = fig.transFigure.inverted()
    w = inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
    _card(fig, x, y - 0.0125, w + 2 * pad, 0.025, fc=fc, ec=ec, r=0.012)
    return x + w + 2 * pad + 0.008


LOGO_PATH = _secret("PM_LOGO", "logo.png")   # połóż logo.png (najlepiej PNG z przezroczystością)


def _logo(fig, x, y):
    """Prawdziwe logo z pliku, a gdy go brak — rysowana wersja zastępcza."""
    if os.path.exists(LOGO_PATH):
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(LOGO_PATH)
            h_img, w_img = img.shape[0], img.shape[1]
            h = 0.030                                   # docelowa wysokość w ułamku figury
            w = h * (w_img / h_img) * (11.69 / 8.27)    # korekta proporcji A4
            ax = fig.add_axes([x, y - h / 2, w, h], zorder=5)
            ax.imshow(img)
            ax.axis("off")
            ax.patch.set_alpha(0)
            return
        except Exception:
            pass
    # fallback: biała pigułka + czerwony kwadrat z „P” + PLAYMAKER.pro
    _card(fig, x, y - 0.013, 0.200, 0.032, fc="#FFFFFF", ec="#FFFFFF", r=0.009)
    _card(fig, x + 0.011, y - 0.0085, 0.023, 0.023, fc=RED, ec=RED, r=0.006)
    fig.text(x + 0.0225, y, "P", fontsize=11, color="#FFFFFF", weight="bold",
             ha="center", va="center", zorder=6)
    fig.text(x + 0.042, y, "PLAYMAKER", fontsize=10.5, color="#0A0A0B", weight="bold",
             va="center", zorder=6)
    fig.text(x + 0.160, y - 0.0015, ".pro", fontsize=6.5, color=RED, weight="bold",
             va="center", zorder=6)


def _tile(fig, x, y, w, h, value, label):
    _card(fig, x, y, w, h, fc=CARD2, ec=EDGE, r=0.012)
    fig.text(x + w / 2, y + h * 0.60, value, fontsize=15, color=TXT, weight="bold",
             ha="center", va="center", zorder=3)
    fig.text(x + w / 2, y + h * 0.24, label, fontsize=6.4, color=MUTED, ha="center",
             va="center", zorder=3)


def _slope(pm_rows):
    if pm_rows is None or not len(pm_rows):
        return None
    y = pd.to_numeric(pm_rows.sort_values("match_date")["_sc"], errors="coerce").to_numpy()
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return None
    return np.polyfit(np.arange(len(y))[ok], y[ok], 1)[0] * 100


# ── STRONA 1 ─────────────────────────────────────────────────────────────────
def _pdf_page1(r, top_pdf, dist_scores, pm_rows, year, min_min):
    fig = _dark_fig()
    X, W = 0.09, 0.82
    pm = float(r.get("pm_score") or 0) * 100
    elig = bool(r.get("eligible"))

    _logo(fig, X, 0.947)
    fig.text(X + W, 0.947, f"SEZON {_secret('PM_SEASON_LABEL', '2025/26')}", fontsize=8.5,
             color=MUTED, weight="bold", ha="right", va="center")

    # ── HERO ──
    # chipy: rocznik, region + wyróżniki (gra ze starszymi / CLJ / seniorzy)
    extras = []
    if bool(r.get("gra_ze_starszymi")):
        # BEZ „+N”: liczba roczników w górę wynika z rocznika szacowanego z wieku,
        # więc nie jest pewna. Sam fakt gry w starszej kategorii — tak.
        extras.append(("Gra ze starszymi", RED_DIM, RED, RED))
    if _num(r.get("clj_minutes")) > 0:
        extras.append(("Minuty w CLJ", AMBER_D, AMBER, AMBER))
    if _num(r.get("senior_minutes")) > 0:
        extras.append(("Minuty w seniorach", GREEN_D, GREEN, GREEN))

    two_rows = len(extras) >= 2
    h_hero = 0.158 if two_rows else 0.132
    _card(fig, X, 0.920 - h_hero, W, h_hero)
    fig.text(X + 0.025, 0.900, "TWÓJ RAPORT", fontsize=7.5, color=RED, weight="bold")
    fig.text(X + 0.025, 0.872, _txt(r.get("zawodnik")), fontsize=21, color=TXT, weight="bold")
    fig.text(X + 0.025, 0.847, _txt(r.get("club_name"))[:44], fontsize=9, color=MUTED)

    row1, row2 = 0.818, 0.788
    cx = _chip(fig, X + 0.025, row1, f"Rocznik {int(year)}")
    cx = _chip(fig, cx, row1, _txt(r.get("region_name")))
    y_chip, xmax = row1, X + 0.470
    for lbl, fc, tc, ec in extras:
        if cx > xmax and y_chip == row1:          # nie mieści się → drugi rząd
            cx, y_chip = X + 0.025, row2
        cx = _chip(fig, cx, y_chip, lbl, fc=fc, tc=tc, ec=ec, weight="bold")

    _card(fig, X + 0.60, 0.828, 0.115, 0.068, fc="#FBE9EA", ec="#FBE9EA", r=0.012)
    fig.text(X + 0.6575, 0.862, f"{pm:.0f}", fontsize=26, color=RED, weight="bold",
             ha="center", va="center", zorder=3)
    fig.text(X + 0.6575, 0.814, "PM SCORE", fontsize=7, color=MUTED, weight="bold", ha="center")
    sl = _slope(pm_rows)
    if sl is not None:
        up = sl >= 0
        _card(fig, X + 0.728, 0.856, 0.075, 0.030, fc=(GREEN + "22") if up else AMBER_D,
              ec=GREEN if up else AMBER, r=0.010)
        fig.text(X + 0.7655, 0.871, f"{'▲' if up else '▼'} {sl:+.1f}", fontsize=9,
                 color=GREEN if up else AMBER, weight="bold", ha="center", va="center", zorder=3)

    # ── CO TO JEST PM SCORE ──
    _card(fig, X, 0.648, W, 0.126)
    fig.text(X + 0.025, 0.752, "Co to jest PM Score?", fontsize=10.5, color=TXT, weight="bold")
    fig.text(X + 0.025, 0.729, "Wskaźnik potencjału piłkarza w skali 0–100. Pokazuje:",
             fontsize=8.6, color=MUTED)
    for i, b in enumerate(["w jakiej formie jest piłkarz,",
                           "jaki ma wpływ na drużynę,",
                           "jaki prezentuje potencjał rozwojowy."]):
        fig.text(X + 0.035, 0.706 - i * 0.017, "•  " + b, fontsize=8.4, color=TXT)

    # ── GDZIE JESTEŚ ──
    _card(fig, X, 0.470, W, 0.160)
    fig.text(X + 0.025, 0.608, "Gdzie jesteś na tle rocznika?", fontsize=11.5, color=TXT, weight="bold")
    if elig:
        fig.text(X + 0.025, 0.588, "Orientacyjnie " + pozycja_opis(r["rank_nat"], r["cohort_n"], year),
                 fontsize=8.6, color=MUTED)
        # pasek gradientowy (niżej, żeby znacznik TY nie wchodził w tekst nad nim)
        axb = fig.add_axes([X + 0.025, 0.528, W - 0.05, 0.016], zorder=3)
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("pm", ["#2A2A30", "#7A1B20", RED])
        axb.imshow(grad, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
        axb.set_xticks([]); axb.set_yticks([]); axb.set_facecolor(BG)
        for sp in axb.spines.values():
            sp.set_visible(False)
        p = float(r["pctl"])
        axb.plot([p], [0.5], "o", ms=9, mfc="#FFFFFF", mec="#FFFFFF", clip_on=False, zorder=5)
        xt = X + 0.025 + p * (W - 0.05)
        _chip(fig, min(max(X + 0.025, xt - 0.018), X + W - 0.065), 0.562, "TY",
              fc="#FFFFFF", tc="#111", ec="#FFFFFF", fs=7.5, weight="bold", pad=0.007)
        fig.text(X + 0.025, 0.521, "Niższy PM Score", fontsize=7, color=MUTED, va="top")
        fig.text(X + W - 0.025, 0.521, "Wyższy PM Score", fontsize=7, color=MUTED, ha="right", va="top")
        _card(fig, X + 0.025, 0.476, W - 0.05, 0.038, fc=CARD2, ec=EDGE, r=0.012)
        przed = p * 100
        top_pct = 100 - przed
        msg = (f"Wyprzedzasz {przed:.0f}% zawodników swojego rocznika w kraju"
               + (f"  ·  TOP {top_pct:.0f}%." if przed >= 50 and top_pct >= 1 else ".")
               + "  Jak rosnąć — patrz dalej.")
        fig.text(X + 0.038, 0.495, msg, fontsize=8.4, color=TXT, va="center", zorder=3)
    else:
        fig.text(X + 0.025, 0.560, f"Za mało minut na ranking krajowy "
                 f"({int(r.get('min_total') or 0)} min, próg {min_min} min).",
                 fontsize=9, color=MUTED)

    # ── TOP 10 ──
    _card(fig, X, 0.055, W, 0.398)
    fig.text(X + 0.025, 0.428, f"Top 10 rocznika {int(year)} w Polsce", fontsize=11.5,
             color=TXT, weight="bold")
    fig.text(X + W - 0.025, 0.428, "PM Score", fontsize=7.5, color=MUTED, ha="right")
    y0, rh = 0.372, 0.0345
    for i, (rank, name, sc) in enumerate(top_pdf[:10]):
        mine = str(name) == str(r.get("zawodnik"))
        _card(fig, X + 0.020, y0 - i * rh, W - 0.040, 0.028,
              fc=RED_DIM if mine else CARD2, ec=RED if mine else EDGE, r=0.010)
        fig.text(X + 0.042, y0 - i * rh + 0.014, f"{i + 1}", fontsize=8, color=RED if mine else MUTED,
                 weight="bold", ha="center", va="center", zorder=3)
        fig.text(X + 0.062, y0 - i * rh + 0.014, str(name), fontsize=9.2,
                 color=TXT, weight="bold" if mine else "normal", va="center", zorder=3)
        fig.text(X + W - 0.042, y0 - i * rh + 0.014, f"{float(sc) * 100:.0f}", fontsize=9.5,
                 color=RED, weight="bold", ha="right", va="center", zorder=3)
    _pdf_footer(fig, X, min_min)      # nota o roczniku dotyczy właśnie tej strony
    return fig


# ── STRONA 2 ─────────────────────────────────────────────────────────────────
def _rank_card(fig, X, W, y_bot, h, naglowek, grupa, r, col=RED, col_dark="#7A1B20", col_hl="#2A1416"):
    """Karta rankingowa: tytul + pozycja + pasek z TY + Top 10 (rozgrywki albo liga).
    Kolor odróżnia trzy rankingi: czerwony rocznik, zielony rozgrywki, żółty liga."""
    _card(fig, X, y_bot, W, h)
    top_y = y_bot + h
    fig.text(X + 0.025, top_y - 0.026, naglowek, fontsize=7.5, color=col, weight="bold")
    fig.text(X + 0.025, top_y - 0.054, _txt(grupa["nazwa"])[:54], fontsize=12.5,
             color=TXT, weight="bold")
    fig.text(X + 0.025, top_y - 0.076,
             pozycja_w_grupie_opis(grupa["rank"], grupa["n"]) + " zawodników",
             fontsize=8, color=MUTED)

    by = top_y - 0.124
    axb = fig.add_axes([X + 0.025, by, W - 0.05, 0.014], zorder=3)
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pm", ["#2A2A30", col_dark, col])
    axb.imshow(grad, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
    axb.set_xticks([])
    axb.set_yticks([])
    axb.set_facecolor(BG)
    for sp in axb.spines.values():
        sp.set_visible(False)
    p = float(grupa["pctl"])
    axb.plot([p], [0.5], "o", ms=8, mfc="#FFFFFF", mec="#FFFFFF", clip_on=False, zorder=5)
    xt = X + 0.025 + p * (W - 0.05)
    _chip(fig, min(max(X + 0.025, xt - 0.018), X + W - 0.065), by + 0.030, "TY",
          fc="#FFFFFF", tc="#111", ec="#FFFFFF", fs=7, weight="bold", pad=0.006)
    fig.text(X + 0.025, by - 0.010,
             "Wyprzedzasz {:.0f}% zawodników tych rozgrywek".format(p * 100),
             fontsize=7.5, color=TXT, va="top")

    ty = by - 0.038
    fig.text(X + 0.025, ty, "Top 10", fontsize=8.5, color=TXT, weight="bold", va="top")
    fig.text(X + W - 0.025, ty, "PM Score", fontsize=6.8, color=MUTED, ha="right", va="top")
    me = _norm_txt(r.get("zawodnik"))
    yy = ty - 0.028
    for poz, nazwa, sc in grupa["top"][:10]:
        mine = _norm_txt(nazwa) == me
        if mine:
            _card(fig, X + 0.022, yy - 0.0085, W - 0.044, 0.020, fc=col_hl, ec=col, r=0.006)
        fig.text(X + 0.036, yy, str(poz), fontsize=7, color=col if mine else MUTED,
                 weight="bold", va="center", zorder=3)
        fig.text(X + 0.060, yy, str(nazwa)[:38], fontsize=7.8,
                 color=TXT if mine else "#D9D9DE", va="center", zorder=3)
        fig.text(X + W - 0.036, yy, "{:.0f}".format(sc * 100), fontsize=7.8,
                 color=col if mine else MUTED, weight="bold", ha="right", va="center", zorder=3)
        yy -= 0.0215


def _pdf_page_ligi(r, kat, play, year, min_min):
    """Strona 2: pozycja na tle wlasnej kategorii i wlasnej ligi."""
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    X, W = 0.09, 0.82
    _logo(fig, X, 0.945)
    fig.text(X + W, 0.947, "{}  ·  rocznik {}".format(_txt(r.get("zawodnik")), int(year)),
             fontsize=8.5, color=MUTED, ha="right", weight="bold")
    fig.text(X, 0.903, "Gdzie jesteś w swoich rozgrywkach?", fontsize=15,
             color=TXT, weight="bold")
    fig.text(X, 0.882, "Ranking rocznikowy porównuje Cię z rówieśnikami z całej Polski.",
             fontsize=8.2, color=MUTED)
    fig.text(X, 0.866, "Tutaj porównujemy Cię z tymi, z którymi realnie grasz.",
             fontsize=8.2, color=MUTED)
    if kat:
        _rank_card(fig, X, W, 0.462, 0.400, "TWOJE ROZGRYWKI", kat, r,
                   col=GREEN, col_dark="#12452E", col_hl="#0E2A1D")
    if play:
        _rank_card(fig, X, W, 0.045, 0.400, "TWOJA LIGA", play, r,
                   col=AMBER, col_dark="#6B4E10", col_hl="#2B2210")
    if not kat and not play:
        fig.text(X, 0.60, "Za mało danych porównawczych w Twoich rozgrywkach.",
                 fontsize=9, color=MUTED)
    _pdf_footer(fig, X, min_min)          # bez legendy W/R/P — na tej stronie nie ma meczów
    return fig


def _pdf_page2(r, pm_rows, year, min_min):
    fig = _dark_fig()
    X, W = 0.09, 0.82
    _logo(fig, X, 0.947)
    fig.text(X + W, 0.947, f"{_txt(r.get('zawodnik'))}  ·  rocznik {int(year)}", fontsize=8.5,
             color=MUTED, weight="bold", ha="right", va="center")

    # ── JAK PODBIĆ PM SCORE (dynamiczne) ──
    head, steps = rekomendacja(r, min_min)
    _card(fig, X, 0.735, W, 0.185)
    fig.text(X + 0.025, 0.898, "Jak podbić swój PM Score?", fontsize=11.5, color=TXT, weight="bold")
    fig.text(X + 0.025, 0.876, head, fontsize=9, color=RED, weight="bold")
    for i, stp in enumerate(steps[:3]):
        yy = 0.842 - i * 0.040
        _card(fig, X + 0.020, yy - 0.013, W - 0.040, 0.034, fc=CARD2, ec=EDGE, r=0.010)
        fig.text(X + 0.040, yy + 0.004, str(i + 1), fontsize=9, color=RED, weight="bold",
                 ha="center", va="center", zorder=3)
        txt = "\n".join(textwrap.wrap(str(stp), 88)[:2])   # zawijaj, nie ucinaj w pół słowa
        fig.text(X + 0.058, yy + 0.004, txt, fontsize=7.4, color=TXT, va="center",
                 linespacing=1.35, zorder=3)

    # ── TWÓJ SEZON W LICZBACH ──
    g = pm_rows.sort_values("match_date").copy() if pm_rows is not None and len(pm_rows) else pd.DataFrame()
    _card(fig, X, 0.545, W, 0.170)
    fig.text(X + 0.025, 0.693, "Twój sezon w liczbach", fontsize=11.5, color=TXT, weight="bold")
    if len(g):
        mn = pd.to_numeric(g["minutes"], errors="coerce").fillna(0)
        gl = pd.to_numeric(g["goals"], errors="coerce").fillna(0)
        yc = pd.to_numeric(g.get("yellow_cards"), errors="coerce").fillna(0)
        rc = pd.to_numeric(g.get("red_cards"), errors="coerce").fillna(0)
        sc = pd.to_numeric(g["_sc"], errors="coerce")
        ms = max(1.0, mn.sum())
        res = g["match_result"].astype(str) if "match_result" in g.columns else pd.Series([], dtype=str)
        vals = [(f"{len(g)}", "MECZE"), (f"{int(mn.sum())}", "MINUTY"), (f"{int(gl.sum())}", "GOLE"),
                (f"{sc.mean() * 100:.0f}", "ŚR. PM SCORE"),
                (f"{gl.sum() / ms * 90:.2f}", "GOLE / 90 MIN"),
                (f"{(yc.sum() + rc.sum()) / ms * 90:.2f}", "KARTKI / 90 MIN"),
                (f"{mn.mean():.0f}", "MIN / MECZ"),
                (f"{(res == 'wygrana').mean() * 100:.0f}%" if len(res) else "—", "ZWYCIĘSTWA")]
        tw, gap = 0.178, 0.019
        for k, (v, lab) in enumerate(vals):
            col, row = k % 4, k // 4
            _tile(fig, X + 0.022 + col * (tw + gap), 0.612 - row * 0.062, tw, 0.052, v, lab)

    # ── WYSTĘPY W ROZGRYWKACH (per play) ──
    per = wystepy_per_play(g)
    _card(fig, X, 0.352, W, 0.172)
    fig.text(X + 0.025, 0.500, "Występy w rozgrywkach", fontsize=11.5, color=TXT, weight="bold")
    if len(per):
        hdr = ((0.500, "KAT."), (0.585, "MECZE"), (0.655, "MIN"), (0.715, "GOLE"))
        for xx, lab in hdr:
            fig.text(X + xx, 0.478, lab, fontsize=6, color=MUTED, ha="right")
        fig.text(X + W - 0.038, 0.478, "ŚR. PM", fontsize=6, color=MUTED, ha="right")
        rh2, y1 = 0.0272, 0.456
        for k, (_, p) in enumerate(per.head(4).iterrows()):
            yy = y1 - k * rh2
            _card(fig, X + 0.020, yy - 0.009, W - 0.040, 0.023, fc=CARD2, ec=EDGE, r=0.008)
            nm = str(p["_play"])[:42]
            fig.text(X + 0.036, yy + 0.0025, nm, fontsize=7.2, color=TXT, va="center", zorder=3)
            fig.text(X + 0.500, yy + 0.0025, _txt(p["league_name"])[:9], fontsize=6.8,
                     color=MUTED, ha="right", va="center", zorder=3)
            for xx, v in ((0.585, p["mecze"]), (0.655, f"{p['minuty']}′"), (0.715, p["gole"])):
                fig.text(X + xx, yy + 0.0025, str(v), fontsize=7.4, color=TXT, ha="right",
                         va="center", zorder=3)
            fig.text(X + W - 0.038, yy + 0.0025, f"{(p['pm'] or 0) * 100:.0f}", fontsize=8,
                     color=RED, weight="bold", ha="right", va="center", zorder=3)
        if len(per) > 4:
            fig.text(X + 0.025, y1 - 4 * rh2 - 0.002, f"… oraz {len(per) - 4} innych rozgrywek",
                     fontsize=7, color=MUTED, style="italic")

    # ── WSZYSTKIE MECZE (część 1; reszta na kolejnych stronach) ──
    _card(fig, X, 0.045, W, 0.288)
    fig.text(X + 0.025, 0.308, "Wszystkie mecze sezonu", fontsize=11.5, color=TXT, weight="bold")
    if not len(g):
        fig.text(X + 0.025, 0.280, "Brak danych meczowych.", fontsize=9, color=MUTED)
        _pdf_footer(fig, X, min_min)
        return fig
    _match_rows(fig, X, W, g.iloc[:MATCH_ROWS_P2], y0=0.262)
    _pdf_footer(fig, X, min_min, mecze=True)
    return fig


MATCH_ROWS_P2 = 7       # mecze na stronie 2 (pod kaflami i rozgrywkami)
MATCH_ROWS_PN = 26      # mecze na kolejnych stronach (cała strona)
_RES = {"wygrana": ("W", GREEN), "remis": ("R", AMBER), "porażka": ("P", RED)}


def _match_rows(fig, X, W, rows, y0):
    """Rysuje wiersze meczów od y0 w dół. Nagłówki kolumn nad pierwszym wierszem."""
    rh = 0.0272
    for xx, lab in ((0.545, "MIN"), (0.625, "GOLE"), (0.700, "KARTKI")):
        fig.text(X + xx, y0 + 0.019, lab, fontsize=6, color=MUTED, ha="right")
    fig.text(X + W - 0.038, y0 + 0.019, "PM", fontsize=6, color=MUTED, ha="right")
    for k, i in enumerate(rows.index):
        yy = y0 - k * rh
        _card(fig, X + 0.020, yy - 0.009, W - 0.040, 0.023, fc=CARD2, ec=EDGE, r=0.008)
        res = str(rows.at[i, "match_result"]) if "match_result" in rows.columns else ""
        rs, rc_ = _RES.get(res, ("–", MUTED))
        fig.text(X + 0.036, yy + 0.0025, rs, fontsize=7.5, color=rc_, weight="bold",
                 ha="center", va="center", zorder=3)
        opp = (str(rows.at[i, "opponent_name"])[:30]
               if "opponent_name" in rows.columns and pd.notna(rows.at[i, "opponent_name"]) else "—")
        fig.text(X + 0.052, yy + 0.0025, opp, fontsize=7.6, color=TXT, va="center", zorder=3)
        d = pd.to_datetime(rows.at[i, "match_date"], errors="coerce")
        fig.text(X + 0.395, yy + 0.0025, d.strftime("%d.%m.%Y") if pd.notna(d) else "—",
                 fontsize=7, color=MUTED, va="center", zorder=3)
        mn_ = int(pd.to_numeric(rows.at[i, "minutes"], errors="coerce") or 0)
        gl_ = int(pd.to_numeric(rows.at[i, "goals"], errors="coerce") or 0)
        kk = int((pd.to_numeric(rows.at[i, "yellow_cards"], errors="coerce") or 0)
                 + (pd.to_numeric(rows.at[i, "red_cards"], errors="coerce") or 0))
        pmv = float(pd.to_numeric(rows.at[i, "_sc"], errors="coerce") or 0) * 100
        for xx, v in ((0.545, f"{mn_}′"), (0.625, f"{gl_}"), (0.700, f"{kk}")):
            fig.text(X + xx, yy + 0.0025, v, fontsize=7.6, color=TXT, ha="right", va="center", zorder=3)
        fig.text(X + W - 0.038, yy + 0.0025, f"{pmv:.0f}", fontsize=8, color=RED, weight="bold",
                 ha="right", va="center", zorder=3)


def _pdf_footer(fig, X, min_min, mecze=False):
    """Stopka. Legenda W/R/P tylko na stronach, gdzie faktycznie są mecze.
    Nota o roczniku zawijana, bo w jednej linii wyjeżdżała poza stronę."""
    y = 0.030
    if mecze:
        fig.text(X, y, "W = wygrana · R = remis · P = porażka · Kartki = żółte + czerwone",
                 fontsize=6.8, color=MUTED)
        y -= 0.012
    fig.text(X, y, f"Pozycja i liczebność przybliżona. Wygenerowano {_dt.date.today():%d.%m.%Y}.",
             fontsize=6.2, color=MUTED)


def _pdf_matches_page(r, rows, year, min_min, part):
    """Kolejna strona z meczami (gdy sezon nie mieści się na stronie 2)."""
    fig = _dark_fig()
    X, W = 0.09, 0.82
    _logo(fig, X, 0.947)
    fig.text(X + W, 0.947, f"{_txt(r.get('zawodnik'))}  ·  rocznik {int(year)}", fontsize=8.5,
             color=MUTED, weight="bold", ha="right", va="center")
    _card(fig, X, 0.045, W, 0.855)
    fig.text(X + 0.025, 0.873, f"Wszystkie mecze sezonu (cd. {part})", fontsize=11.5,
             color=TXT, weight="bold")
    _match_rows(fig, X, W, rows, y0=0.828)
    _pdf_footer(fig, X, min_min, mecze=True)
    return fig

def check_password():
    pw = _secret("APP_PASSWORD", "")
    if not pw:
        return True
    if st.session_state.get("_ok"):
        return True
    st.title("Raport rocznikowy · PlayMaker")
    val = st.text_input("Kod dostępu", type="password")
    if val and val == pw:
        st.session_state["_ok"] = True
        st.rerun()
    elif val:
        st.error("Błędny kod.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# DANE: agg w całości (parquet), mecze jednego gracza na żądanie
# ─────────────────────────────────────────────────────────────────────────────
PEWNOSC_OPIS = {"potwierdzony": ("Rocznik potwierdzony", "#22A06B"),
                "skorygowany": ("Rocznik skorygowany z historii lig", "#F0B429"),
                "szacowany": ("Rocznik szacowany (do potwierdzenia)", "#8B8B93")}


def pick_requester(agg):
    """Wybór zawodnika (nazwisko → lista) + zadeklarowany rocznik ze zgłoszenia."""
    qp_player = st.query_params.get("player")
    qp_rok = st.query_params.get("rocznik")

    if qp_player and str(qp_player) in set(agg["player_id"]):
        req = agg[agg["player_id"] == str(qp_player)].iloc[0]
    else:
        q = st.sidebar.text_input("Szukaj zawodnika (nazwisko)", "")
        cand = agg
        if q.strip():
            cand = agg[agg["zawodnik"].str.contains(q.strip(), case=False, na=False)]
        cand = cand.sort_values("zawodnik").head(300)
        if cand.empty:
            st.sidebar.info("Wpisz nazwisko, by znaleźć zawodnika.")
            return None, None
        labels = {f"{r.zawodnik} · {int(r.rocznik_final)} · {_txt(r.club_name)}": r.player_id
                  for r in cand.itertuples()}
        choice = st.sidebar.selectbox("Zawodnik", list(labels.keys()))
        req = agg[agg["player_id"] == labels[choice]].iloc[0]

    years = [y for y in sorted(agg["rocznik_final"].dropna().astype(int).unique())
             if y <= MAX_ROCZNIK]
    if not years:
        st.error(f"Brak roczników <= {MAX_ROCZNIK} w danych.")
        return None, None
    default_y = int(req["rocznik_final"]) if pd.notna(req["rocznik_final"]) else years[-1]
    if qp_rok and str(qp_rok).isdigit() and int(qp_rok) in years:
        declared_y = int(qp_rok)
    else:
        declared_y = st.sidebar.selectbox("Rocznik ze zgłoszenia", years,
                                          index=years.index(default_y) if default_y in years else len(years) - 1,
                                          help="Rocznik podany przez zawodnika w zgłoszeniu — ma priorytet.")
    return req, declared_y


# ─────────────────────────────────────────────────────────────────────────────
# GŁÓWNY WIDOK
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not check_password():
        return

    st.markdown(
        "<style>[data-testid='StyledFullScreenButton'],button[title='View fullscreen']"
        "{display:none!important;}</style>", unsafe_allow_html=True)

    st.sidebar.header("Ustawienia")
    min_min = st.sidebar.slider("Min. minut do oceny", 0, 1500, MIN_MIN_DEFAULT, 50,
                                help="Poniżej progu nie przypisujemy percentyla — za mała próba.")

    if not os.path.exists(AGG_PATH):
        st.warning("Brak danych. Wygeneruj lokalnie: `python precompute.py` i wrzuć folder `data/`.")
        return
    agg = load_agg()

    req, year = pick_requester(agg)
    if req is None:
        st.info("Wybierz zawodnika w panelu po lewej, aby wygenerować raport.")
        return

    pool = build_pool(agg, year, req)
    df = apply_percentiles(pool, min_min)
    n_all = len(df)
    n_elig = int(df["eligible"].sum())
    pid = str(req["player_id"])
    r = df[df["player_id"] == pid].iloc[0]
    trend = load_player_matches(pid, req.get("rocznik_final"))

    # „Gra ze starszymi” liczymy wobec ROCZNIKA ZE ZGŁOSZENIA (pewnego), a nie wobec
    # rocznika szacowanego z wieku — inaczej zawodnik grający własną kategorię dostaje
    # ten znacznik fałszywie, gdy w bazie ma zaniżony rocznik.
    if len(trend) and "league_name" in trend.columns:
        _maxy = _cat_maxyear_series(trend)
        _mn = pd.to_numeric(trend.get("minutes"), errors="coerce").fillna(0)
        _up = (_mn > 0) & _maxy.notna() & (int(year) > _maxy)
        r = r.copy()
        r["gra_ze_starszymi"] = bool(_up.any())

    st.markdown(f"### {r['zawodnik']}")
    st.markdown(
        f"<span style='color:#8B8B93'>Rocznik {int(year)} (ze zgłoszenia) · "
        f"{_txt(r.get('club_name'))} · {_txt(r.get('region_name'))}</span>",
        unsafe_allow_html=True)
    st.caption(f"Rocznik {int(year)}: **{n_all}** zawodników w bazie. Miejsce liczymy wśród "
               f"**{n_elig}** z min. {min_min} min (reszta ma za małą próbę), ale odnosimy je "
               f"do całego rocznika. Pozycja i liczebność przybliżone.")

    if not r["eligible"]:
        st.info(f"⚠️ Za mało minut na wiarygodną ocenę w skali kraju "
                f"({int(r['min_total'] or 0)} min, {int(r['mecze'] or 0)} mecz). "
                f"Percentyl przypisujemy od {min_min} min. Poniżej i tak pokazujemy formę i profil.")
    else:
        gfig, top = fig_gauge(r["pctl"])
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.markdown(f"#### {pozycja_txt(float(r['pctl']))}")
            st.metric("Orientacyjna pozycja w roczniku",
                      pozycja_krotko(r["rank_nat"], r["cohort_n"]),
                      help="Pozycja i liczebność są przybliżone — pula rówieśników jest szacowana "
                           "z wieku skorygowanego historią lig, nie ze spisu. "
                           "TOP-N zaokrąglamy w górę, żeby nie zawyżać.")
            st.plotly_chart(gfig, use_container_width=True, config=PLOTLY_CFG)
        with c2:
            st.markdown("#### Gdzie jesteś na tle rocznika")
            st.plotly_chart(fig_distribution(df.loc[df["eligible"], "pm_score"], r["pm_score"]),
                            use_container_width=True, config=PLOTLY_CFG)

    badges = []
    if bool(r.get("gra_ze_starszymi")):
        badges.append(("⚡ Gra ze starszymi", "#E8232A"))
    if (r.get("clj_minutes") or 0) > 0:
        badges.append((f"🏅 {int(r['clj_minutes'])}′ w CLJ", "#F0B429"))
    if (r.get("senior_minutes") or 0) > 0:
        badges.append((f"⚽ {int(r['senior_minutes'])}′ w seniorach", "#22A06B"))
    if badges:
        html = " ".join(
            f"<span style='border:1px solid {c};color:{c};border-radius:999px;"
            f"padding:3px 10px;margin-right:6px;font-size:0.85em;white-space:nowrap;'>{t}</span>"
            for t, c in badges)
        st.markdown(html, unsafe_allow_html=True)

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Profil na tle rocznika")
        if r["eligible"]:
            st.plotly_chart(fig_radar(r), use_container_width=True, config=PLOTLY_CFG)
            st.caption("Każda oś: percentyl w roczniku (100 = najlepszy w Polsce).")
        else:
            st.caption("Profil percentylowy dostępny po przekroczeniu progu minut.")
    with c4:
        st.markdown("#### Trend formy (sezon)")
        pm_rows = trend[trend["player_id"] == pid] if trend is not None and not trend.empty else pd.DataFrame()
        med = df.loc[df["eligible"], "pm_score"].median() if r["eligible"] else np.nan
        if len(pm_rows):
            st.plotly_chart(fig_trend(pm_rows, med), use_container_width=True, config=PLOTLY_CFG)
            if pd.notna(r.get("forma")):
                arrow = "↗ rośnie" if r["forma"] > 0.03 else ("↘ spada" if r["forma"] < -0.03 else "→ stabilna")
                st.caption(f"Ostatnie mecze vs średnia sezonu: **{arrow}**.")
        else:
            st.caption("Brak danych meczowych do wykresu formy dla tego zawodnika.")

    st.divider()

    st.markdown("#### Podsumowanie sezonu — mecze")
    if len(pm_rows) and "_sc" in pm_rows.columns:
        g = pm_rows.sort_values("match_date").copy()
        mn = pd.to_numeric(g["minutes"], errors="coerce").fillna(0)
        gl = pd.to_numeric(g["goals"], errors="coerce").fillna(0)
        yc = pd.to_numeric(g.get("yellow_cards"), errors="coerce").fillna(0)
        rc = pd.to_numeric(g.get("red_cards"), errors="coerce").fillna(0)
        sc = pd.to_numeric(g["_sc"], errors="coerce")
        minsum = max(1.0, mn.sum())
        res = g.get("match_result").astype(str) if "match_result" in g.columns else pd.Series([], dtype=str)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mecze", int(len(g)))
        m2.metric("Minuty", int(mn.sum()))
        m3.metric("Gole", int(gl.sum()))
        m4.metric("Śr. PM Score", f"{sc.mean() * 100:.0f}")
        # wiersz analityki (per 90 / średnie)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Gole / 90", f"{gl.sum() / minsum * 90:.2f}")
        a2.metric("Kartki / 90", f"{(yc.sum() + rc.sum()) / minsum * 90:.2f}")
        a3.metric("Minuty / mecz", f"{mn.mean():.0f}")
        if len(res):
            a4.metric("Zwycięstwa", f"{(res == 'wygrana').mean() * 100:.0f}%")

        rozgrywki = g.get("play_name")
        if rozgrywki is None or rozgrywki.isna().all():
            rozgrywki = g["league_name"]
        log = pd.DataFrame({
            "Data": pd.to_datetime(g["match_date"]).dt.strftime("%Y-%m-%d"),
            "Rozgrywki": rozgrywki,
            "Przeciwnik": g.get("opponent_name"),
            "Min": mn.astype(int),
            "Gole": gl.astype(int),
            "Żółte": yc.astype(int),
            "Czerw.": rc.astype(int),
            "Wynik": g.get("match_result"),
            "Strona": g.get("team_side"),
            "PM Score": (sc * 100).round(0).astype("Int64"),
        })
        st.dataframe(log, hide_index=True, use_container_width=True)

        # ── występy per rozgrywki (play) ──
        st.markdown("##### Występy w rozgrywkach")
        per = wystepy_per_play(g)
        if len(per):
            show = per.rename(columns={"_play": "Rozgrywki", "league_name": "Kategoria",
                                       "mecze": "Mecze", "minuty": "Minuty", "gole": "Gole",
                                       "kartki": "Kartki", "pm": "Śr. PM"})
            show["Śr. PM"] = (show["Śr. PM"] * 100).round(0).astype("Int64")
            st.dataframe(show, hide_index=True, use_container_width=True)
    else:
        st.caption("Brak danych meczowych dla tego zawodnika.")

    st.divider()

    st.markdown("#### Twój następny krok")
    _head, _steps = rekomendacja(r, min_min)
    st.success(f"**{_head}**")
    for _s in _steps:
        st.markdown(f"- {_s}")
    if int(r.get("szczebel") or 0) == 0:
        st.caption("Szczebel rozgrywek nierozpoznany z nazwy ligi — wskazówka o awansie pominięta.")

    st.divider()

    st.markdown("#### Kategoria wiekowa")
    native = PZPN_CAT.get(int(year), "—")
    cats = r.get("kategorie")
    if isinstance(cats, list):
        cats = ", ".join(cats)
    elif isinstance(cats, str):
        cats = cats.replace("; ", ", ")
    else:
        cats = "—"
    st.write(f"Kategoria macierzysta rocznika {int(year)} (PZPN 25/26): **{native}**.  \n"
             f"Rozgrywki, w których grał w tym sezonie: {cats}.")
    if bool(r.get("gra_ze_starszymi")):
        st.success(f"Gra w kategorii starszej o **{int(r['roczniki_w_gore'])}** rocznik(i) — "
                   f"historycznie silny sygnał talentu (choć bywa też skutkiem braków kadrowych).")

    st.divider()

    st.markdown(f"#### Top 10 rocznika {int(year)} w Polsce")
    cols = ["rank_nat", "zawodnik", "club_name", "region_name", "pm_score", "mecze", "min_total"]
    top10 = df[df["eligible"]].sort_values("pm_score", ascending=False).head(10)[cols].copy()
    if r["eligible"] and int(r.get("rank_nat") or 0) > 10:
        top10 = pd.concat([top10, df[df["player_id"] == pid][cols]])
    top10.columns = ["#", "Zawodnik", "Klub", "Województwo", "PM Score", "Mecze", "Minuty"]
    top10["#"] = pd.to_numeric(top10["#"], errors="coerce").astype("Int64")
    top10["PM Score"] = top10["PM Score"].round(3)

    def _hl(row):
        return ["background-color:#1c3a4a" if row["Zawodnik"] == r["zawodnik"] else "" for _ in row]
    st.dataframe(top10.style.apply(_hl, axis=1), hide_index=True, use_container_width=True)

    st.divider()

    # ── POZYCJA W SWOICH ROZGRYWKACH (kategoria + liga) ──────────────────────
    kat = pozycja_w_grupie(agg, r, "kategoria_glowna", min_min) if r["eligible"] else None
    play = pozycja_w_grupie(agg, r, "play_glowna", min_min) if r["eligible"] else None
    if kat or play:
        st.markdown("#### Gdzie jeste\u015b w swoich rozgrywkach")
        st.caption("Ranking rocznikowy por\u00f3wnuje z r\u00f3wie\u015bnikami z ca\u0142ej "
                   "Polski. Tutaj \u2014 z tymi, z kt\u00f3rymi realnie grasz "
                   f"(pr\u00f3g {min_min} min, wszystkie roczniki).")
        gc = st.columns(2)
        for c, g, tyt in ((gc[0], kat, "Twoje rozgrywki"), (gc[1], play, "Twoja liga")):
            with c:
                if not g:
                    st.caption(f"{tyt}: za ma\u0142o danych por\u00f3wnawczych.")
                    continue
                st.markdown(f"**{tyt}:** {_txt(g['nazwa'])}")
                st.metric("Pozycja", pozycja_w_grupie_opis(g["rank"], g["n"]),
                          help="W\u015br\u00f3d zawodnik\u00f3w, dla kt\u00f3rych to "
                               "g\u0142\u00f3wne rozgrywki (ka\u017cdy rocznik).")
                st.progress(min(max(float(g["pctl"]), 0.0), 1.0),
                            text=f"Wyprzedzasz {g['pctl'] * 100:.0f}%")
                t = pd.DataFrame(g["top"], columns=["#", "Zawodnik", "PM Score"])
                t["PM Score"] = t["PM Score"].round(3)
                st.dataframe(t.style.apply(
                    lambda row: ["background-color:#1c3a4a"
                                 if _norm_txt(row["Zawodnik"]) == _norm_txt(r["zawodnik"]) else ""
                                 for _ in row], axis=1),
                    hide_index=True, use_container_width=True)
        st.divider()

    st.markdown("#### Raport PDF dla zawodnika")
    top_pdf = (df[df["eligible"]].sort_values("pm_score", ascending=False).head(10)
               [["rank_nat", "zawodnik", "pm_score"]].values.tolist())
    pm_rows = trend[trend["player_id"] == pid] if trend is not None and not trend.empty else pd.DataFrame()
    dist_scores = df.loc[df["eligible"], "pm_score"].to_numpy()
    try:
        pdf_bytes = build_pdf(r, top_pdf, pm_rows, dist_scores, year, min_min, kat, play)
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(r["zawodnik"])).strip("_")
        st.download_button("⬇️ Pobierz PDF zawodnika", pdf_bytes,
                           file_name=f"raport_{safe}_{int(year)}.pdf", mime="application/pdf")
    except Exception as e:
        st.warning(f"Nie udało się zbudować PDF: {e}")

    with st.expander("Jak podnieść PM Score? (kierunki rozwoju)"):
        st.markdown(
            "PM Score rośnie, gdy zawodnik:\n\n"
            "- **gra w mocniejszych rozgrywkach** — poziom ligi jest mnożnikiem oceny "
            "(klasa okręgowa/wojewódzka > klasa A/B/C juniorska; CLJ najwyżej),\n"
            "- **gra w starszej kategorii** niż własny rocznik — premia za granie w górę,\n"
            "- **łapie minuty w seniorach** (dla juniorów dodatkowa premia),\n"
            "- **ma więcej rozegranych minut** — regularność gry realnie podnosi wynik,\n"
            "- **gra w wygrywającym zespole / na wyjeździe** — część oceny zależy od wyniku i strony "
            "boiska (to akurat mniej zależy od samego zawodnika).\n\n"
            "Innymi słowy profil idealny: **regularny gracz mocnej drużyny, występujący powyżej "
            "swojego rocznika, z minutami w seniorach.**"
        )

    with st.expander("Jak liczymy PM Score i ten ranking?"):
        st.markdown(
            "**PM Score** to ocena meczowa PlayMaker (~0–1) uwzględniająca **poziom rozgrywek** "
            "(ta sama gra w mocniejszej lidze jest warta więcej) oraz **grę powyżej swojego rocznika**. "
            "Sezonowy wynik to średnia meczów ważona minutami.\n\n"
            "**Ranking krajowy** porównuje zawodnika z całym jego rocznikiem w Polsce — bo wynik jest już "
            "skorygowany o poziom ligi, porównanie między różnymi ligami/województwami jest uczciwe.\n\n"
            f"Do rankingu wchodzą zawodnicy z min. **{min_min} minut** w sezonie."
        )


if __name__ == "__main__":
    main()