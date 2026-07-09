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
import datetime as _dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# scoring i helpery z istniejącej apki (UI app.py jest pod main() → import bezpieczny)
from app import compute_pm_score, _coerce, _cat_maxyear_series, _secret

CURRENT_SEASON = _secret("PM_SEASON_ID", "e9d66181-d03e-4bb3-b889-4da848f4831d")
DATA_MODE = (_secret("PM_DATA_MODE", "csv") or "csv").lower()
MIN_MIN_DEFAULT = int(float(_secret("PM_MIN_MINUTES", "300") or "300"))

AGG_PATH = _secret("PM_AGG_CSV", "kohorta_agg.csv.gz")
TREND_PATH = _secret("PM_TREND_CSV", "kohorta_trend.csv.gz")
COHORT_CSV = _secret("PM_COHORT_CSV", "kohorta.csv")

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


@st.cache_data(show_spinner=False)
def load_agg(path):
    df = _read_csv(path)
    for c in _AGG_NUMS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "gra_ze_starszymi" in df.columns:
        df["gra_ze_starszymi"] = (df["gra_ze_starszymi"].astype(str).str.strip().str.lower()
                                  .isin(["true", "1", "t", "yes"]))
    return df


@st.cache_data(show_spinner=False)
def load_trend(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["player_id", "match_date", "minutes", "_sc"])
    df = _read_csv(path)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    for c in ("minutes", "goals", "yellow_cards", "red_cards", "_sc"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_cohort_csv(path):
    return _coerce(_read_csv(path))


@st.cache_data(show_spinner=False)
def load_cohort_db(season_id, birth_year):
    import psycopg2
    import re
    sql = open("kohorta_rocznik.sql", encoding="utf-8").read()
    sql = re.sub(r"'[^']*'::text\s+AS season_id", f"'{season_id}'::text AS season_id", sql, count=1)
    sql = re.sub(r"\b\d{4}::int\s+AS birth_year", f"{int(birth_year)}::int AS birth_year", sql, count=1)
    conn = psycopg2.connect(host=_secret("PGHOST"), dbname=_secret("PGDATABASE"),
                            user=_secret("PGUSER"), password=_secret("PGPASSWORD"),
                            port=_secret("PGPORT", "5432") or "5432")
    df = pd.read_sql(sql, conn)
    conn.close()
    return _coerce(df)


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
    """Percentyle i rank liczone TYLKO wśród zawodników z wiarygodną próbą (min_min)."""
    df = base.copy()
    elig = df["min_total"].fillna(0) >= min_min
    df["eligible"] = elig
    sub = df[elig]
    df.loc[elig, "pctl"] = sub["pm_score"].rank(pct=True)
    df.loc[elig, "rank_nat"] = sub["pm_score"].rank(ascending=False, method="min")
    df["cohort_n"] = int(elig.sum())
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
            f"({r.get('szczebel_nazwa') or '—'}).",
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

    # 4) mocny, ale jeszcze nie dominuje szczebla
    if pd.notna(lvl) and lvl >= 0.60:
        return ("Jesteś blisko — dołóż minut i stabilności", [
            f"W swoim szczeblu ({r.get('szczebel_nazwa') or '—'}) wyprzedzasz "
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
            if sz == 0 else f"Rozważ testy o szczebel wyżej niż {r.get('szczebel_nazwa') or '—'}.",
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
    """Uczciwy opis pozycji: bez mylącego 'TOP 88%'."""
    przed = pctl * 100
    if przed >= 50:
        return f"Wyprzedzasz {przed:.0f}% rocznika  ·  TOP {100 - przed:.0f}%"
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
def build_pdf(r, top_pdf, pm_rows, dist_scores, year, min_min):
    """Jednostronicowy PDF: PM Score, ranking, rozkład rocznika, trend, Top 10, wskazówki."""
    RED, GREEN, INK, GREY, BG = "#e2231a", "#22a06b", "#1b1f24", "#8a94a3", "#eef1f5"
    pm = float(r.get("pm_score") or 0) * 100
    elig = bool(r.get("eligible"))
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150)  # A4
    fig.patch.set_facecolor("white")

    # ── nagłówek ──
    fig.text(0.07, 0.960, "RAPORT PLAYMAKER", fontsize=13, weight="bold", color=INK)
    fig.text(0.93, 0.960, "playmaker.pro", fontsize=10, color=RED, ha="right", weight="bold")
    fig.text(0.07, 0.925, str(r.get("zawodnik") or "—"), fontsize=22, weight="bold", color=INK)
    meta = f"Rocznik {int(year)}   ·   {r.get('club_name') or '—'}   ·   {r.get('region_name') or '—'}"
    fig.text(0.07, 0.900, meta, fontsize=11, color=GREY)
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.885, 0.885], color=BG, lw=2))

    # ── donut PM Score ──
    axd = fig.add_axes([0.07, 0.70, 0.26, 0.15])
    axd.pie([pm, max(0.0, 100 - pm)], colors=[RED, BG], startangle=90,
            counterclock=False, wedgeprops=dict(width=0.34))
    axd.text(0, 0.08, f"{pm:.0f}", ha="center", va="center", fontsize=28, weight="bold", color=INK)
    axd.text(0, -0.30, "PM Score", ha="center", va="center", fontsize=11, color=GREY)
    axd.set(aspect="equal")

    # ── rankingi (liczby) — jednakowy rozmiar, pogrubiony tylko licznik, bez nakładania ──
    if elig:
        fig.text(0.42, 0.820, f"Ranking rocznika {int(year)}", fontsize=11, color=GREY)
        t_rank = fig.text(0.42, 0.775, f"{int(r['rank_nat'])}.", fontsize=17, color=INK, weight="bold")
        fig.canvas.draw()  # potrzebne, by zmierzyć szerokość tekstu
        bb = t_rank.get_window_extent(renderer=fig.canvas.get_renderer())
        x_after = fig.transFigure.inverted().transform((bb.x1, 0))[0]
        fig.text(x_after + 0.010, 0.775, f"/  {int(r['cohort_n'])} w Polsce", fontsize=17, color=INK)
        fig.text(0.42, 0.725, pozycja_txt(float(r["pctl"])), fontsize=13, weight="bold", color=RED)
    else:
        fig.text(0.42, 0.80, "Za mało minut na ranking krajowy", fontsize=12, color=GREY)
        fig.text(0.42, 0.760, f"{int(r.get('min_total') or 0)} min", fontsize=20, weight="bold", color=INK)
        fig.text(0.42, 0.725, f"(próg {min_min} min)", fontsize=11, color=GREY)

    # ── znaczniki ──
    badges = []
    if bool(r.get("gra_ze_starszymi")):
        n = r.get("roczniki_w_gore")
        badges.append(f"gra ze starszymi (+{int(n)})" if pd.notna(n) and n >= 1 else "gra ze starszymi")
    if (r.get("senior_minutes") or 0) > 0:
        badges.append(f"{int(r['senior_minutes'])}' w seniorach")
    if (r.get("clj_minutes") or 0) > 0:
        badges.append(f"{int(r['clj_minutes'])}' w CLJ")
    if badges:
        fig.text(0.07, 0.672, "  ·  ".join(badges), fontsize=10.5, color=INK,
                 bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec="none"))

    # ── rozkład rocznika: gdzie jesteś ──
    fig.text(0.07, 0.640, "Gdzie jesteś na tle rocznika", fontsize=12, weight="bold", color=INK)
    axh = fig.add_axes([0.07, 0.505, 0.86, 0.115])
    if dist_scores is not None and len(dist_scores):
        axh.hist(np.asarray(dist_scores, dtype=float) * 100, bins=40, color="#c9d2dc")
        if elig:
            axh.axvline(pm, color=RED, lw=2.4)
            axh.text(pm, axh.get_ylim()[1] * 0.92, " tu jesteś", color=RED, fontsize=9, weight="bold")
    axh.set_xlabel("PM Score", fontsize=8, color=GREY)
    for s in ("top", "right", "left"):
        axh.spines[s].set_visible(False)
    axh.set_yticks([])
    axh.tick_params(labelsize=8, colors=GREY)

    # ── trend formy ──
    fig.text(0.07, 0.472, "Trend PM Score", fontsize=12, weight="bold", color=INK)
    axt = fig.add_axes([0.07, 0.345, 0.86, 0.105])
    slope_txt = "—"
    if pm_rows is not None and len(pm_rows):
        g = pm_rows.sort_values("match_date")
        y = pd.to_numeric(g["_sc"], errors="coerce").to_numpy()
        roll = pd.Series(y).rolling(3, min_periods=1).mean().to_numpy()
        x = np.arange(len(y))
        axt.plot(x, y, marker="o", ms=3, lw=0, color="#c9d2dc")
        axt.plot(x, roll, lw=2.4, color=GREEN)
        if len(y) >= 3 and np.isfinite(y).sum() >= 3:
            b = np.polyfit(x[np.isfinite(y)], y[np.isfinite(y)], 1)[0] * 100
            arrow = "rośnie" if b > 0.05 else ("spada" if b < -0.05 else "stabilna")
            slope_txt = f"{arrow} {b:+.1f} / kolejkę"
    fig.text(0.93, 0.472, slope_txt, ha="right", fontsize=12, weight="bold", color=GREEN)
    axt.set_xticks([])
    for s in ("top", "right", "left"):
        axt.spines[s].set_visible(False)
    axt.tick_params(labelsize=8, colors=GREY)

    # ── Top 10 rocznika ──
    fig.text(0.07, 0.310, f"Top 10 rocznika {int(year)} w Polsce", fontsize=12, weight="bold", color=INK)
    fig.text(0.60, 0.310, "PM Score", fontsize=8.5, color=GREY)
    y0 = 0.284
    for i, (rank, name, sc) in enumerate(top_pdf[:10]):
        mine = str(name) == str(r.get("zawodnik"))
        w = "bold" if mine else "normal"
        col = RED if mine else INK
        fig.text(0.09, y0 - i * 0.0195, f"{i + 1:>2}.", fontsize=10, color=GREY)
        fig.text(0.15, y0 - i * 0.0195, str(name), fontsize=10, color=col, weight=w)
        fig.text(0.61, y0 - i * 0.0195, f"{float(sc) * 100:.0f}", fontsize=10, color=col, weight=w)

    # ── twój następny krok (dynamiczne wskazówki) ──
    head, steps = rekomendacja(r, min_min)
    fig.text(0.07, 0.105, "TWÓJ NASTĘPNY KROK", fontsize=8.5, weight="bold", color=GREY)
    fig.text(0.07, 0.086, head, fontsize=11, weight="bold", color=INK)
    yy = 0.068
    for stp in steps[:3]:
        fig.text(0.07, yy, "• " + stp[:118], fontsize=7.6, color=GREY)
        yy -= 0.0145

    # ── stopka ──
    foot = (f"PM Score uwzględnia poziom rozgrywek i grę powyżej rocznika. "
            f"Ranking krajowy wśród zawodników z min. {min_min} min. "
            f"Wygenerowano {_dt.date.today():%Y-%m-%d}.")
    fig.text(0.07, 0.028, foot, fontsize=7.5, color=GREY)

    # ── strona 2: podsumowanie sezonu — mecze ──
    fig2 = _pdf_match_log(r, pm_rows, year)

    buf = io.BytesIO()
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(buf) as pp:
        pp.savefig(fig)
        if fig2 is not None:
            pp.savefig(fig2)
    plt.close(fig)
    if fig2 is not None:
        plt.close(fig2)
    buf.seek(0)
    return buf.getvalue()


def _pdf_match_log(r, pm_rows, year):
    """Strona 2 PDF: log meczów sezonu (data, rozgrywki, przeciwnik, minuty, gole, kartki, wynik, PM)."""
    if pm_rows is None or not len(pm_rows) or "_sc" not in pm_rows.columns:
        return None
    RED, INK, GREY, BG = "#e2231a", "#1b1f24", "#8a94a3", "#eef1f5"
    RESMAP = {"wygrana": "W", "remis": "R", "porażka": "P"}
    g = pm_rows.sort_values("match_date").copy()
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.960, "PODSUMOWANIE SEZONU — MECZE", fontsize=13, weight="bold", color=INK)
    fig.text(0.07, 0.933, f"{r.get('zawodnik') or '—'}   ·   rocznik {int(year)}", fontsize=12, color=GREY)

    mn = pd.to_numeric(g["minutes"], errors="coerce").fillna(0)
    gl = pd.to_numeric(g["goals"], errors="coerce").fillna(0)
    yc = pd.to_numeric(g.get("yellow_cards"), errors="coerce").fillna(0)
    rc = pd.to_numeric(g.get("red_cards"), errors="coerce").fillna(0)
    sc = pd.to_numeric(g["_sc"], errors="coerce")
    minsum = max(1.0, mn.sum())
    res = g.get("match_result").astype(str) if "match_result" in g.columns else pd.Series([], dtype=str)
    tot = (f"Mecze: {len(g)}   ·   Minuty: {int(mn.sum())}   ·   Gole: {int(gl.sum())}   ·   "
           f"Śr. PM Score: {sc.mean() * 100:.0f}")
    fig.text(0.07, 0.908, tot, fontsize=10.5, weight="bold", color=INK,
             bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec="none"))
    ana = (f"Gole/90: {gl.sum() / minsum * 90:.2f}    Kartki/90: {(yc.sum() + rc.sum()) / minsum * 90:.2f}"
           f"    Min/mecz: {mn.mean():.0f}")
    if len(res):
        ana += f"    Zwycięstwa: {(res == 'wygrana').mean() * 100:.0f}%"
    fig.text(0.07, 0.882, ana, fontsize=9.5, color=GREY)

    # rozgrywki = play_name, fallback league_name
    rozg = g.get("play_name")
    if rozg is None or rozg.isna().all():
        rozg = g.get("league_name")

    # nagłówki kolumn (x, ha)
    C = {"Data": (0.07, "left"), "Rozgrywki": (0.155, "left"), "Przeciwnik": (0.375, "left"),
         "Min": (0.685, "right"), "G": (0.735, "right"), "K": (0.785, "right"),
         "W": (0.825, "left"), "PM": (0.950, "right")}
    y = 0.848
    for name, (x, ha) in C.items():
        fig.text(x, y, name, fontsize=8.5, weight="bold", color=GREY, ha=ha)
    fig.add_artist(plt.Line2D([0.07, 0.955], [y - 0.006, y - 0.006], color=BG, lw=1))

    step, y = 0.0196, y - 0.024
    maxrows = int((y - 0.045) / step)
    idx = g.index[:maxrows]
    for k, i in enumerate(idx):
        yy = y - k * step
        date = pd.to_datetime(g.at[i, "match_date"]).strftime("%Y-%m-%d") if pd.notna(g.at[i, "match_date"]) else "—"
        rz = str(rozg.loc[i] if rozg is not None else "")[:22]
        opp = str(g.at[i, "opponent_name"])[:26] if "opponent_name" in g.columns and pd.notna(g.at[i, "opponent_name"]) else "—"
        fig.text(0.07, yy, date, fontsize=7.6, color=INK)
        fig.text(0.155, yy, rz, fontsize=7.6, color=INK)
        fig.text(0.375, yy, opp, fontsize=7.6, color=INK)
        fig.text(0.685, yy, f"{int(mn.loc[i])}", fontsize=7.6, color=INK, ha="right")
        fig.text(0.735, yy, f"{int(gl.loc[i])}", fontsize=7.6, color=INK, ha="right")
        fig.text(0.785, yy, f"{int(yc.loc[i] + rc.loc[i])}", fontsize=7.6, color=INK, ha="right")
        fig.text(0.825, yy, RESMAP.get(str(g.at[i, "match_result"]) if "match_result" in g.columns else "", ""),
                 fontsize=7.6, color=GREY)
        fig.text(0.950, yy, f"{float(sc.loc[i]) * 100:.0f}", fontsize=7.6, color=INK, ha="right", weight="bold")
    if len(g) > maxrows:
        fig.text(0.07, y - maxrows * step, f"… oraz {len(g) - maxrows} kolejnych meczów",
                 fontsize=8, color=GREY, style="italic")
    fig.text(0.07, 0.028, "W = wygrana · R = remis · P = porażka · K = kartki (żółte+czerwone)",
             fontsize=7.5, color=GREY)
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
# DANE: precomputed jeśli jest, inaczej fallback match-level
# ─────────────────────────────────────────────────────────────────────────────
def get_year_and_data(year_widget_key="rok"):
    precomputed = os.path.exists(AGG_PATH)
    if precomputed:
        agg = load_agg(AGG_PATH)
        years = sorted(pd.to_numeric(agg["est_birth_year"], errors="coerce").dropna().astype(int).unique())
        if not years:
            return None, None, None
        year = st.sidebar.selectbox("Rocznik", years, index=len(years) - 1, key=year_widget_key)
        base = agg[pd.to_numeric(agg["est_birth_year"], errors="coerce") == year].copy()
        trend_all = load_trend(TREND_PATH)
        trend = (trend_all[trend_all["player_id"].isin(set(base["player_id"]))]
                 if not trend_all.empty else trend_all)
        return year, base, trend

    # fallback (lokalnie): match-level z DB albo dużego CSV
    if DATA_MODE == "db":
        year = st.sidebar.number_input("Rocznik", 2004, 2018, 2010, 1, key=year_widget_key)
        raw = load_cohort_db(CURRENT_SEASON, int(year))
    else:
        raw = load_cohort_csv(COHORT_CSV)
        years = sorted(pd.to_numeric(raw["est_birth_year"], errors="coerce").dropna().astype(int).unique())
        year = st.sidebar.selectbox("Rocznik", years, index=len(years) - 1, key=year_widget_key)
        raw = raw[pd.to_numeric(raw["est_birth_year"], errors="coerce") == year].copy()
    if raw.empty:
        return year, raw, None
    base, trend = build_cohort(raw)
    return year, base, trend


# ─────────────────────────────────────────────────────────────────────────────
# GŁÓWNY WIDOK
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not check_password():
        return

    # ukryj przycisk pełnego ekranu na wykresach
    st.markdown(
        "<style>[data-testid='StyledFullScreenButton'],button[title='View fullscreen']"
        "{display:none!important;}</style>", unsafe_allow_html=True)

    st.sidebar.header("Ustawienia")
    min_min = st.sidebar.slider("Min. minut do oceny", 0, 1500, MIN_MIN_DEFAULT, 50,
                                help="Poniżej progu nie przypisujemy percentyla — za mała próba.")

    year, base, trend = get_year_and_data()
    if base is None or base.empty:
        st.warning("Brak danych. W trybie chmurowym potrzebny jest kohorta_agg.csv.gz "
                   "(wygeneruj lokalnie: python precompute.py).")
        return

    df = apply_percentiles(base, min_min)
    n_all = len(df)
    n_elig = int(df["eligible"].sum())

    # wybór zawodnika (per-link ?player=<id> albo z listy)
    qp_player = st.query_params.get("player")
    names = df.sort_values("zawodnik")[["player_id", "zawodnik", "club_name"]]
    labels = {f"{r.zawodnik} ({r.club_name})": r.player_id for r in names.itertuples()}
    if qp_player and qp_player in set(df["player_id"]):
        pid = qp_player
    else:
        choice = st.sidebar.selectbox(f"Zawodnik (rocznik {year})", list(labels.keys()))
        pid = labels[choice]

    r = df[df["player_id"] == pid].iloc[0]

    st.markdown(f"### {r['zawodnik']}")
    st.caption(f"Rocznik {int(year)} · {r.get('club_name') or '—'} · {r.get('region_name') or '—'} · "
               f"liga wiodąca: {r.get('league_name') or '—'}")
    st.caption(f"W rankingu rocznika: **{n_all}** zawodników w bazie, **{n_elig}** z wiarygodną próbą "
               f"(≥ {min_min} min).")

    if not r["eligible"]:
        st.info(f"⚠️ Za mało minut na wiarygodną ocenę w skali kraju "
                f"({int(r['min_total'] or 0)} min, {int(r['mecze'] or 0)} mecz). "
                f"Percentyl przypisujemy od {min_min} min. Poniżej i tak pokazujemy formę i profil.")
    else:
        gfig, top = fig_gauge(r["pctl"])
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.markdown(f"#### {pozycja_txt(float(r['pctl']))}")
            st.metric("Miejsce w kraju (rocznik)", f"{int(r['rank_nat'])} / {int(r['cohort_n'])}")
            st.plotly_chart(gfig, use_container_width=True, config=PLOTLY_CFG)
        with c2:
            st.markdown("#### Gdzie jesteś na tle rocznika")
            st.plotly_chart(fig_distribution(df.loc[df["eligible"], "pm_score"], r["pm_score"]),
                            use_container_width=True, config=PLOTLY_CFG)

    badges = []
    if bool(r.get("gra_ze_starszymi")):
        n = r.get("roczniki_w_gore")
        badges.append(f"↑ gra ze starszymi (+{int(n)})" if pd.notna(n) and n >= 1 else "↑ gra ze starszymi")
    if (r.get("senior_minutes") or 0) > 0:
        badges.append(f"⚽ {int(r['senior_minutes'])}′ w seniorach")
    if (r.get("clj_minutes") or 0) > 0:
        badges.append(f"🏅 {int(r['clj_minutes'])}′ w CLJ")
    if badges:
        st.markdown(" ".join(f"`{b}`" for b in badges))

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
    st.markdown("#### Raport PDF dla zawodnika")
    top_pdf = (df[df["eligible"]].sort_values("pm_score", ascending=False).head(10)
               [["rank_nat", "zawodnik", "pm_score"]].values.tolist())
    pm_rows = trend[trend["player_id"] == pid] if trend is not None and not trend.empty else pd.DataFrame()
    dist_scores = df.loc[df["eligible"], "pm_score"].to_numpy()
    try:
        pdf_bytes = build_pdf(r, top_pdf, pm_rows, dist_scores, year, min_min)
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