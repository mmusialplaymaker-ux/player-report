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
             "clj_minutes", "senior_minutes"]

st.set_page_config(page_title="Raport rocznikowy · PlayMaker", layout="wide")


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
    df["_sc"] = pd.to_numeric(df["_sc"], errors="coerce")
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
    return df


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
            st.markdown(f"#### Jesteś w **TOP {top:.0f}%** rocznika w Polsce")
            st.metric("Miejsce w kraju (rocznik)", f"{int(r['rank_nat'])} / {int(r['cohort_n'])}")
            st.plotly_chart(gfig, use_container_width=True)
        with c2:
            st.markdown("#### Gdzie jesteś na tle rocznika")
            st.plotly_chart(fig_distribution(df.loc[df["eligible"], "pm_score"], r["pm_score"]),
                            use_container_width=True)

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
            st.plotly_chart(fig_radar(r), use_container_width=True)
            st.caption("Każda oś: percentyl w roczniku (100 = najlepszy w Polsce).")
        else:
            st.caption("Profil percentylowy dostępny po przekroczeniu progu minut.")
    with c4:
        st.markdown("#### Trend formy (sezon)")
        pm_rows = trend[trend["player_id"] == pid] if trend is not None and not trend.empty else pd.DataFrame()
        med = df.loc[df["eligible"], "pm_score"].median() if r["eligible"] else np.nan
        if len(pm_rows):
            st.plotly_chart(fig_trend(pm_rows, med), use_container_width=True)
            if pd.notna(r.get("forma")):
                arrow = "↗ rośnie" if r["forma"] > 0.03 else ("↘ spada" if r["forma"] < -0.03 else "→ stabilna")
                st.caption(f"Ostatnie mecze vs średnia sezonu: **{arrow}**.")
        else:
            st.caption("Brak danych meczowych do wykresu formy dla tego zawodnika.")

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