"""
raport_rocznikowy.py
────────────────────
Raport rocznikowy PlayMaker: gdzie zawodnik jest na tle CAŁEGO rocznika w Polsce.
Odbiorca: zawodnik / rodzic (rodzic płaci). Nacisk na jasny werdykt + kierunek rozwoju,
nie na tabelę metryk.

JEDNO ŹRÓDŁO PRAWDY DLA SCORE:
    importujemy compute_pm_score() i helpery wprost z app.py — ten sam wzór PM Score 2.0,
    league-aware (leagueMultiplier zależny od poziomu) + ageDiscount (premia za grę wyżej).
    Dzięki temu krajowy ranking rocznika jest porównywalny między ligami: dobra gra w słabej
    lidze daje niższy mnożnik niż ta sama gra w CLJ. To rozwiązuje cross-league comparability.

DANE WEJŚCIOWE (per mecz), z kohorta_rocznik.sql:
    player_id, firstname, lastname, match_id, match_date, play_id, play_name, region_name,
    league_id, league_name, team_id, team_name, club_id, club_name,
    minutes, goals, yellow_cards, red_cards, match_result, team_side,
    player_age, est_birth_year, age_at_match, is_junior_comp   [, position — opcjonalnie]

URUCHOMIENIE:
    pip install streamlit pandas numpy plotly psycopg2-binary
    streamlit run raport_rocznikowy.py
    # tryb CSV (domyślny): plik kohorta.csv (jeden lub wiele roczników — filtr po est_birth_year)
    # tryb DB: secret PM_DATA_MODE=db + PGHOST/PGDATABASE/PGUSER/PGPASSWORD/PGPORT + PM_SEASON_ID

DOSTĘP PER-LINK (dane nieletnich): ?player=<player_id> pokazuje od razu danego zawodnika.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── scoring i helpery z istniejącej apki (bez odpalania jej UI — jest pod main()) ──
from app import compute_pm_score, _coerce, _cat_maxyear_series, _secret

CURRENT_SEASON = _secret("PM_SEASON_ID", "e9d66181-d03e-4bb3-b889-4da848f4831d")
DATA_MODE = (_secret("PM_DATA_MODE", "csv") or "csv").lower()
MIN_MIN_DEFAULT = int(float(_secret("PM_MIN_MINUTES", "300") or "300"))

# rocznik → nazwa kategorii PZPN (sezon 25/26). Kategoria macierzysta = wpis dla własnego rocznika.
PZPN_CAT = {2006: "A1 / U-19", 2007: "A2 / U-18", 2008: "B1 / U-17", 2009: "B2 / U-16",
            2010: "C1 / U-15", 2011: "C2 / U-14", 2012: "D1 / U-13", 2013: "D2 / U-12",
            2014: "E1 / U-11", 2015: "E2 / U-10", 2016: "F1 / U-9", 2017: "F2 / U-8"}

DIMS = ["Jakość gry", "Skuteczność", "Regularność gry", "Równość formy", "Dyscyplina"]

st.set_page_config(page_title="Raport rocznikowy · PlayMaker", layout="wide")


# ─────────────────────────────────────────────────────────────────────────────
# ŁADOWANIE DANYCH
# ─────────────────────────────────────────────────────────────────────────────
def _read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1")


@st.cache_data(show_spinner=False)
def load_cohort_csv(path="kohorta.csv"):
    return _coerce(_read_csv(path))


@st.cache_data(show_spinner=False)
def load_cohort_db(season_id, birth_year):
    import psycopg2
    import re
    sql = open("kohorta_rocznik.sql", encoding="utf-8").read()
    # Wstrzyknij sezon + rocznik do bloku params (SQL jest uruchamialny też wprost,
    # z wartościami domyślnymi — apka je tu nadpisuje wyborem z panelu).
    sql = re.sub(r"'[^']*'::text\s+AS season_id",
                 f"'{season_id}'::text AS season_id", sql, count=1)
    sql = re.sub(r"\b\d{4}::int\s+AS birth_year",
                 f"{int(birth_year)}::int AS birth_year", sql, count=1)
    conn = psycopg2.connect(host=_secret("PGHOST"), dbname=_secret("PGDATABASE"),
                            user=_secret("PGUSER"), password=_secret("PGPASSWORD"),
                            port=_secret("PGPORT", "5432") or "5432")
    df = pd.read_sql(sql, conn)
    conn.close()
    return _coerce(df)


# ─────────────────────────────────────────────────────────────────────────────
# BUDOWA KOHORTY (metryki bazowe per zawodnik) — cache; percentyle liczone osobno
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_cohort(m):
    m = m.copy()
    m["zawodnik"] = (m["firstname"].fillna("") + " " + m["lastname"].fillna("")).str.strip()
    comp = compute_pm_score(m)
    m["_sc"] = comp["score"].values          # pełny PM Score (age_part + stats_part), 0..1
    m["_sp"] = comp["stats_part"].values     # league + performance (bez wieku)
    mn = pd.to_numeric(m["minutes"], errors="coerce").fillna(0)
    m["_mn"] = mn
    m["_maxy"] = _cat_maxyear_series(m)       # max rocznik dywizji (PZPN) per mecz

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

    # klub / region / liga wiodąca = najwięcej minut
    lead = (m.sort_values("_mn", ascending=False)
            .groupby("player_id")[["club_name", "region_name", "league_name"]].first())
    out = out.join(lead)

    # forma (ostatnie 5 vs średnia) i konsekwencja (stabilność) ze stats_part
    def _form(g):
        x = g.sort_values("match_date")["_sp"].dropna()
        mm = x.mean()
        return pd.Series({"forma": ((x.tail(5).mean() - mm) / mm) if len(x) >= 3 and mm else np.nan,
                          "kons": (1 / (1 + x.std(ddof=0))) if len(x) >= 2 else np.nan})
    out = out.join(gp.apply(_form))

    # gra ze starszymi: własny rocznik > max rocznik dywizji (grał, minuty>0)
    py = m["est_birth_year"]
    jun_older = (mn > 0) & m["_maxy"].notna() & py.notna() & (py > m["_maxy"])
    out["roczniki_w_gore"] = (py - m["_maxy"]).where(jun_older).groupby(m["player_id"]).max()
    out["gra_ze_starszymi"] = jun_older.groupby(m["player_id"]).any().reindex(out.index).fillna(False)
    # minuty w CLJ / w seniorach — sygnały poziomu (do znaczników)
    is_clj = m["league_name"].astype(str).str.contains(r"\bCLJ\b|Centralna Liga Junior", case=False, regex=True, na=False)
    out["clj_minutes"] = (mn * is_clj).groupby(m["player_id"]).sum()
    is_senior = (~m["is_junior_comp"].fillna(False)) & (m["age_at_match"].between(12, 19))
    out["senior_minutes"] = (mn * is_senior).groupby(m["player_id"]).sum()

    out["kategorie"] = gp["league_name"].agg(lambda s: sorted(set(s.dropna().astype(str))))
    return out.reset_index(), m


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
    fig.add_trace(go.Histogram(x=sub_scores, nbinsx=40, marker_color="#2b3b4d",
                               name="rocznik"))
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
                                 radialaxis=dict(range=[0, 100], showticklabels=True, tickvals=[25, 50, 75, 100])))
    return fig


def fig_trend(pm_rows, cohort_median):
    g = pm_rows.sort_values("match_date")
    roll = g["_sc"].rolling(3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["match_date"], y=g["_sc"], mode="markers",
                             marker=dict(size=6, color="#3b4b5d"), name="mecz"))
    fig.add_trace(go.Scatter(x=g["match_date"], y=roll, mode="lines",
                             line=dict(color="#5db0ff", width=3), name="forma (3 mecze)"))
    fig.add_hline(y=cohort_median, line_dash="dash", line_color="#f5c451",
                  annotation_text="mediana rocznika", annotation_font_color="#f5c451")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#cdd6e0", legend=dict(orientation="h", y=1.15),
                      xaxis_title="", yaxis_title="PM Score / mecz")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DOSTĘP (opcjonalne hasło — spójne z Almanachem: secret APP_PASSWORD)
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
# GŁÓWNY WIDOK
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not check_password():
        return

    st.sidebar.header("Ustawienia")
    min_min = st.sidebar.slider("Min. minut do oceny", 0, 1500, MIN_MIN_DEFAULT, 50,
                                help="Poniżej progu nie przypisujemy percentyla — za mała próba.")

    # wybór rocznika + wczytanie kohorty
    if DATA_MODE == "db":
        year = st.sidebar.number_input("Rocznik", 2004, 2018, 2010, 1)
        raw = load_cohort_db(CURRENT_SEASON, int(year))
    else:
        raw = load_cohort_csv(_secret("PM_COHORT_CSV", "kohorta.csv"))
        years = sorted(pd.to_numeric(raw["est_birth_year"], errors="coerce").dropna().astype(int).unique())
        year = st.sidebar.selectbox("Rocznik", years, index=max(0, len(years) - 1))
        raw = raw[pd.to_numeric(raw["est_birth_year"], errors="coerce") == year].copy()

    if raw.empty:
        st.warning("Brak danych dla wybranego rocznika.")
        return

    base, scored = build_cohort(raw)
    df = apply_percentiles(base, min_min)

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

    # ── NAGŁÓWEK ──
    st.markdown(f"### {r['zawodnik']}")
    st.caption(f"Rocznik {int(year)} · {r.get('club_name') or '—'} · "
               f"{r.get('region_name') or '—'} · liga wiodąca: {r.get('league_name') or '—'}")

    if not r["eligible"]:
        st.info(f"⚠️ Za mało minut na wiarygodną ocenę w skali kraju "
                f"({int(r['min_total'] or 0)} min, {int(r['mecze'] or 0)} mecz). "
                f"Percentyl przypisujemy od {min_min} min. Poniżej i tak pokazujemy formę i profil.")
    else:
        gfig, top = fig_gauge(r["pctl"])
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.markdown(f"#### Jesteś w **TOP {top:.0f}%** rocznika w Polsce")
            st.metric("Miejsce w kraju (rocznik)",
                      f"{int(r['rank_nat'])} / {int(r['cohort_n'])}")
            st.plotly_chart(gfig, use_container_width=True)
        with c2:
            st.markdown("#### Gdzie jesteś na tle rocznika")
            sub = df[df["eligible"]]
            st.plotly_chart(fig_distribution(sub["pm_score"], r["pm_score"]),
                            use_container_width=True)

    # znaczniki kontekstu
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

    # ── PROFIL (radar) + TREND ──
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
        pm_rows = scored[scored["player_id"] == pid]
        med = df.loc[df["eligible"], "pm_score"].median()
        st.plotly_chart(fig_trend(pm_rows, med), use_container_width=True)
        if pd.notna(r.get("forma")):
            arrow = "↗ rośnie" if r["forma"] > 0.03 else ("↘ spada" if r["forma"] < -0.03 else "→ stabilna")
            st.caption(f"Ostatnie mecze vs średnia sezonu: **{arrow}**.")

    st.divider()

    # ── KONTEKST KATEGORII (A1/A2…) ──
    st.markdown("#### Kategoria wiekowa")
    native = PZPN_CAT.get(int(year), "—")
    played_cats = ", ".join(r.get("kategorie") or []) or "—"
    st.write(f"Kategoria macierzysta rocznika {int(year)} (PZPN 25/26): **{native}**.  \n"
             f"Rozgrywki, w których grał w tym sezonie: {played_cats}.")
    if bool(r.get("gra_ze_starszymi")):
        st.success(f"Gra w kategorii starszej o **{int(r['roczniki_w_gore'])}** rocznik(i) — "
                   f"historycznie silny sygnał talentu (choć bywa też skutkiem braków kadrowych).")

    st.divider()

    # ── TOP 10 ROCZNIKA ──
    st.markdown(f"#### Top 10 rocznika {int(year)} w Polsce")
    top10 = (df[df["eligible"]].sort_values("pm_score", ascending=False).head(10)
             [["rank_nat", "zawodnik", "club_name", "region_name", "pm_score", "mecze", "min_total"]]
             .copy())
    if int(r.get("rank_nat") or 0) > 10 and r["eligible"]:
        top10 = pd.concat([top10, df[df["player_id"] == pid]
                          [["rank_nat", "zawodnik", "club_name", "region_name", "pm_score", "mecze", "min_total"]]])
    top10.columns = ["#", "Zawodnik", "Klub", "Województwo", "PM Score", "Mecze", "Minuty"]
    top10["#"] = top10["#"].astype(int)
    top10["PM Score"] = top10["PM Score"].round(3)

    def _hl(row):
        return ["background-color:#1c3a4a" if row["Zawodnik"] == r["zawodnik"] else "" for _ in row]
    st.dataframe(top10.style.apply(_hl, axis=1), hide_index=True, use_container_width=True)

    with st.expander("Jak liczymy PM Score i ten ranking?"):
        st.markdown(
            "**PM Score** to ocena meczowa PlayMaker w skali ~0–1, uwzględniająca **poziom rozgrywek** "
            "(ta sama gra w mocniejszej lidze jest warta więcej) oraz **grę powyżej swojego rocznika**. "
            "Sezonowy wynik to średnia meczów ważona minutami.\n\n"
            "**Ranking krajowy** porównuje zawodnika z całym jego rocznikiem w Polsce — bo wynik jest już "
            "skorygowany o poziom ligi, porównanie między różnymi ligami/województwami jest uczciwe.\n\n"
            f"Do rankingu wchodzą zawodnicy z min. **{min_min} minut** w sezonie (mniejsza próba = brak "
            "wiarygodnego percentyla)."
        )


if __name__ == "__main__":
    main()