"""
PlayMaker Pro - dashboard skautingowy (prototyp).
Dwa widoki: Odkrywanie (leaderboard + scatter) i Porownanie (radar <=5 graczy).

Uruchomienie:
    pip install -r requirements.txt
    streamlit run app.py

Tryb danych:
    DATA_MODE = "csv"  -> czyta stats_test.csv / matches_test.csv (prototyp)
    DATA_MODE = "db"   -> czyta z Postgresa zapytaniami v5.1 (parametry season+play)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

DATA_MODE = os.environ.get("PM_DATA_MODE", "csv")

# kolumny liczbowe (uwaga: score'y maja przecinek dziesietny z eksportu)
NUMERIC_COMMA = ["match_score", "m_overall_score", "m_season_score",
                 "overall_score", "season_score",
                 "global_last_overall_score", "global_last_season_score"]
NUMERIC_PLAIN = ["minutes", "goals", "yellow_cards", "red_cards",
                 "est_birth_year", "age_at_match", "senior_minutes",
                 "senior_matches_played", "senior_squad_apps",
                 "play_cohort_birth_year", "matches_count"]

AXES = ["Ofensywa", "Jakość", "Forma", "Konsekwencja", "Dostępność", "Dyscyplina"]
PM_WEIGHTS = {"Jakość": 0.40, "Ofensywa": 0.25, "Forma": 0.15,
              "Dostępność": 0.10, "Konsekwencja": 0.10}


# ----------------------------------------------------------------------------
# WCZYTYWANIE DANYCH
# ----------------------------------------------------------------------------
def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COMMA:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False)
                     .replace({"NULL": None, "": None, "NaN": None}),
                errors="coerce")
    for c in NUMERIC_PLAIN:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace({"NULL": None, "": None}), errors="coerce")
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    for c in ["in_selected_play", "is_junior_comp", "gra_ze_starszymi"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False}).where(lambda s: s.notna(), other=pd.NA)
    return df


def _read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "cp1250", "latin-1"):   # eksport PBI bywa cp1250
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1")


@st.cache_data(show_spinner=False)
def load_data(stats_path="stats_test.csv", matches_path="matches_test.csv"):
    if DATA_MODE == "db":
        return load_from_db()
    stats = _coerce(_read_csv(stats_path))
    matches = _coerce(_read_csv(matches_path))
    return stats, matches


def load_from_db(season_id=None, play_id=None):
    """Tryb produkcyjny: psycopg2 + zapytania v5.1 z parametrami.
    Wklej tresc analiza_stats5_1.sql / analiza_mecze5_1.sql, podmieniajac
    literaly w CTE params na %(season_id)s / %(play_id)s."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ["PGHOST"], dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"], password=os.environ["PGPASSWORD"],
        port=os.environ.get("PGPORT", "5432"))
    params = {"season_id": season_id, "play_id": play_id}
    stats = pd.read_sql(open("analiza_stats5_1.sql").read(), conn, params=params)
    matches = pd.read_sql(open("analiza_mecze5_1.sql").read(), conn, params=params)
    conn.close()
    return _coerce(stats), _coerce(matches)


# ----------------------------------------------------------------------------
# METRYKI
# ----------------------------------------------------------------------------
def player_attrs(stats: pd.DataFrame) -> pd.DataFrame:
    """Atrybuty per gracz z wierszy wybranego play."""
    sel = stats[stats["in_selected_play"] == True].copy()
    sel["zawodnik"] = sel["firstname"].fillna("") + " " + sel["lastname"].fillna("")
    cols = ["player_id", "zawodnik", "team_name", "est_birth_year",
            "gra_ze_starszymi", "status_seniorski", "senior_minutes",
            "roczniki_w_gore"]
    cols = [c for c in cols if c in sel.columns]
    return sel[cols].drop_duplicates("player_id")


def compute_metrics(matches: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """Metryki per gracz liczone z meczow w wybranej lidze (in_selected_play)."""
    m = matches[matches["in_selected_play"] == True].copy()

    def per_player(g):
        mins = g["minutes"].sum()
        goals = g["goals"].sum()
        cards = g["yellow_cards"].sum() + 2 * g["red_cards"].sum()
        scores = g["match_score"].dropna()
        # forma: srednia z 5 ostatnich vs srednia sezonu
        gg = g.sort_values("match_date")
        s_all = gg["match_score"].dropna()
        last5 = gg["match_score"].dropna().tail(5)
        forma = ((last5.mean() - s_all.mean()) / s_all.mean()
                 if len(s_all) >= 3 and s_all.mean() else np.nan)
        return pd.Series({
            "minuty": mins,
            "mecze": g["match_id"].nunique(),
            "gole": goals,
            "avg_score": scores.mean() if len(scores) else np.nan,
            "gole_per90": (goals / mins * 90) if mins else np.nan,
            "kartki_per90": (cards / mins * 90) if mins else np.nan,
            "konsekwencja": (1 / (1 + scores.std(ddof=0))) if len(scores) >= 2 else np.nan,
            "forma": forma,
        })

    met = m.groupby("player_id").apply(per_player).reset_index()
    df = attrs.merge(met, on="player_id", how="left")
    return add_percentiles(df)


def add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    df["Ofensywa"] = df["gole_per90"].rank(pct=True)
    df["Jakość"] = df["avg_score"].rank(pct=True)
    df["Forma"] = df["forma"].rank(pct=True)
    df["Konsekwencja"] = df["konsekwencja"].rank(pct=True)
    df["Dostępność"] = df["minuty"].rank(pct=True)
    df["Dyscyplina"] = (-df["kartki_per90"]).rank(pct=True)   # mniej kartek = wyzszy percentyl
    df["PM_Index"] = sum(df[ax].fillna(0) * w for ax, w in PM_WEIGHTS.items())
    return df


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="PlayMaker Pro", layout="wide")
    stats, matches = load_data()
    attrs = player_attrs(stats)
    data = compute_metrics(matches, attrs)

    liga = stats.loc[stats["in_selected_play"] == True, "play_name"].dropna().iloc[0] \
        if (stats["in_selected_play"] == True).any() else "—"
    st.title("PlayMaker Pro — raport rozgrywek")
    st.caption(f"Liga: **{liga}**  ·  zawodników: {len(data)}  ·  "
               f"graczy w seniorach: {(data['senior_minutes'].fillna(0) > 0).sum()}")

    tab1, tab2 = st.tabs(["🔍 Odkrywanie", "📊 Porównanie"])

    # ---- ODKRYWANIE ----
    with tab1:
        c1, c2, c3 = st.columns(3)
        min_minut = c1.slider("Min. minut w lidze", 0, int(data["minuty"].max() or 0), 300)
        only_up = c2.checkbox("Tylko grający ze starszymi rocznikiem")
        only_sen = c3.checkbox("Tylko z minutami w seniorach")

        f = data[data["minuty"].fillna(0) >= min_minut].copy()
        if only_up:
            f = f[f["gra_ze_starszymi"] == True]
        if only_sen:
            f = f[f["senior_minutes"].fillna(0) > 0]

        st.subheader("Scatter: dostępność × jakość (kolor = rocznik, rozmiar = minuty w seniorach)")
        sc = f.dropna(subset=["avg_score", "minuty"])
        if len(sc):
            fig = px.scatter(
                sc, x="minuty", y="avg_score",
                color=sc["est_birth_year"].astype("Int64").astype(str),
                size=sc["senior_minutes"].fillna(0) + 1,
                hover_name="zawodnik",
                hover_data={"team_name": True, "PM_Index": ":.2f", "gole": True},
                labels={"minuty": "Minuty w lidze", "avg_score": "Śr. ocena meczu",
                        "color": "Rocznik"})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Leaderboard (PM Index)")
        show = ["zawodnik", "team_name", "est_birth_year", "PM_Index",
                "avg_score", "gole_per90", "minuty", "senior_minutes",
                "gra_ze_starszymi", "status_seniorski"]
        show = [c for c in show if c in f.columns]
        st.dataframe(
            f.sort_values("PM_Index", ascending=False)[show].reset_index(drop=True),
            use_container_width=True, height=480)

    # ---- PORÓWNANIE ----
    with tab2:
        opcje = data.sort_values("PM_Index", ascending=False)["zawodnik"].tolist()
        wybrani = st.multiselect("Wybierz 2–5 zawodników", opcje, max_selections=5,
                                 default=opcje[:3])
        if len(wybrani) >= 1:
            fig = go.Figure()
            for z in wybrani:
                row = data[data["zawodnik"] == z]
                if row.empty:
                    continue
                vals = [float(row[a].fillna(0).iloc[0]) for a in AXES]
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=AXES + [AXES[0]], fill="toself", name=z))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
                              showlegend=True, height=560)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Osie to percentyl względem stawki tej ligi (0–1).")

            kk = data[data["zawodnik"].isin(wybrani)][
                [c for c in ["zawodnik", "est_birth_year", "status_seniorski",
                             "senior_minutes", "avg_score", "gole_per90",
                             "minuty", "PM_Index"] if c in data.columns]]
            st.dataframe(kk.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Zaznacz przynajmniej jednego zawodnika.")


if __name__ == "__main__":
    main()
