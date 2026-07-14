"""
precompute.py — JEDNORAZOWO, LOKALNIE. Zamyka sezon w gotowe pliki do apki.
────────────────────────────────────────────────────────────────────────────────
Wejście:  kohorta_ALL.csv  (JEDEN eksport całego zakresu 2003–2014 z kohorta_rocznik.sql)
          — albo wiele plików kohorta_*.csv; oba tryby działają tak samo.
          rocznik_status.csv  (opcjonalnie, z wyznacz_rocznik.py — korekta rocznika)
          teamy_kluby_25_26.csv (opcjonalnie — mapa nazw)

Wyjście (do repo):
  data/kohorta_agg.parquet          — 1 wiersz na zawodnika, WSZYSTKIE roczniki
  data/matches/rocznik_final=<ROK>/*.parquet  — mecze, partycjonowane po roczniku

Jak działa (strumieniowo, żeby nie wciągać całego GB do RAM):
  ETAP 1 (split):     czyta wejście chunkami, nakłada korektę rocznika, filtruje
                      dziewczynki, rozwiązuje nazwy, ROZDZIELA wiersze do przegródek
                      data/_split/rocznik=<ROK>/ — po SKORYGOWANYM roczniku.
  ETAP 2 (aggregate): każdą przegródkę (jeden rocznik) wczytuje osobno, liczy PM Score
                      i agregat, zapisuje partycję meczów. Na koniec sprząta _split.

Uruchomienie:
    pip install pandas numpy pyarrow streamlit
    python precompute.py kohorta_ALL.csv
    python precompute.py                    # bierze wszystkie kohorta_*.csv w folderze
"""
import glob
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

from app import compute_pm_score, _coerce, _cat_maxyear_series

OUT_DIR = "data"
AGG_PATH = os.path.join(OUT_DIR, "kohorta_agg.parquet")
MATCHES_DIR = os.path.join(OUT_DIR, "matches")
SPLIT_DIR = os.path.join(OUT_DIR, "_split")
CHUNK = 400_000

MALE_EXCEPTIONS = {"kuba", "luka", "nikita", "barnaba", "ilia", "illya", "mikita", "danila",
                   "oleksa", "seva", "diaa", "dima", "mykola", "mykyta", "ilya", "illia"}


def _is_female(firstname):
    f = str(firstname).strip().lower()
    return f.endswith("a") and f not in MALE_EXCEPTIONS


SZCZEBEL_NAZWA = {5: "CLJ / Makroregionalna", 4: "I liga wojewódzka", 3: "II liga wojewódzka",
                  2: "III liga wojewódzka", 1: "liga okręgowa", 0: "—"}
_ROMAN = {"i": 1, "ii": 2, "iii": 3}
_WOJ = ("małopolsk", "śląsk", "świętokrzysk", "dolnośląsk", "wielkopolsk", "pomorsk", "mazowieck",
        "lubelsk", "podkarpack", "kujawsko", "warmińsko", "zachodniopomorsk", "lubusk", "łódzk",
        "opolsk", "podlask")


def _szczebel(play_name, league_name):
    s = f"{play_name} {league_name}".lower()
    if "clj" in s or "centralna liga" in s or "makroregion" in s:
        return 5
    mm = re.search(r"\b(i{1,3})\s+liga\s+(wojew|okr)", s)
    if mm:
        n, typ = _ROMAN[mm.group(1)], mm.group(2)
        return {1: 4, 2: 3, 3: 2}[n] if typ == "wojew" else 1
    if "okręgow" in s or "okregow" in s:
        return 1
    if re.match(r"^[a-ząćęłńóśźż\- ]+:", s):
        return 1
    if any(w in s for w in _WOJ) or re.search(r"\bwlj\b|wojewódzk", s):
        return 4
    return 0


def _read_flex(path):
    for enc in ("utf-8-sig", "utf-8", "cp1250", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise SystemExit(f"Nie udało się odczytać {path}.")


def _detect_encoding(path):
    for enc in ("utf-8-sig", "utf-8", "cp1250", "latin-1"):
        try:
            pd.read_csv(path, encoding=enc, nrows=200)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"


def load_status(path="rocznik_status.csv"):
    if not os.path.exists(path):
        print("  (rocznik_status.csv nie znaleziony — rocznik z wieku, pewnosc='szacowany')")
        return {}, {}
    st = _read_flex(path)
    if "player_id" not in st.columns:
        print("  (rocznik_status.csv bez player_id — pomijam korektę)")
        return {}, {}
    fin, pew = {}, {}
    ycol = "rocznik_final" if "rocznik_final" in st.columns else None
    for _, r in st.iterrows():
        pid = str(r["player_id"])
        status = str(r.get("status", "")).upper()
        y = r.get(ycol) if ycol else None
        try:
            y = int(float(y)) if str(y).strip() not in ("", "nan", "None") else None
        except (ValueError, TypeError):
            y = None
        if y is not None:
            fin[pid] = y
        pew[pid] = ("potwierdzony" if status == "POTWIERDZONY"
                    else "skorygowany" if status == "KOREKTA"
                    else "szacowany")
    print(f"  korekta rocznika: {len(fin)} zawodników z rocznik_final")
    return fin, pew


def load_name_maps(path_base="teamy_kluby_25_26"):
    for cand in (path_base + ".csv", path_base, path_base + ".xlsx"):
        if os.path.exists(cand):
            path = cand
            break
    else:
        print("  (mapa nazw nie znaleziona — nazwy z bazy)")
        return {}, {}
    try:
        mp = pd.read_excel(path) if path.endswith("xlsx") else _read_flex(path)
    except Exception as e:
        print(f"  (nie wczytano mapy nazw: {e})")
        return {}, {}
    cols = {c.lower().strip(): c for c in mp.columns}

    def pick(idkw, namekws):
        idc = next((cols[k] for k in cols if idkw in k and "id" in k), None)
        namec = next((cols[k] for k in cols for nk in namekws if nk in k and "id" not in k), None)
        return idc, namec

    ti, tn = pick("team", ["team", "nazwa", "name"])
    ci, cn = pick("club", ["club", "klub", "nazwa", "name"])
    tmap = dict(zip(mp[ti].astype(str), mp[tn].astype(str))) if ti and tn else {}
    cmap = dict(zip(mp[ci].astype(str), mp[cn].astype(str))) if ci and cn else {}
    print(f"  mapa nazw: team {len(tmap)} | club {len(cmap)}")
    return tmap, cmap


def _resolve(series_id, series_name, id2name):
    if not id2name or series_id is None:
        return series_name
    return series_id.astype(str).map(id2name).fillna(series_name)


# ── ETAP 1: przygotowanie chunku (row-independent) + rozdział do przegródek ──
def _prep_chunk(m, fin, pew, tmap, cmap):
    m = _coerce(m)
    m = m[~m["firstname"].map(_is_female)].copy()
    m["zawodnik"] = (m["firstname"].fillna("") + " " + m["lastname"].fillna("")).str.strip()
    if "club_id" in m.columns:
        m["club_name"] = _resolve(m["club_id"], m["club_name"], cmap)
    if "team_id" in m.columns:
        m["team_name"] = _resolve(m["team_id"], m["team_name"], tmap)
    if "opponent_id" in m.columns:
        m["opponent_name"] = _resolve(m["opponent_id"], m.get("opponent_name"), tmap)
    elif "opponent_name" not in m.columns:
        m["opponent_name"] = np.nan
    pid_str = m["player_id"].astype(str)
    corr = pid_str.map(fin)
    m["rocznik_final"] = pd.to_numeric(corr, errors="coerce").fillna(
        pd.to_numeric(m["est_birth_year"], errors="coerce"))
    m["rocznik_pewnosc"] = pid_str.map(pew).fillna("szacowany")
    return m[m["rocznik_final"].notna()]


def split_inputs(inputs, fin, pew, tmap, cmap):
    shutil.rmtree(SPLIT_DIR, ignore_errors=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    part = 0
    for path in inputs:
        enc = _detect_encoding(path)
        print(f"\n→ split {path} (enc={enc})")
        for ci, chunk in enumerate(pd.read_csv(path, encoding=enc, chunksize=CHUNK, low_memory=False)):
            m = _prep_chunk(chunk, fin, pew, tmap, cmap)
            for Y, sub in m.groupby(m["rocznik_final"].astype(int)):
                d = os.path.join(SPLIT_DIR, f"rocznik={int(Y)}")
                os.makedirs(d, exist_ok=True)
                sub.to_parquet(os.path.join(d, f"part_{part}.parquet"), index=False)
                part += 1
            print(f"   chunk {ci}: {len(m)} wierszy → roczniki {sorted(m['rocznik_final'].astype(int).unique().tolist())}")


# ── ETAP 2: agregacja jednego rocznika ───────────────────────────────────────
def aggregate_rocznik(Y):
    m = pd.read_parquet(os.path.join(SPLIT_DIR, f"rocznik={Y}"))
    m["match_date"] = pd.to_datetime(m["match_date"], errors="coerce")
    m["rocznik_final"] = pd.to_numeric(m["rocznik_final"], errors="coerce")

    comp = compute_pm_score(m)
    m["_sc"] = comp["score"].values
    m["_sp"] = comp["stats_part"].values
    mn = pd.to_numeric(m["minutes"], errors="coerce").fillna(0)
    m["_mn"] = mn
    m["_maxy"] = _cat_maxyear_series(m)
    m["_szcz"] = [_szczebel(p, l) for p, l in zip(
        m.get("play_name", pd.Series("", index=m.index)).fillna(""), m["league_name"].fillna(""))]

    gp = m.groupby("player_id")
    den = mn.groupby(m["player_id"]).sum().replace(0, np.nan)

    def wmean(c):
        return (m[c] * mn).groupby(m["player_id"]).sum() / den

    out = pd.DataFrame(index=den.index)
    out.index.name = "player_id"
    out["zawodnik"] = gp["zawodnik"].first()
    out["rocznik_final"] = int(Y)
    out["rocznik_pewnosc"] = gp["rocznik_pewnosc"].first()
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
    out = out.join(gp[["match_date", "_sp"]].apply(_form))

    py = m["rocznik_final"].astype("float")
    jun = (mn > 0) & m["_maxy"].notna() & py.notna() & (py > m["_maxy"])
    out["roczniki_w_gore"] = (py - m["_maxy"]).where(jun).groupby(m["player_id"]).max()
    out["gra_ze_starszymi"] = jun.groupby(m["player_id"]).any().reindex(out.index).fillna(False)
    is_clj = m["league_name"].astype(str).str.contains(r"\bCLJ\b|Centralna Liga Junior",
                                                       case=False, regex=True, na=False)
    out["clj_minutes"] = (mn * is_clj).groupby(m["player_id"]).sum()
    is_sen = (~m["is_junior_comp"].fillna(False)) & (m["age_at_match"].between(12, 19))
    out["senior_minutes"] = (mn * is_sen).groupby(m["player_id"]).sum()
    out["kategoria_glowna"] = (m.assign(_w=mn).groupby(["player_id", "league_name"])["_w"].sum()
                               .reset_index().sort_values("_w", ascending=False)
                               .groupby("player_id")["league_name"].first().reindex(out.index))
    sz = (m[m["_szcz"] > 0].assign(_w=mn[m["_szcz"] > 0]).groupby(["player_id", "_szcz"])["_w"].sum()
          .reset_index().sort_values("_w", ascending=False).groupby("player_id").first())
    out["szczebel"] = sz["_szcz"].reindex(out.index).fillna(0).astype(int)
    out["szczebel_nazwa"] = out["szczebel"].map(SZCZEBEL_NAZWA)
    out = out.reset_index()

    matches = pd.DataFrame({
        "player_id": m["player_id"], "rocznik_final": int(Y),
        "match_date": m["match_date"], "league_name": m["league_name"],
        "play_name": m.get("play_name"), "opponent_name": m.get("opponent_name"),
        "minutes": mn, "goals": m["goals"], "yellow_cards": m["yellow_cards"],
        "red_cards": m["red_cards"], "match_result": m["match_result"],
        "team_side": m["team_side"], "_sc": m["_sc"]})
    matches = matches[matches["minutes"] > 0]
    return out, matches


def main():
    inputs = sys.argv[1:] or sorted(glob.glob("kohorta_*.csv"))
    if not inputs:
        raise SystemExit("Brak plików wejściowych. Podaj kohorta_ALL.csv albo połóż kohorta_*.csv w folderze.")
    print("Pliki wejściowe:", ", ".join(inputs))
    fin, pew = load_status()
    tmap, cmap = load_name_maps()

    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.rmtree(MATCHES_DIR, ignore_errors=True)
    os.makedirs(MATCHES_DIR, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    print("\n=== ETAP 1: rozdział po skorygowanym roczniku ===")
    split_inputs(inputs, fin, pew, tmap, cmap)
    roczniki = sorted(int(d.split("=")[1]) for d in os.listdir(SPLIT_DIR)
                      if d.startswith("rocznik="))
    print(f"\nRoczniki do agregacji: {roczniki}")

    print("\n=== ETAP 2: agregacja per rocznik ===")
    all_agg = []
    for Y in roczniki:
        agg, matches = aggregate_rocznik(Y)
        all_agg.append(agg)
        matches["rocznik_final"] = matches["rocznik_final"].astype("int64")
        pq.write_to_dataset(pa.Table.from_pandas(matches, preserve_index=False),
                            root_path=MATCHES_DIR, partition_cols=["rocznik_final"])
        print(f"  {Y}: {agg['player_id'].nunique()} zawodników | {len(matches)} meczów")

    agg = pd.concat(all_agg, ignore_index=True).drop_duplicates("player_id", keep="first")
    agg.to_parquet(AGG_PATH, index=False)
    shutil.rmtree(SPLIT_DIR, ignore_errors=True)

    sz_agg = os.path.getsize(AGG_PATH) / 1e6
    sz_m = sum(os.path.getsize(os.path.join(r, fn))
               for r, _, fs in os.walk(MATCHES_DIR) for fn in fs) / 1e6
    print(f"\n✓ {AGG_PATH}  {sz_agg:.1f} MB  ({len(agg)} zawodników, roczniki {roczniki})")
    print(f"✓ {MATCHES_DIR}/  {sz_m:.1f} MB (partycje po roczniku)")
    print("\nRozkład pewności rocznika:")
    print(agg["rocznik_pewnosc"].value_counts().to_string())
    print("\nDo repo idzie CAŁY folder data/ (bez _split). Duże kohorta_*.csv zostają lokalnie.")


if __name__ == "__main__":
    main()
