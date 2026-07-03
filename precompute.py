"""
precompute.py — uruchom JEDNORAZOWO, LOKALNIE (tam gdzie masz duży kohorta.csv).
────────────────────────────────────────────────────────────────────────────────
Z dużego match-level `kohorta.csv` robi dwa MAŁE pliki, które idą do repo/chmury:
  • kohorta_agg.csv.gz    — 1 wiersz na zawodnika (CAŁA kohorta → realne N do percentyli)
  • kohorta_trend.csv.gz  — minimalne wiersze meczowe (player_id, match_date, minutes, _sc)
                            tylko do wykresu formy; tylko mecze z minutami > 0

Wyklucza dziewczynki (raport jest o roczniku chłopców) — heurystyką imion na „-a"
minus lista męskich wyjątków. PM Score liczony tym samym wzorem co wszędzie
(import compute_pm_score z app.py).

Uruchomienie:
    pip install pandas numpy streamlit
    python precompute.py                 # czyta kohorta.csv
    python precompute.py sciezka.csv     # albo inny plik źródłowy
"""
import os
import sys
import numpy as np
import pandas as pd

from app import compute_pm_score, _coerce, _cat_maxyear_series

SRC = sys.argv[1] if len(sys.argv) > 1 else "kohorta.csv"

# ── męskie wyjątki: imiona kończące się na „-a", które NIE są kobiece ──────────
MALE_EXCEPTIONS = {
    # utrzymywana lista (slim_data.py)
    "kuba", "luka", "nikita", "barnaba", "ilia", "illya", "mikita",
    "danila", "oleksa", "seva", "diaa",
    # dodatkowe warianty zaobserwowane w tym pliku — ZAUDYTUJ / edytuj wg potrzeb:
    "dima", "mykola", "mykyta", "ilya", "illia",
}


def _is_female(firstname):
    f = str(firstname).strip().lower()
    return f.endswith("a") and f not in MALE_EXCEPTIONS


def _read_any(path):
    for enc in ("cp1250", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise SystemExit("Nie udało się odczytać CSV (kodowanie).")


print(f"Wczytuję {SRC} ...")
m = _coerce(_read_any(SRC))
n0 = m["player_id"].nunique()

# ── odrzuć dziewczynki (na poziomie zawodnika) ────────────────────────────────
fem_mask = m["firstname"].map(_is_female)
removed_names = sorted(m.loc[fem_mask, "firstname"].dropna().str.strip().str.title().unique())
m = m[~fem_mask].copy()
n1 = m["player_id"].nunique()
print(f"  {len(m)} wierszy meczowych, {n1} zawodników "
      f"(odrzucono {n0 - n1} dziewczynek z {n0})")
if removed_names:
    print("  przykłady odrzuconych imion:", ", ".join(removed_names[:25])
          + (" ..." if len(removed_names) > 25 else ""))

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


out = out.join(gp[["match_date", "_sp"]].apply(_form))

py = m["est_birth_year"]
jun_older = (mn > 0) & m["_maxy"].notna() & py.notna() & (py > m["_maxy"])
out["roczniki_w_gore"] = (py - m["_maxy"]).where(jun_older).groupby(m["player_id"]).max()
out["gra_ze_starszymi"] = jun_older.groupby(m["player_id"]).any().reindex(out.index).fillna(False)

is_clj = m["league_name"].astype(str).str.contains(r"\bCLJ\b|Centralna Liga Junior",
                                                   case=False, regex=True, na=False)
out["clj_minutes"] = (mn * is_clj).groupby(m["player_id"]).sum()
is_senior = (~m["is_junior_comp"].fillna(False)) & (m["age_at_match"].between(12, 19))
out["senior_minutes"] = (mn * is_senior).groupby(m["player_id"]).sum()
out["kategorie"] = gp["league_name"].agg(lambda s: "; ".join(sorted(set(s.dropna().astype(str)))))

out = out.reset_index()
out.to_csv("kohorta_agg.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

trend = pd.DataFrame({"player_id": m["player_id"], "match_date": m["match_date"],
                      "league_name": m["league_name"], "minutes": mn,
                      "goals": m["goals"], "yellow_cards": m["yellow_cards"],
                      "red_cards": m["red_cards"], "match_result": m["match_result"],
                      "team_side": m["team_side"], "_sc": m["_sc"]})
trend = trend[trend["minutes"] > 0]
trend.to_csv("kohorta_trend.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

print(f"\n✓ kohorta_agg.csv.gz    {os.path.getsize('kohorta_agg.csv.gz')/1e6:6.2f} MB  "
      f"({len(out)} zawodników)")
print(f"✓ kohorta_trend.csv.gz  {os.path.getsize('kohorta_trend.csv.gz')/1e6:6.2f} MB  "
      f"({len(trend)} wierszy)")
print("\nDo repo idą TE DWA pliki (+ kod). Duży kohorta.csv zostaje lokalnie (jest w .gitignore).")