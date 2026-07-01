# Raport rocznikowy — PlayMaker (v1)

Zawodnik na tle **całego swojego rocznika w Polsce**: percentyl, rozkład, profil (radar),
trend formy, kategoria PZPN (A1/A2…) i gra ze starszymi, top 10 rocznika.
Odbiorca: zawodnik / rodzic.

PM Score liczony jest tym samym wzorem co reszta systemu — apka importuje `compute_pm_score`
z `app.py` (jedno źródło prawdy), więc wynik jest **league-aware** i porównywalny między ligami.

---

## Pliki w repo

| plik | rola | wymagany na Streamlit Cloud |
|---|---|---|
| `raport_rocznikowy.py` | główna apka (main file) | tak |
| `app.py` | źródło wzoru PM Score (import) | tak |
| `kohorta.csv` | dane (jeden lub wiele roczników) | tak (tryb CSV) |
| `requirements.txt` | zależności | tak |
| `kohorta_rocznik.sql` | zapytanie do wygenerowania danych / tryb DB | opcjonalnie |
| `play_id_rank_p_v7.json` | poziomy rozgrywek (v7) | opcjonalnie (jest fallback) |

> **Uwaga:** Streamlit Cloud NIE dosięgnie bazy przez tunel SSH. W chmurze działa tylko
> tryb CSV — dlatego `kohorta.csv` musi być w repo. Tryb DB (`PM_DATA_MODE=db`) działa
> tylko lokalnie, przy aktywnym tunelu.

---

## Krok 1 — wygeneruj `kohorta.csv`

1. Otwórz `kohorta_rocznik.sql`, ustaw w bloku `params` (linie 17–18) **sezon** i **rocznik**.
2. Odpal w kliencie bazy, wynik zapisz jako `kohorta.csv` (UTF-8).
3. Chcesz kilka roczników w jednym demie? Odpal SQL osobno dla każdego rocznika i sklej pliki
   (te same nagłówki) — apka filtruje po kolumnie `est_birth_year`.

## Krok 2 — wrzuć do repo

Repo: `mmusialplaymaker-ux/player-report` — **ustaw jako prywatne** (dane nieletnich).

```bash
git clone https://github.com/mmusialplaymaker-ux/player-report.git
cd player-report
# skopiuj tu: raport_rocznikowy.py app.py kohorta_rocznik.sql kohorta.csv requirements.txt README.md
git add -A
git commit -m "v1: raport rocznikowy"
git remote -v          # sprawdź, że origin wskazuje na player-report
git push -u origin main
```

## Krok 3 — deploy na Streamlit Cloud

1. https://share.streamlit.io → **New app** → repo `mmusialplaymaker-ux/player-report`,
   branch `main`, **Main file path: `raport_rocznikowy.py`**.
2. **Advanced → Secrets**:
   ```toml
   APP_PASSWORD = "wybierz-haslo"
   # PM_DATA_MODE domyślnie "csv" — nie trzeba ustawiać
   # PM_MIN_MINUTES = "300"   # opcjonalnie: próg minut do oceny
   ```
3. **Deploy**. Link + hasło wyślij koledze.

## Deep-link do jednego zawodnika

```
https://<twoja-apka>.streamlit.app/?player=<player_id>
```
Otwiera od razu danego zawodnika (bez wyszukiwarki) — wygodne, gdy pokazujesz konkretne dziecko.

---

## Na co patrzeć przy pierwszym realnym przebiegu

- **Próg minut** (domyślnie 300): w młodszych rocznikach sezon bywa krótszy — sprawdź, czy nie
  wycina zbyt wielu. Regulujesz suwakiem w panelu lub sekretem `PM_MIN_MINUTES`.
- **Rozkład PM Score w roczniku**: jeśli jest zdominowany przez 1–2 poziomy lig, to sygnał, że
  `leagueMultiplier` dla tego rocznika wart jest kalibracji, zanim pokażesz to rodzicom.