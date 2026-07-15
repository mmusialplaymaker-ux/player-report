-- ─────────────────────────────────────────────────────────────────────────────
-- rocznik_historia.sql — HISTORIA GRY W KATEGORIACH (wejście do wyznacz_rocznik.py)
--
-- Po co: rocznik w bazie pochodzi z WIEKU ze scrape'u i jest systematycznie o rok
--        za młody. Ale w danej lidze nie zagra nikt starszy niż rocznik graniczny,
--        więc historia gry daje twarde DOLNE ograniczenie na rocznik (metoda „floor”).
--
-- Zakres: sezon bieżący 25/26 + 4 wstecz (24/25, 23/24, 22/23, 21/22).
--         Najmłodsze roczniki (2013–2014) w najstarszych sezonach po prostu nie wystąpią
--         — to normalne, wyznacz_rocznik.py oznaczy je BRAK_HISTORII.
--
-- Sezony dopasowujemy po PREFIKSIE season_id (8 znaków) — te same, których używa
-- SEASON_END_YEAR w wyznacz_rocznik.py. Jeśli dodasz sezon, dopisz go w OBU miejscach.
--
-- Wyjście (1 wiersz na zawodnik × sezon × liga):
--   player_id, firstname, lastname, est_birth_year, season_id, league_name, matches, minutes
--
-- Eksport (duży wynik → strumieniowo, nie przez Excel):
--   python export_all.py --sql rocznik_historia.sql --out rocznik_historia.csv
-- Potem:
--   python wyznacz_rocznik.py --hist rocznik_historia.csv
-- ─────────────────────────────────────────────────────────────────────────────

WITH params AS (
    SELECT
        ARRAY[
            'e9d66181',   -- 25/26 (bieżący)
            '4be7b40c',   -- 24/25
            '29d748c8',   -- 23/24
            'b004c86c',   -- 22/23
            'b682af6d'    -- 21/22
        ]::text[] AS season_prefixes,
        2003::int AS rok_od,     -- <<< ten sam zakres co w kohorta_rocznik.sql
        2014::int AS rok_do
)

SELECT
    m.player_id,
    p.firstname,
    p.lastname,
    substring(p.date_of_birth::text from '[0-9]{4}')::int  AS est_birth_year,
    m.season_id,
    l.name                                                 AS league_name,
    COUNT(DISTINCT m.match_id)                             AS matches,
    SUM(COALESCE(m.minutes, 0))                            AS minutes

FROM pm_player_match_stats m
JOIN players p       ON p._id = m.player_id
LEFT JOIN leagues l  ON l._id = m.league_id
CROSS JOIN params prm

WHERE left(m.season_id::text, 8) = ANY(prm.season_prefixes)
  AND substring(p.date_of_birth::text from '[0-9]{4}')::int BETWEEN prm.rok_od AND prm.rok_do
  AND l.name IS NOT NULL          -- bez nazwy ligi nie da się wyznaczyć granicy wiekowej

GROUP BY m.player_id, p.firstname, p.lastname,
         substring(p.date_of_birth::text from '[0-9]{4}')::int,
         m.season_id, l.name

ORDER BY m.player_id, m.season_id, l.name;
