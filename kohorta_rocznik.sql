-- kohorta_rocznik.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- KRAJOWA kohorta jednego rocznika: wszystkie mecze sezonu wszystkich zawodników
-- urodzonych w danym roku (bez filtra ligi/regionu). Zwraca wiersze per mecz —
-- kolumny są takie, jakich oczekuje compute_pm_score() z app.py, więc PM Score
-- liczymy potem w Pythonie (jedno źródło prawdy).
--
-- URUCHOMIENIE RĘCZNE (DBeaver/pgAdmin → zapisz wynik jako kohorta.csv):
--   ustaw DWIE wartości w bloku params poniżej (sezon + rocznik) i odpal całość.
-- APKA (tryb DB): podstawia sezon + rocznik sama (nie musisz nic edytować).
--
-- date_of_birth jest tekstem → wyciągamy rok przez substring(... '[0-9]{4}').
-- ─────────────────────────────────────────────────────────────────────────────

WITH params AS (
    SELECT
        'e9d66181-d03e-4bb3-b889-4da848f4831d'::text AS season_id,   -- <<< USTAW sezon (domyślnie 25/26)
        2003::int                                    AS rok_od,       -- <<< dolny rocznik (włącznie)
        2014::int                                    AS rok_do        -- <<< górny rocznik (włącznie)
),

-- wszyscy zawodnicy z zakresu roczników, którzy mają JAKIKOLWIEK mecz w sezonie
roster AS (
    SELECT DISTINCT m.player_id
    FROM pm_player_match_stats m
    JOIN players p ON p._id = m.player_id
    CROSS JOIN params prm
    WHERE m.season_id = prm.season_id
      AND substring(p.date_of_birth::text from '[0-9]{4}')::int BETWEEN prm.rok_od AND prm.rok_do
)

SELECT
    m.player_id,
    p.firstname,
    p.lastname,
    m.match_id,
    m.match_date,
    m.play_id,
    pl.name  AS play_name,
    rg.name  AS region_name,
    m.league_id,
    l.name   AS league_name,
    m.team_id,
    t.name   AS team_name,
    m.club_id,
    c.name   AS club_name,
    m.minutes,
    m.goals,
    m.yellow_cards,
    m.red_cards,

    CASE m.result
        WHEN 0 THEN 'wygrana' WHEN 1 THEN 'remis' WHEN 2 THEN 'porażka' ELSE 'unknown'
    END AS match_result,

    -- strona boiska (potrzebna w scoringu: side_ratio). Preferuj s.team_side,
    -- a gdy brak rekordu score — policz z m.team_id vs matches.host/guest.
    CASE COALESCE(s.team_side,
                  CASE WHEN m.team_id = mat.host_id  THEN 'host'
                       WHEN m.team_id = mat.guest_id THEN 'guest' END)
        WHEN 'host'  THEN 'gospodarz'
        WHEN 'guest' THEN 'gość'
        ELSE 'unknown'
    END AS team_side,

    -- ── PRZECIWNIK (niezależnie od score'a: z m.team_id vs matches.host/guest) ──
    CASE COALESCE(s.team_side,
                  CASE WHEN m.team_id = mat.host_id  THEN 'host'
                       WHEN m.team_id = mat.guest_id THEN 'guest' END)
        WHEN 'host'  THEN mat.guest_id
        WHEN 'guest' THEN mat.host_id ELSE NULL END                   AS opponent_id,
    CASE COALESCE(s.team_side,
                  CASE WHEN m.team_id = mat.host_id  THEN 'host'
                       WHEN m.team_id = mat.guest_id THEN 'guest' END)
        WHEN 'host'  THEN guest_team.name
        WHEN 'guest' THEN host_team.name ELSE NULL END                AS opponent_name,

    s.age AS player_age,                                        -- wiek z score (jeśli jest)
    substring(p.date_of_birth::text from '[0-9]{4}')::int AS est_birth_year,
    (substring(m.match_date::text from '[0-9]{4}')::int
        - substring(p.date_of_birth::text from '[0-9]{4}')::int) AS age_at_match,

    -- rozgrywki juniorskie? (do etykiet / filtra kategorii)
    CASE
        WHEN l.name ~* '^(A1|A2|B1|B2|C1|C2|D1|D2)$'
             OR l.name ILIKE 'CLJ%' OR l.name ILIKE '%U-1%'
             OR pl.name ~* '(junior|trampkarz|m[lł]odzik|[zż]ak|orlik|skrzat)'
        THEN true ELSE false
    END AS is_junior_comp

    -- ── OPCJONALNIE: pozycja (odkomentuj gdy potwierdzisz kolumnę) ──
    -- , p.position AS position          -- albo s.position / m.position — do sprawdzenia

FROM pm_player_match_stats m
CROSS JOIN params prm
JOIN roster r        ON r.player_id = m.player_id
LEFT JOIN players p  ON m.player_id = p._id
LEFT JOIN plays pl   ON m.play_id   = pl._id
LEFT JOIN regions rg ON pl.region_id = rg._id
LEFT JOIN teams t    ON m.team_id   = t._id
LEFT JOIN clubs c    ON m.club_id   = c._id
LEFT JOIN leagues l  ON m.league_id = l._id
LEFT JOIN matches mat ON m.match_id = mat._id
LEFT JOIN teams host_team  ON mat.host_id  = host_team._id
LEFT JOIN teams guest_team ON mat.guest_id = guest_team._id
LEFT JOIN pm_player_match_score s
    ON m.match_id = s.match_id AND m.player_id = s.player_id AND m.season_id = s.season_id

WHERE m.season_id = prm.season_id
ORDER BY m.player_id, m.match_date;
