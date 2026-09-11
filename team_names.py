"""
NFL team abbreviation -> full name mapping.

Extracted from predictions.py (where it existed as an inline dict,
duplicated four times). Centralised here so pages/2_Player_Props.py
can import it without duplicating the mapping a third time.

Usage:
    from team_names import NFL_TEAM_NAMES, expand_team

    full = expand_team('KC')       # 'Kansas City Chiefs'
    full = expand_team('UNKNOWN')  # 'UNKNOWN'  (fail-open, no crash)
"""

# Canonical abbrev -> full name.
# 'LA' and 'LAR' both map to Rams (nflverse uses LAR, other sources use LA).
NFL_TEAM_NAMES: dict = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB':  'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC':  'Kansas City Chiefs',
    'LA':  'Los Angeles Rams',
    'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams',
    'LV':  'Las Vegas Raiders',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE':  'New England Patriots',
    'NO':  'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SF':  'San Francisco 49ers',
    'SEA': 'Seattle Seahawks',
    'TB':  'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders',
}

# Tracks any abbrevs seen in data that are not in the dict above.
_unmapped: set = set()


def expand_team(abbrev: str, warn: bool = True) -> str:
    """Return the full team name for abbrev, or the raw abbrev if not found.

    Args:
        abbrev: Team abbreviation string, e.g. 'KC', 'SEA'.
        warn:   If True (default), record unmapped abbrevs in _unmapped.

    Returns:
        Full team name string, or the original abbrev if unmapped.
    """
    if not abbrev or not isinstance(abbrev, str):
        return str(abbrev)
    result = NFL_TEAM_NAMES.get(abbrev.strip().upper())
    if result is None:
        if warn:
            _unmapped.add(abbrev)
        return abbrev  # fail-open: show raw abbrev, never crash
    return result


def get_unmapped() -> frozenset:
    """Return the set of abbrevs seen that are not in NFL_TEAM_NAMES."""
    return frozenset(_unmapped)
