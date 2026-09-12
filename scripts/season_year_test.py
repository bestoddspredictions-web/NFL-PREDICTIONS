from datetime import date, datetime

# Copied logic from update_schedule.py

def compute_season_year(d: date) -> int:
    # use same condition as update_schedule.py (month >= 9 => season_year = year)
    if d.month >= 9:
        season_year = d.year
    else:
        season_year = d.year - 1
    return season_year


test_dates = [
    date(2026, 2, 10),   # February 2026
    date(2026, 7, 15),   # July 2026
    date(2026, 8, 20),   # August 2026
    date(2026, 9, 11),   # today
]

for d in test_dates:
    sy = compute_season_year(d)
    print(f"{d} -> season_year = {sy}")
