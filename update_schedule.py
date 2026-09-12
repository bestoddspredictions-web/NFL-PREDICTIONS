import nfl_data_py as nfl
import pandas as pd
from os import path
from datetime import datetime

DATA_DIR = 'data_files/'
# Determine season year dynamically based on today's date:
# - If month is Sep(9)-Dec(12): season_year = current year
# - If month is Jan(1)-Aug(8): season_year = current year - 1
# This follows the convention that the season is named after the year it starts.
today = datetime.utcnow().date()
if today.month >= 9:
	season_year = today.year
else:
	season_year = today.year - 1

# Fetch schedule for current season
print(f"Fetching NFL schedule for {season_year} (determined from today's date: {today})...")
schedule = nfl.import_schedules([season_year])
print("Columns in schedule:", schedule.columns.tolist())

# Select and format columns for the schedule file
# Assuming columns: adjust based on actual
stadium_col = 'stadium' if 'stadium' in schedule.columns else 'venue'

schedule_df = schedule[['week', 'gameday', 'home_team', 'away_team', stadium_col]].copy()
schedule_df['date'] = pd.to_datetime(schedule_df['gameday']).dt.strftime('%Y-%m-%d')
schedule_df['venue'] = schedule_df[stadium_col]
schedule_df['status'] = 'REG'
schedule_df = schedule_df[['week', 'date', 'home_team', 'away_team', 'venue', 'status']]

# Save to CSV
out_path = f"nfl_schedule_{season_year}.csv"
full_path = path.join(DATA_DIR, out_path)
schedule_df.to_csv(full_path, index=False)
print(f"Saved updated schedule to {full_path}")
print(f"Total games: {len(schedule_df)}")