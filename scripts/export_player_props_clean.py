"""Export a cleaned player props CSV.
Writes: data_files/player_props_predictions_clean.csv
Prints: row count and date range.
"""
from pathlib import Path
import pandas as pd

p = Path('data_files/player_props_predictions.csv')
out = Path('data_files/player_props_predictions_clean.csv')
if not p.exists():
    print('ERROR: source file not found:', p)
    raise SystemExit(1)

df = pd.read_csv(p, dtype=str)
# parse and normalize dates
if 'game_date' in df.columns:
    df['game_date'] = pd.to_datetime(df['game_date'], utc=True, errors='coerce')
    df['game_date_iso'] = df['game_date'].dt.strftime('%Y-%m-%d')
    df.drop(columns=['game_date'], inplace=True)
    df.rename(columns={'game_date_iso':'game_date'}, inplace=True)

# numeric conversions
num_cols = ['week','line_value','prob_over','prob_under','confidence','opponent_def_rank','avg_L3','avg_L5','avg_L10']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# desired column order
cols_order = ['week','game_date','player_name','display_name','position','team','opponent','is_home',
              'prop_type','line_type','line_value','prob_over','prob_under','recommendation','confidence',
              'opponent_def_rank','avg_L3','avg_L5','avg_L10','trend','weather_adjusted','weather_conditions','injury_note']
cols = [c for c in cols_order if c in df.columns]

# write clean CSV
out.parent.mkdir(parents=True, exist_ok=True)
# ensure UTF-8 and no index
df.to_csv(out, columns=cols, index=False, encoding='utf-8')

# print summary
rows = len(df)
min_date = df['game_date'].min() if 'game_date' in df.columns else None
max_date = df['game_date'].max() if 'game_date' in df.columns else None
print('WROTE', out)
print('rows=', rows)
print('date_min=', min_date)
print('date_max=', max_date)
