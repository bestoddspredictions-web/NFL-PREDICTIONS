from pathlib import Path
import pandas as pd

p = Path('data_files/player_props_predictions.csv')
if not p.exists():
    print('MISSING')
    raise SystemExit(1)

df = pd.read_csv(p)
# parse game_date column safely
if 'game_date' in df.columns:
    try:
        df['game_date'] = pd.to_datetime(df['game_date'], utc=True)
    except Exception:
        # fallback: remove timezone text then parse
        df['game_date'] = pd.to_datetime(df['game_date'].astype(str).str.replace('\+00:00',''), errors='coerce')

print('rows:', len(df))
if 'game_date' in df.columns:
    print('min:', df['game_date'].min())
    print('max:', df['game_date'].max())
    uniq = sorted(df['game_date'].dropna().unique())
    print('sample dates (first 5):')
    for d in uniq[:5]:
        print(' ', d)
else:
    print('no game_date column')
