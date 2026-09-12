"""Quick player-props validator: generates lightweight predictions using Laplace-smoothed historical hit rates.
This avoids web scraping and heavy model inference so it runs quickly and verifies schedule usage.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent))
from predict import PROP_LINES, POSITION_MAP, load_schedule, load_player_stats, get_recent_starters, get_player_position

DATA_DIR = Path(__file__).parent.parent / 'data_files'
OUT = DATA_DIR / 'player_props_predictions_quick.csv'


def laplace_rate(over_count, total_games, alpha=1, beta=1):
    # (over_count + alpha) / (total_games + alpha + beta)
    return (over_count + alpha) / (total_games + alpha + beta)


def generate_quick():
    schedule = load_schedule()
    if schedule is None or schedule.empty:
        print('No schedule loaded')
        return
    stats = load_player_stats()
    rows = []
    for _, game in schedule.iterrows():
        week = game['week']
        date = pd.to_datetime(game['game_date']) if 'game_date' in game else pd.to_datetime(game['date'])
        home = game['home_team']
        away = game['away_team']
        for team in [home, away]:
            # pick recent starters
            for stat_type, df in stats.items():
                if df is None:
                    continue
                starters = get_recent_starters(df, team, n_games=3)
                for player in starters[:3]:
                    pos, display = get_player_position(player)
                    # choose props based on position
                    if pos is None:
                        continue
                    props = POSITION_MAP.get(pos, [])
                    for prop in props:
                        line_type = 'starter'
                        line = PROP_LINES.get(prop, {}).get(line_type, 0)
                        # historical: count games where player's stat >= line
                        player_stats = df[(df['player_name'] == player) & (df['team'] == team)]
                        total = len(player_stats)
                        if total == 0:
                            continue
                        stat_col = None
                        if prop in ['passing_yards']:
                            stat_col = 'passing_yards'
                        elif prop in ['rushing_yards']:
                            stat_col = 'rushing_yards'
                        elif prop in ['receiving_yards']:
                            stat_col = 'receiving_yards'
                        elif prop in ['receptions']:
                            stat_col = 'receptions'
                        elif prop in ['passing_tds']:
                            stat_col = 'pass_tds'
                        elif prop in ['rushing_tds']:
                            stat_col = 'rush_tds'
                        elif prop in ['receiving_tds']:
                            stat_col = 'rec_tds'
                        else:
                            continue
                        over_count = int((player_stats[stat_col] >= line).sum()) if stat_col in player_stats.columns else 0
                        prob_over = laplace_rate(over_count, total)
                        prob_under = 1 - prob_over
                        rec = 'OVER' if prob_over >= 0.5 else 'UNDER'
                        rows.append({
                            'week': week,
                            'game_date': date.strftime('%Y-%m-%d'),
                            'player_name': player,
                            'display_name': display or player,
                            'position': pos,
                            'team': team,
                            'opponent': home if team != home else away,
                            'is_home': team == home,
                            'prop_type': prop,
                            'line_type': line_type,
                            'line_value': line,
                            'prob_over': prob_over,
                            'prob_under': prob_under,
                            'recommendation': rec,
                            'confidence': max(prob_over, prob_under)
                        })
    if rows:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"Wrote {len(rows)} quick predictions to {OUT}")
    else:
        print('No predictions generated')

if __name__ == '__main__':
    generate_quick()
