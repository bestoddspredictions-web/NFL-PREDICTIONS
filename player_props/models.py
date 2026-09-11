"""
Player Props XGBoost Models
Train baseline models for predicting player prop over/under outcomes.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('data_files')
MODELS_DIR = Path('player_props/models')
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Common prop betting lines - UPDATED TO COMPREHENSIVE TIERED SYSTEM
PROP_LINES = {
    'passing_yards': {
        'elite_qb': 275.0,     # For QBs averaging 280+ yards/game (Stafford, Purdy)
        'star_qb': 250.0,      # For QBs averaging 250-280 yards/game (Burrow, Goff)
        'good_qb': 225.0,      # For QBs averaging 220-250 yards/game (Maye, Daniels)
        'starter': 200.0       # For QBs averaging <220 yards/game
    },
    'passing_tds': {
        'elite_qb': 2.5,       # For QBs averaging 2.0+ tds/game (Stafford, Purdy)
        'star_qb': 1.5,        # For QBs averaging 1.5-2.0 tds/game (Maye, Prescott)
        'good_qb': 1.5,        # For QBs averaging 1.0-1.5 tds/game (Most QBs)
        'starter': 0.5         # For QBs averaging <1.0 tds/game
    },
    'rushing_yards': {
        'elite_rb': 55.0,      # For RBs averaging 80+ yards/game (Henry, Barkley)
        'star_rb': 45.0,       # For RBs averaging 60-80 yards/game (Taylor, Allen)
        'good_rb': 35.0,       # For RBs averaging 40-60 yards/game (Jacobs, Cook)
        'starter': 24.0        # For RBs averaging <40 yards/game
    },
    'receiving_yards': {
        'elite_wr': 75.0,      # For WRs averaging 80+ yards/game (Adams, Diggs)
        'star_wr': 60.0,       # For WRs averaging 60-80 yards/game (Wilson, Lamb)
        'good_wr': 45.0,       # For WRs averaging 40-60 yards/game (Kraft, Waller)
        'starter': 27.0        # For WRs averaging <40 yards/game
    },
    'rushing_tds': {
        'elite_rb': 0.5,       # For RBs averaging 1.0+ tds/game (Taylor, Henry)
        'star_rb': 0.5,        # For RBs averaging 0.7-1.0 tds/game (Allen, Jacobs)
        'good_rb': 0.5,        # For RBs averaging 0.4-0.7 tds/game (Cook, Charbonnet)
        'anytime': 0.5         # For RBs averaging <0.4 tds/game
    },
    'receiving_tds': {
        'elite_wr': 0.5,       # For WRs averaging 1.0+ tds/game (Adams)
        'star_wr': 0.5,        # For WRs averaging 0.7-1.0 tds/game (Wilson, Kraft)
        'good_wr': 0.5,        # For WRs averaging 0.4-0.7 tds/game (Higgins, Goedert)
        'anytime': 0.5         # For WRs averaging <0.4 tds/game
    },
    'receptions': {
        'elite_wr': 7.5,       # Elite WRs: 8+ receptions (Adams, Diggs)
        'star_wr': 5.5,        # Star WRs: 6+ receptions (Wilson, Lamb)
        'good_wr': 4.5,        # Good WRs: 5+ receptions (Kraft, Waller)
        'starter': 3.5         # Role players: 4+ receptions
    }
}

# Training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_GAMES_PLAYED = 5  # Minimum games to have reliable rolling stats

# ============================================================================
# TIER POPULATION BOUNDARIES
#
# These turn the PROP_LINES comments ("For QBs averaging 280+ yards/game")
# into real filters. Each tier's model is trained ONLY on games from players
# whose rolling average (the *_L5 column) falls in that tier's range at the
# time of the game -- e.g. the elite_qb model never sees games from a
# backup QB averaging 180 yards/game, and vice versa.
#
# Boundaries are [min, max) on the relevant *_L5 rolling-average column,
# taken directly from the PROP_LINES comments above. `None` means unbounded
# on that side (e.g. elite has no upper bound, starter has no lower bound).
# ============================================================================
TIER_BOUNDARIES = {
    'passing_yards': {
        'rolling_col': 'passing_yards_L5',
        'tiers': {
            'elite_qb': (280.0, None),
            'star_qb': (250.0, 280.0),
            'good_qb': (220.0, 250.0),
            'starter': (None, 220.0),
        },
    },
    'passing_tds': {
        'rolling_col': 'pass_tds_L5',
        'tiers': {
            'elite_qb': (2.0, None),
            'star_qb': (1.5, 2.0),
            'good_qb': (1.0, 1.5),
            'starter': (None, 1.0),
        },
    },
    'rushing_yards': {
        'rolling_col': 'rushing_yards_L5',
        'tiers': {
            'elite_rb': (80.0, None),
            'star_rb': (60.0, 80.0),
            'good_rb': (40.0, 60.0),
            'starter': (None, 40.0),
        },
    },
    'rushing_tds': {
        'rolling_col': 'rush_tds_L5',
        'tiers': {
            'elite_rb': (1.0, None),
            'star_rb': (0.7, 1.0),
            'good_rb': (0.4, 0.7),
            'anytime': (None, 0.4),
        },
    },
    'receiving_yards': {
        'rolling_col': 'receiving_yards_L5',
        'tiers': {
            'elite_wr': (80.0, None),
            'star_wr': (60.0, 80.0),
            'good_wr': (40.0, 60.0),
            'starter': (None, 40.0),
        },
    },
    'receiving_tds': {
        'rolling_col': 'rec_tds_L5',
        'tiers': {
            'elite_wr': (1.0, None),
            'star_wr': (0.7, 1.0),
            'good_wr': (0.4, 0.7),
            'anytime': (None, 0.4),
        },
    },
    'receptions': {
        'rolling_col': 'receptions_L5',
        'tiers': {
            'elite_wr': (8.0, None),
            'star_wr': (6.0, 8.0),
            'good_wr': (5.0, 6.0),
            'starter': (None, 5.0),
        },
    },
}


def filter_to_tier(df, stat_key, tier_name):
    """
    Filter a stats DataFrame down to only the games belonging to a tier's
    player population, based on their rolling average AT THE TIME of that
    game (the *_L5 column), not their season-long or career average.

    This is what actually makes an "elite_qb" model different from a
    "starter" model: each one only ever sees games from players who were
    performing at that level going into the game. Without this filter, all
    four tier models train on the exact same population and only differ by
    which threshold they're asked to predict against.

    Args:
        df: Stats DataFrame (must include the tier's rolling_col)
        stat_key: One of the top-level PROP_LINES/TIER_BOUNDARIES keys,
            e.g. 'passing_yards', 'rushing_tds'
        tier_name: e.g. 'elite_qb', 'starter'

    Returns:
        Filtered DataFrame (rows outside the tier's rolling-average range
        are dropped). Returns the input unchanged if boundaries aren't
        defined for this stat_key/tier_name (fail-open, with a warning).
    """
    config = TIER_BOUNDARIES.get(stat_key)
    if config is None:
        print(f"   Warning: no tier boundaries defined for '{stat_key}'; using full population")
        return df

    rolling_col = config['rolling_col']
    bounds = config['tiers'].get(tier_name)
    if bounds is None or rolling_col not in df.columns:
        print(f"   Warning: no tier boundary for '{stat_key}/{tier_name}'; using full population")
        return df

    lower, upper = bounds
    mask = pd.Series(True, index=df.index)
    if lower is not None:
        mask &= df[rolling_col] >= lower
    if upper is not None:
        mask &= df[rolling_col] < upper

    filtered = df[mask].copy()
    print(f"   Tier '{tier_name}': {len(filtered):,} / {len(df):,} rows "
          f"({rolling_col} in [{lower}, {upper}))")
    return filtered

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_player_stats(stat_type='passing'):
    """Load aggregated player stats."""
    file_map = {
        'passing': 'player_passing_stats.csv',
        'rushing': 'player_rushing_stats.csv',
        'receiving': 'player_receiving_stats.csv'
    }
    
    file_path = DATA_DIR / file_map[stat_type]
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print(f"   Run 'python player_props/aggregators.py' first")
        return None
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df):,} {stat_type} records")
    return df


def create_prop_targets(df, stat_col, line_values):
    """
    Create binary targets for different prop lines.
    
    Args:
        df: DataFrame with player stats
        stat_col: Column name for the stat (e.g., 'passing_yards')
        line_values: Dict of prop lines to create targets for
        
    Returns:
        DataFrame with added target columns
    """
    for line_name, line_value in line_values.items():
        target_col = f'over_{line_name}'
        df[target_col] = (df[stat_col] > line_value).astype(int)
    
    return df


def prepare_training_features(df, stat_type='passing'):
    """
    Prepare features for model training.
    
    Features include:
    - Rolling averages (L3, L5, L10)
    - Rolling std dev for consistency
    - Team strength indicators
    - Home/away
    - Recent trends
    - Matchup features (opponent defense rank, home/away, days rest)
    """
    # Core stat column mapping
    stat_col_map = {
        'passing': 'passing_yards',
        'rushing': 'rushing_yards',
        'receiving': 'receiving_yards'
    }
    
    stat_col = stat_col_map[stat_type]
    
    # Filter to only games with valid rolling averages
    # (need at least 3 games for L3 average)
    rolling_col = f'{stat_col}_L3'
    df = df.dropna(subset=[rolling_col]).copy()
    
    # Base features (rolling averages already calculated by aggregator)
    feature_cols = [
        f'{stat_col}_L3',
        f'{stat_col}_L5', 
        f'{stat_col}_L10',
        f'{stat_col}_std_L5'   # volatility: how much this player's stat swings week to week
    ]
    
    # Add position-specific features
    if stat_type == 'passing':
        extra_cols = [
            'pass_tds_L3', 'pass_tds_L5',
            'completions_L3', 'completions_L5',
            'attempts_L3', 'attempts_L5'
        ]
    elif stat_type == 'rushing':
        extra_cols = [
            'rush_tds_L3', 'rush_tds_L5',
            'rush_attempts_L3', 'rush_attempts_L5'
        ]
    elif stat_type == 'receiving':
        extra_cols = [
            'rec_tds_L3', 'rec_tds_L5',
            'receptions_L3', 'receptions_L5',
            'targets_L3', 'targets_L5',
            'target_share_L3', 'target_share_L5'
        ]
    
    # Only add features that exist in the dataframe
    extra_cols = [c for c in extra_cols if c in df.columns]
    feature_cols.extend(extra_cols)
    
    # Add matchup and situational features
    matchup_cols = [
        'opponent_def_rank',
        'is_home', 
        'days_rest'
    ]
    
    # Only add matchup features that exist
    matchup_cols = [c for c in matchup_cols if c in df.columns]
    feature_cols.extend(matchup_cols)
    
    # Filter to only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"📊 Using {len(feature_cols)} features: {feature_cols[:5]}...")
    
    return df, feature_cols


def prepare_td_training_features(df, stat_type='passing'):
    """
    Prepare features for TD model training.
    
    TD models use different features than yards models:
    - Focus on TD-related stats and volume metrics
    - Exclude yards (to avoid data leakage for TD predictions)
    - Include matchup features for better predictions
    """
    # Filter to only games with valid rolling TD averages
    td_col = 'pass_tds' if stat_type == 'passing' else ('rush_tds' if stat_type == 'rushing' else 'rec_tds')
    rolling_td_col = f'{td_col}_L3'
    df = df.dropna(subset=[rolling_td_col]).copy()
    
    # TD-specific features (no yards to avoid leakage)
    if stat_type == 'passing':
        feature_cols = [
            'pass_tds_L3', 'pass_tds_L5', 'pass_tds_L10', 'pass_tds_std_L5',
            'completions_L3', 'completions_L5', 'completions_L10',
            'attempts_L3', 'attempts_L5', 'attempts_L10'
        ]
    elif stat_type == 'rushing':
        feature_cols = [
            'rush_tds_L3', 'rush_tds_L5', 'rush_tds_L10', 'rush_tds_std_L5',
            'rush_attempts_L3', 'rush_attempts_L5', 'rush_attempts_L10'
        ]
    elif stat_type == 'receiving':
        feature_cols = [
            'rec_tds_L3', 'rec_tds_L5', 'rec_tds_L10', 'rec_tds_std_L5',
            'receptions_L3', 'receptions_L5', 'receptions_L10',
            'targets_L3', 'targets_L5', 'targets_L10'
        ]
    
    # Add matchup and situational features
    matchup_cols = [
        'opponent_def_rank',
        'is_home', 
        'days_rest'
    ]
    
    # Only add matchup features that exist
    matchup_cols = [c for c in matchup_cols if c in df.columns]
    feature_cols.extend(matchup_cols)
    
    # Filter to only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"🏈 Using {len(feature_cols)} TD features: {feature_cols[:5]}...")
    
    return df, feature_cols


def prepare_receptions_training_features(df):
    """
    Prepare features specifically for receptions model training.
    
    Receptions models use reception-focused features:
    - Rolling receptions averages (L3, L5, L10)
    - Rolling targets and TD features
    - Matchup features (opponent defense rank, home/away, days rest)
    """
    # Filter to only games with valid rolling receptions averages
    df = df.dropna(subset=['receptions_L3']).copy()
    
    # Core receptions features
    feature_cols = [
        'receptions_L3',
        'receptions_L5', 
        'receptions_L10',
        'receptions_std_L5',   # volatility: how much this player's receptions swing week to week
        'rec_tds_L3', 
        'rec_tds_L5',
        'targets_L3', 
        'targets_L5',
        'target_share_L3',
        'target_share_L5'
    ]
    
    # Add matchup and situational features
    matchup_cols = [
        'opponent_def_rank',
        'is_home', 
        'days_rest'
    ]
    
    # Only add matchup features that exist
    matchup_cols = [c for c in matchup_cols if c in df.columns]
    feature_cols.extend(matchup_cols)
    
    # Filter to only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"📊 Using {len(feature_cols)} receptions features: {feature_cols}")
    
    return df, feature_cols


def walk_forward_evaluate(df, features, target_col, model_name, n_splits=5, min_test_size=15):
    """
    Evaluate a model across multiple chronological cut points instead of one,
    and report mean +/- std ROC-AUC. This exists because a single
    train/test split on a few hundred rows (typical for "elite" tiers)
    produces a number that's mostly noise -- two runs of the same idea can
    differ by 0.05-0.10 ROC-AUC just from which games happened to land in
    the test slice. Averaging several splits, each one still strictly
    chronological (train on earlier games, test on a later slice -- never
    shuffled), tells us whether an effect is real or within the noise band.

    This does NOT replace train_prop_model: that function trains the one
    final model that actually gets saved and used for predictions. This
    function is a diagnostic -- run it when deciding whether a feature or
    change is a genuine improvement before committing to it.

    Args:
        df: DataFrame with features, target, season, week
        features: List of feature column names
        target_col: Target column name
        model_name: Label for print output only
        n_splits: Number of expanding-window folds to evaluate
        min_test_size: Skip a fold if its test slice would be smaller than this

    Returns:
        dict with 'mean_auc', 'std_auc', 'fold_aucs' (list), 'n_folds_used'
        or None if there isn't enough data to run any folds.
    """
    keep_cols = features + [target_col, 'season', 'week']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_clean = df[keep_cols].dropna(subset=[c for c in features + [target_col] if c in keep_cols])

    if 'season' not in df_clean.columns or 'week' not in df_clean.columns:
        print(f"   Warning: {model_name} has no season/week; walk-forward evaluation skipped")
        return None

    df_clean = df_clean.sort_values(['season', 'week']).reset_index(drop=True)
    n = len(df_clean)
    if n < 200:
        print(f"   Warning: {model_name} has only {n} rows; walk-forward evaluation skipped")
        return None

    # Expanding window: fold i trains on the first cut_i rows, tests on the
    # next slice. Cuts are spaced so the final fold's test slice ends at the
    # very end of the data (matching what train_prop_model's single split
    # already evaluates), and earlier folds test on progressively earlier
    # slices -- giving genuinely different train/test boundaries per fold
    # rather than re-testing the same slice with tiny train-set changes.
    test_frac = TEST_SIZE / n_splits
    fold_aucs = []
    for i in range(n_splits):
        train_end = int(n * (1 - TEST_SIZE + i * test_frac))
        test_end = int(n * (1 - TEST_SIZE + (i + 1) * test_frac))
        if train_end < 50 or (test_end - train_end) < min_test_size:
            continue

        train_df = df_clean.iloc[:train_end]
        test_df = df_clean.iloc[train_end:test_end]

        y_train = train_df[target_col]
        y_test = test_df[target_col]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        if min(y_train.value_counts()) < 5:
            continue

        X_train, X_test = train_df[features], test_df[features]
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        fold_aucs.append(float(roc_auc_score(y_test, proba)))

    if not fold_aucs:
        print(f"   Warning: {model_name} had no usable folds for walk-forward evaluation")
        return None

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    print(f"   {model_name}: walk-forward ROC-AUC = {mean_auc:.3f} +/- {std_auc:.3f} "
          f"over {len(fold_aucs)} folds {[round(a, 3) for a in fold_aucs]}")

    return {
        'model_name': model_name,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'fold_aucs': fold_aucs,
        'n_folds_used': len(fold_aucs),
    }


def train_prop_model(df, features, target_col, model_name):
    """
    Train an XGBoost + LightGBM soft-voting ensemble for a specific prop line.
    Both models are saved; inference averages their predicted probabilities.

    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target_col: Target column name
        model_name: Name for saving model (e.g. 'passing_yards_elite_qb')

    Returns:
        Trained XGB model (primary), metrics dict
    """
    # Carry season/week through so we can do a time-based split, even though
    # they aren't model features themselves.
    keep_cols = features + [target_col]
    if 'season' in df.columns:
        keep_cols = keep_cols + ['season']
    if 'week' in df.columns:
        keep_cols = keep_cols + ['week']

    # Remove rows with missing features or target
    df_clean = df[keep_cols].dropna(subset=features + [target_col])

    if len(df_clean) < 100:
        print(f"Warning: Not enough data for {target_col}: {len(df_clean)} rows")
        return None, None

    y_all = df_clean[target_col]

    # Check class balance
    class_dist = y_all.value_counts()
    print(f"\nTarget distribution for {target_col}:")
    print(f"   Over (1): {class_dist.get(1, 0):,} ({class_dist.get(1, 0)/len(y_all)*100:.1f}%)")
    print(f"   Under (0): {class_dist.get(0, 0):,} ({class_dist.get(0, 0)/len(y_all)*100:.1f}%)")

    # Skip if too imbalanced (less than 10% of either class)
    if min(class_dist.get(0, 0), class_dist.get(1, 0)) / len(y_all) < 0.1:
        print(f"Warning: Skipping {target_col}: too imbalanced")
        return None, None

    # ------------------------------------------------------------------
    # Time-based (season/week-forward) split instead of a random split.
    #
    # A random split lets a player's Week 10 game train the model that
    # predicts their Week 3 game, which leaks future performance into the
    # training set and inflates test metrics. Sorting chronologically and
    # taking the most recent slice as the held-out test set means the model
    # is always evaluated on data that comes strictly after everything it
    # was trained on -- the same situation it faces in real deployment.
    # ------------------------------------------------------------------
    if 'season' in df_clean.columns and 'week' in df_clean.columns:
        df_clean = df_clean.sort_values(['season', 'week'])
        split_idx = int(len(df_clean) * (1 - TEST_SIZE))
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]

        X_train, y_train = train_df[features], train_df[target_col]
        X_test, y_test = test_df[features], test_df[target_col]

        print(
            f"   Time split: train seasons {train_df['season'].min()}-{train_df['season'].max()} "
            f"-> test seasons {test_df['season'].min()}-{test_df['season'].max()} "
            f"(test starts week {test_df['week'].iloc[0]} of {test_df['season'].iloc[0]})"
        )
    else:
        # Fallback: season/week not available on this df for some reason.
        print("   Warning: season/week not found; falling back to random split")
        X = df_clean[features]
        y = df_clean[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # ------------------------------------------------------------------
    # XGBoost model
    # ------------------------------------------------------------------
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # LightGBM model (optional — falls back to XGB-only if unavailable)
    # ------------------------------------------------------------------
    ensemble_proba = xgb_proba
    if LGBM_AVAILABLE:
        lgbm_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            verbosity=-1
        )
        lgbm_model.fit(X_train, y_train)
        lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
        # Soft-vote: equal weight average
        ensemble_proba = (xgb_proba + lgbm_proba) / 2.0
        # Persist LGBM model alongside XGB
        lgbm_path = MODELS_DIR / f'{model_name}_lgbm.txt'
        lgbm_model.booster_.save_model(str(lgbm_path))

    y_pred = (ensemble_proba >= 0.5).astype(int)

    # A chronological split can leave y_test with only one class present
    # (e.g. every recent game happened to go "over"). roc_auc_score requires
    # both classes, so guard against it rather than crashing mid-run.
    if y_test.nunique() < 2:
        print(f"   Warning: y_test has only one class for {target_col}; ROC-AUC undefined")
        roc_auc = float('nan')
    else:
        roc_auc = float(roc_auc_score(y_test, ensemble_proba))

    # Metrics computed on ensemble output
    metrics = {
        'model_name': model_name,
        'target': target_col,
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': roc_auc,
        'ensemble': LGBM_AVAILABLE
    }

    print(f"\n{model_name} Results (ensemble={LGBM_AVAILABLE}):")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1: {metrics['f1']:.3f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")

    # Persist XGB model (always present)
    model_path = MODELS_DIR / f'{model_name}.json'
    xgb_model.save_model(model_path)
    print(f"Saved XGB model to {model_path}")

    return xgb_model, metrics


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_all_models():
    """Train all player prop models with comprehensive tiered system."""
    print("=" * 70)
    print("🏈 NFL Player Props Model Training - Comprehensive Tiered System")
    print("=" * 70)

    all_metrics = []
    all_wf_results = []

    # ========================================================================
    # 1. PASSING YARDS MODELS
    # ========================================================================
    print("\n\n📊 PASSING YARDS MODELS")
    print("-" * 70)

    passing_df = load_player_stats('passing')
    if passing_df is not None:
        # Create targets for all QB tiers
        passing_df = create_prop_targets(
            passing_df,
            'passing_yards',
            PROP_LINES['passing_yards']
        )

        passing_df, pass_features = prepare_training_features(passing_df, 'passing')

        # Train models for each tier
        for line_type in PROP_LINES['passing_yards'].keys():
            target_col = f'over_{line_type}'
            model_name = f'passing_yards_{line_type}'

            tier_df = filter_to_tier(passing_df, 'passing_yards', line_type)
            wf_result = walk_forward_evaluate(tier_df, pass_features, target_col, model_name)
            if wf_result:
                all_wf_results.append(wf_result)
            model, metrics = train_prop_model(
                tier_df, pass_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)

    # ========================================================================
    # 2. PASSING TDS MODELS
    # ========================================================================
    print("\n\n🏈 PASSING TDS MODELS")
    print("-" * 70)

    if passing_df is not None:
        # Create targets for all TD tiers
        passing_df = create_prop_targets(
            passing_df,
            'pass_tds',
            PROP_LINES['passing_tds']
        )

        # Prepare TD-specific features (no yards leakage)
        passing_df, pass_td_features = prepare_td_training_features(passing_df, 'passing')

        # Train models for each TD tier
        for line_type in PROP_LINES['passing_tds'].keys():
            target_col = f'over_{line_type}'
            model_name = f'passing_tds_{line_type}'

            tier_df = filter_to_tier(passing_df, 'passing_tds', line_type)
            wf_result = walk_forward_evaluate(tier_df, pass_td_features, target_col, model_name)
            if wf_result:
                all_wf_results.append(wf_result)
            model, metrics = train_prop_model(
                tier_df, pass_td_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)

    # ========================================================================
    # 3. RUSHING YARDS MODELS
    # ========================================================================
    print("\n\n🏃 RUSHING YARDS MODELS")
    print("-" * 70)

    rushing_df = load_player_stats('rushing')
    if rushing_df is not None:
        rushing_df = create_prop_targets(
            rushing_df,
            'rushing_yards',
            PROP_LINES['rushing_yards']
        )

        rushing_df, rush_features = prepare_training_features(rushing_df, 'rushing')

        for line_type in PROP_LINES['rushing_yards'].keys():
            target_col = f'over_{line_type}'
            model_name = f'rushing_yards_{line_type}'

            tier_df = filter_to_tier(rushing_df, 'rushing_yards', line_type)
            wf_result = walk_forward_evaluate(tier_df, rush_features, target_col, model_name)
            if wf_result:
                all_wf_results.append(wf_result)
            model, metrics = train_prop_model(
                tier_df, rush_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)

    # ========================================================================
    # 4. RUSHING TDS MODELS
    # ========================================================================
    print("\n\n🏃 RUSHING TDS MODELS")
    print("-" * 70)

    if rushing_df is not None:
        # Create targets for all TD tiers
        rushing_df = create_prop_targets(
            rushing_df,
            'rush_tds',
            PROP_LINES['rushing_tds']
        )

        # Prepare TD-specific features (no yards leakage)
        rushing_df, rush_td_features = prepare_td_training_features(rushing_df, 'rushing')

        # Train models for each TD tier
        for line_type in PROP_LINES['rushing_tds'].keys():
            target_col = f'over_{line_type}'
            model_name = f'rushing_tds_{line_type}'

            tier_df = filter_to_tier(rushing_df, 'rushing_tds', line_type)
            wf_result = walk_forward_evaluate(tier_df, rush_td_features, target_col, model_name)
            if wf_result:
                all_wf_results.append(wf_result)
            model, metrics = train_prop_model(
                tier_df, rush_td_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)

    # ========================================================================
    # 5. RECEIVING YARDS MODELS
    # ========================================================================
    print("\n\n🤲 RECEIVING YARDS MODELS")
    print("-" * 70)

    receiving_df = load_player_stats('receiving')
    if receiving_df is not None:
        receiving_df = create_prop_targets(
            receiving_df,
            'receiving_yards',
            PROP_LINES['receiving_yards']
        )

        receiving_df, rec_features = prepare_training_features(receiving_df, 'receiving')

        for line_type in PROP_LINES['receiving_yards'].keys():
            target_col = f'over_{line_type}'
            model_name = f'receiving_yards_{line_type}'

            tier_df = filter_to_tier(receiving_df, 'receiving_yards', line_type)
            wf_result = walk_forward_evaluate(tier_df, rec_features, target_col, model_name)
            if wf_result:
                all_wf_results.append(wf_result)
            model, metrics = train_prop_model(
                tier_df, rec_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)

    # ========================================================================
    # 6. RECEIVING TDS MODELS
    # ========================================================================
    print("\n\n🤲 RECEIVING TDS MODELS")
    print("-" * 70)

    if receiving_df is not None:
        # Create targets for all TD tiers
        receiving_df = create_prop_targets(
            receiving_df,
            'rec_tds',
            PROP_LINES['receiving_tds']
        )

        # Prepare TD-specific features (no yards leakage)
        receiving_df, rec_td_features = prepare_td_training_features(receiving_df, 'receiving')

        # Train models for each TD tier
        for line_type in PROP_LINES['receiving_tds'].keys():
            target_col = f'over_{line_type}'
            model_name = f'receiving_tds_{line_type}'

            tier_df = filter_to_tier(receiving_df, 'receiving_tds', line_type)
            wf_result = walk_forward_evaluate(tier_df, rec_td_features, target_col, model_name)
            if wf_result:
                all_wf_results.append(wf_result)
            model, metrics = train_prop_model(
                tier_df, rec_td_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)

    # ========================================================================
    # 7. RECEPTIONS MODELS
    # ========================================================================
    print("\n\n🤲 RECEPTIONS MODELS")
    print("-" * 70)

    if receiving_df is not None:
        # Create targets for all reception tiers
        receiving_df = create_prop_targets(
            receiving_df,
            'receptions',
            PROP_LINES['receptions']
        )

        # Prepare reception-specific features
        receiving_df, rec_features = prepare_receptions_training_features(receiving_df)

        # Train models for each reception tier
        for line_type in PROP_LINES['receptions'].keys():
            target_col = f'over_{line_type}'
            model_name = f'receptions_{line_type}'

            tier_df = filter_to_tier(receiving_df, 'receptions', line_type)
            wf_result = walk_forward_evaluate(tier_df, rec_features, target_col, model_name)
            if wf_result:
                all_wf_results.append(wf_result)
            model, metrics = train_prop_model(
                tier_df, rec_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)

    # ========================================================================
    # SAVE SUMMARY METRICS
    # ========================================================================
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = MODELS_DIR / 'model_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n\n💾 Saved metrics summary to {metrics_path}")

        # Save JSON version too
        metrics_json_path = MODELS_DIR / 'model_metrics.json'
        with open(metrics_json_path, 'w') as f:
            json.dump({
                'trained_at': datetime.now().isoformat(),
                'models': all_metrics
            }, f, indent=2)

        # Print summary table
        print("\n" + "=" * 70)
        print("📊 MODEL PERFORMANCE SUMMARY (single chronological split)")
        print("=" * 70)
        print(metrics_df[['model_name', 'accuracy', 'f1', 'roc_auc']].to_string(index=False))

    # ========================================================================
    # SAVE WALK-FORWARD (REPEATED) EVALUATION RESULTS
    #
    # This is a separate, more reliable measure than the single-split
    # roc_auc above. A single split on a few hundred rows produces a number
    # that's mostly noise -- these mean/std values are computed across
    # several chronological folds and tell you whether a difference between
    # two versions of a model is real or within the noise band.
    # ========================================================================
    if all_wf_results:
        wf_df = pd.DataFrame([
            {
                'model_name': r['model_name'],
                'mean_auc': r['mean_auc'],
                'std_auc': r['std_auc'],
                'n_folds': r['n_folds_used'],
            }
            for r in all_wf_results
        ])
        wf_path = MODELS_DIR / 'walk_forward_metrics.csv'
        wf_df.to_csv(wf_path, index=False)
        print(f"\n💾 Saved walk-forward evaluation summary to {wf_path}")

        wf_json_path = MODELS_DIR / 'walk_forward_metrics.json'
        with open(wf_json_path, 'w') as f:
            json.dump({
                'evaluated_at': datetime.now().isoformat(),
                'note': (
                    'mean_auc/std_auc are computed across multiple chronological '
                    'folds per model. Treat differences between two runs as noise '
                    'unless they exceed roughly one std_auc.'
                ),
                'models': all_wf_results,
            }, f, indent=2)

        print("\n" + "=" * 70)
        print("📊 WALK-FORWARD SUMMARY (mean +/- std across folds)")
        print("=" * 70)
        print(wf_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print(f"📁 Models saved to: {MODELS_DIR}")
    print("\nNext steps:")
    print("  1. Review metrics in player_props/models/model_metrics.csv")
    print("  2. Review noise-aware metrics in player_props/models/walk_forward_metrics.csv")
    print("  3. Create prediction pipeline for upcoming games")
    print("  4. Integrate with Player Props UI page")
    print("=" * 70)


if __name__ == '__main__':
    train_all_models()