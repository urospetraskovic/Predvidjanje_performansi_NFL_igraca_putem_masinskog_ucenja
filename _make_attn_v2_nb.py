"""Builder for WR_RNN_Attention_PlayerEmbed_v2.ipynb.

Relaxed-regularization re-run of the attention+player-embedding experiment:
  dropout    0.25 -> 0.15
  rnn_dropout 0.15 -> 0.10
  noise      0.05 -> 0.00  (GaussianNoise omitted)
  wd         5e-4 -> 1e-4
  lr         5e-4 -> 2e-4
  patience   25   -> 50
Diagnostic cell: if best_epoch < 10 for all 4 configs, declare architecture problem.
"""
import json

nb = {"cells": [], "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5}

def md(s):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})

def code(s):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": s.splitlines(keepends=True)})

md("""# WR RNN - Attention + Player Embeddings **v2 (Relaxed Regularization)**

U originalnom notebook-u (`WR_RNN_Attention_PlayerEmbed`) sva 4 modela su zavrsila sa `best_epoch = 3`.
To ukazuje da je regularizacija bila **prejaka za ovu velicinu dataseta** - model nije imao priliku
da nauci korisne reprezentacije pre nego sto ga je EarlyStopping zakucao.

**Promene u odnosu na v1:**
| Parametar    | v1     | v2     |
|--------------|--------|--------|
| dropout      | 0.25   | 0.15   |
| rnn_dropout  | 0.15   | 0.10   |
| noise        | 0.05   | 0.00 (izbaceno) |
| weight_decay | 5e-4   | 1e-4   |
| lr           | 5e-4   | 2e-4   |
| patience     | 25     | 50     |
| epochs cap   | 500    | 500    |

Sve ostalo (arhitektura, podaci, seq_len=6, sw_strength=0.7) je **identicno** radi cistog A/B poredjenja.

**Dijagnostika:** Ako i posle ovoga sva 4 modela ostanu sa `best_epoch < 10`, zakljucak je da je
sama arhitektura (attention + player embeddings) nepodesna za ovaj skup podataka, a ne hiperparametri.
""")

md("---\n## 1. Imports")
code("""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os, json, time, random

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, losses, Model, Input, regularizers
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Bidirectional,
    LayerNormalization,
    Masking, Concatenate, Embedding, Flatten
)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['figure.figsize'] = (14, 5)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

print(f'TensorFlow {tf.__version__}')
""")

md("---\n## 2. Data Loading & Feature Engineering\n\nIdenticno v1 notebook-u.")
code("""df = pd.read_csv('../data/fully combined/wr_all_weeks.csv')
df['week'] = df['game_id'].str.split('_').str[1].astype(int)
df = df.sort_values(['receiver_player_id', 'season', 'week']).reset_index(drop=True)

df['player_team_inferred'] = np.where(
    (df['home_team'] == df['defteam']) & (df['away_team'] != df['defteam']),
    df['away_team'],
    np.where(
        (df['away_team'] == df['defteam']) & (df['home_team'] != df['defteam']),
        df['home_team'], np.nan
    )
)
prev_team = df.groupby('receiver_player_id')['player_team_inferred'].shift(1)
df['team_changed'] = (
    df['player_team_inferred'].notna() & prev_team.notna()
    & (df['player_team_inferred'] != prev_team)
).astype(int)

prev_season = df.groupby('receiver_player_id')['season'].shift(1)
df['is_new_season'] = (prev_season.notna() & (df['season'] != prev_season)).astype(int)

df['season_week_abs'] = (df['season'] - 2015) * 22 + df['week']
df['weeks_since_last_game'] = (
    df.groupby('receiver_player_id')['season_week_abs']
    .diff().fillna(1).clip(lower=1).astype(float)
)
df.drop(columns=['season_week_abs'], inplace=True)

season_career = (
    df.groupby(['receiver_player_id', 'season'], as_index=False)
    .agg(avg_yards=('receiving_yards', 'mean'),
         avg_target_share=('target_share', 'mean'),
         avg_epa=('epa', 'mean'),
         avg_air_yard_share=('air_yard_share', 'mean'),
         avg_catch_rate=('catch_rate', 'mean'),
         games_played=('game_id', 'count'))
    .sort_values(['receiver_player_id', 'season'])
)
for c in ['avg_yards', 'avg_target_share', 'avg_epa',
          'avg_air_yard_share', 'avg_catch_rate', 'games_played']:
    season_career[f'{c}_last_season'] = (
        season_career.groupby('receiver_player_id')[c].shift(1)
    )
career_cols = [f'{c}_last_season' for c in
               ['avg_yards', 'avg_target_share', 'avg_epa',
                'avg_air_yard_share', 'avg_catch_rate', 'games_played']]
season_career = season_career[['receiver_player_id', 'season'] + career_cols]
df = df.merge(season_career, on=['receiver_player_id', 'season'], how='left')
df[career_cols] = df[career_cols].fillna(0)

temporal_base = ['receiving_yards', 'targets', 'receptions', 'epa', 'target_share', 'air_yard_share']
df = df.sort_values(['receiver_player_id', 'season', 'week']).reset_index(drop=True)
g = df.groupby('receiver_player_id', group_keys=False)
for col in temporal_base:
    df[f'{col}_lag1'] = g[col].shift(1).fillna(0)
    df[f'{col}_roll3'] = g[col].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)

lag_cols = [f'{c}_lag1' for c in temporal_base]
roll_cols = [f'{c}_roll3' for c in temporal_base]
print(f'After engineering: {df.shape}')
""")

md("---\n## 3. Feature Groups")
code("""seq_raw_cols = [
    'receiving_yards', 'targets', 'receptions', 'air_yards', 'yac',
    'first_downs', 'tds', 'epa', 'wpa',
    'catch_rate', 'yards_per_target', 'avg_depth', 'adot', 'success_rate',
    'target_share', 'air_yard_share', 'target_share_std', 'reception_std',
    'team_pass_attempts', 'team_air_yards', 'team_epa',
    'qb_completions', 'qb_attempts', 'qb_comp_pct', 'qb_cpoe',
    'avg_score_diff', 'wp_var',
    'yards_Q1', 'yards_Q2', 'yards_Q3',
    'def_yards_dev', 'def_epa_dev',
    'weeks_since_last_game',
]
seq_feature_cols = seq_raw_cols + lag_cols + roll_cols

static_feature_cols = [
    'pregame_spread', 'pregame_total',
    'surface', 'is_dome', 'temp_f', 'humidity_pct', 'wind_mph',
    'is_rain', 'is_snow', 'is_clear',
    'week', 'team_changed', 'is_new_season', 'weeks_since_last_game',
    'avg_yards_last_season', 'avg_target_share_last_season',
    'avg_epa_last_season', 'avg_air_yard_share_last_season',
    'avg_catch_rate_last_season', 'games_played_last_season',
]

n_seq_features = len(seq_feature_cols)
n_static_features = len(static_feature_cols)
print(f'Sequence channels: {n_seq_features}  Static: {n_static_features}')
""")

md("---\n## 4. Player ID Encoding")
code("""train_seasons = list(range(2015, 2022))
val_seasons = [2022, 2023]
test_seasons = [2024, 2025]

train_players = df.loc[df['season'].isin(train_seasons), 'receiver_player_id'].unique()
player_to_idx = {pid: i + 1 for i, pid in enumerate(sorted(train_players))}
n_players = len(player_to_idx) + 1

df['player_idx'] = df['receiver_player_id'].map(player_to_idx).fillna(0).astype(int)
print(f'Total unique player indices: {n_players} (incl. OOV=0)')
""")

md("---\n## 5. Scaling")
code("""df[seq_feature_cols] = df[seq_feature_cols].fillna(0)
df[static_feature_cols] = df[static_feature_cols].fillna(0)

train_mask_df = df['season'].isin(train_seasons)

seq_scaler = StandardScaler()
seq_scaler.fit(df.loc[train_mask_df, seq_feature_cols])
static_scaler = StandardScaler()
static_scaler.fit(df.loc[train_mask_df, static_feature_cols])

df_scaled = df.copy()
df_scaled[seq_feature_cols] = seq_scaler.transform(df[seq_feature_cols])
df_scaled[static_feature_cols] = static_scaler.transform(df[static_feature_cols])
df_scaled['receiving_yards_orig'] = df['receiving_yards'].values
print('Scaled.')
""")

md("---\n## 6. Build Padded Sequences")
code("""SEQ_LEN = 6

def build_sequences(df_scaled, seq_feats, static_feats, seq_len):
    X_seq, X_static, X_pid, y, seasons = [], [], [], [], []
    for pid, group in df_scaled.groupby('receiver_player_id'):
        group = group.sort_values(['season', 'week'])
        n = len(group)
        if n < 2:
            continue
        seq_vals = group[seq_feats].values
        static_vals = group[static_feats].values
        yards = group['receiving_yards_orig'].values
        season_vals = group['season'].values
        pid_vals = group['player_idx'].values
        for t in range(1, n):
            start = max(0, t - seq_len)
            past = seq_vals[start:t]
            actual_len = past.shape[0]
            if actual_len < seq_len:
                padding = np.zeros((seq_len - actual_len, len(seq_feats)))
                past = np.vstack([padding, past])
            X_seq.append(past)
            X_static.append(static_vals[t])
            X_pid.append(pid_vals[t])
            y.append(yards[t])
            seasons.append(season_vals[t])
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_static, dtype=np.float32),
        np.array(X_pid, dtype=np.int32),
        np.array(y, dtype=np.float32),
        np.array(seasons),
    )

t0 = time.time()
X_seq_all, X_static_all, X_pid_all, y_all, seasons_all = build_sequences(
    df_scaled, seq_feature_cols, static_feature_cols, SEQ_LEN)
print(f'Built {len(X_seq_all)} sequences in {time.time()-t0:.1f}s')
print(f'Shape seq={X_seq_all.shape}  static={X_static_all.shape}')
""")

md("---\n## 7. Temporal Splits")
code("""tr = np.isin(seasons_all, train_seasons)
va = np.isin(seasons_all, val_seasons)
te = np.isin(seasons_all, test_seasons)

X_seq_tr, X_static_tr, X_pid_tr, y_tr = X_seq_all[tr], X_static_all[tr], X_pid_all[tr], y_all[tr]
X_seq_va, X_static_va, X_pid_va, y_va = X_seq_all[va], X_static_all[va], X_pid_all[va], y_all[va]
X_seq_te, X_static_te, X_pid_te, y_te = X_seq_all[te], X_static_all[te], X_pid_all[te], y_all[te]

print(f'train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}')
""")

md("---\n## 8. Target Transforms")
code("""class TargetTransform:
    def __init__(self, name):
        self.name = name
    def forward(self, y):
        if self.name == 'sqrt':
            return np.sqrt(np.clip(y, 0, None))
        elif self.name == 'log1p':
            return np.log1p(np.clip(y, 0, None))
        raise ValueError(self.name)
    def inverse(self, y_pred):
        y_pred = np.clip(y_pred, 0, None)
        if self.name == 'sqrt':
            return y_pred ** 2
        elif self.name == 'log1p':
            return np.expm1(y_pred)
        raise ValueError(self.name)

sqrt_t = TargetTransform('sqrt')
log1p_t = TargetTransform('log1p')

y_tr_sqrt = sqrt_t.forward(y_tr);  y_va_sqrt = sqrt_t.forward(y_va);  y_te_sqrt = sqrt_t.forward(y_te)
y_tr_log  = log1p_t.forward(y_tr); y_va_log  = log1p_t.forward(y_va); y_te_log  = log1p_t.forward(y_te)
print('sqrt  range:', y_tr_sqrt.min(), y_tr_sqrt.max())
print('log1p range:', y_tr_log.min(), y_tr_log.max())
""")

md("---\n## 9. Attention Pooling Layer")
code("""class AttentionPool(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = layers.Dense(1)
        self.supports_masking = True

    def call(self, x, mask=None):
        scores = self.score(x)
        if mask is not None:
            mask_f = tf.cast(mask, tf.float32)[..., tf.newaxis]
            scores = scores + (1.0 - mask_f) * -1e9
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(x * weights, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

print('AttentionPool defined.')
""")

md("""---
## 10. Model Builder (v2 - Relaxed Regularization)

**Razlike u odnosu na v1:**
- `dropout=0.15` (v1: 0.25)
- `rnn_dropout=0.10` (v1: 0.15)
- GaussianNoise je potpuno izbacen (v1: noise=0.05)
- `wd=1e-4` (v1: 5e-4)
- `lr=2e-4` (v1: 5e-4)
""")
code("""def build_model_v2(rnn_type, seq_len, n_seq_feat, n_static_feat, n_players,
                   rnn_units=128, player_emb_dim=8,
                   dropout=0.15, rnn_dropout=0.10,
                   huber_delta=1.0, lr=2e-4, wd=1e-4):
    RNNCell = LSTM if rnn_type == 'LSTM' else GRU

    seq_input = Input(shape=(seq_len, n_seq_feat), name='seq_input')
    x = Masking(mask_value=0.0)(seq_input)
    x = Bidirectional(RNNCell(rnn_units, return_sequences=True,
                              dropout=rnn_dropout, recurrent_dropout=rnn_dropout))(x)
    x = AttentionPool()(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout)(x)

    pid_input = Input(shape=(), dtype='int32', name='pid_input')
    p = Embedding(n_players, player_emb_dim,
                  embeddings_regularizer=regularizers.l2(1e-5))(pid_input)
    p = Flatten()(p)

    # Static branch - NO GaussianNoise in v2
    static_input = Input(shape=(n_static_feat,), name='static_input')
    s = Dense(64, activation='relu')(static_input)
    s = LayerNormalization()(s)
    s = Dropout(dropout)(s)

    merged = Concatenate()([x, p, s])
    merged = Dense(96, activation='relu')(merged)
    merged = LayerNormalization()(merged)
    merged = Dropout(dropout * 0.5)(merged)
    output = Dense(1)(merged)

    model = Model(inputs=[seq_input, pid_input, static_input], outputs=output)
    opt = optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    model.compile(optimizer=opt, loss=losses.Huber(delta=huber_delta), metrics=['mae'])
    return model

_m = build_model_v2('GRU', SEQ_LEN, n_seq_features, n_static_features, n_players)
print(f'params: {_m.count_params():,}')
del _m
""")

md("""---
## 11. Train 4 Models: {LSTM, GRU} x {sqrt, log1p}

**patience=50** (v1: 25), sve ostalo identicno (BATCH_SIZE=32, SW_STRENGTH=0.7, EPOCHS=500).
""")
code("""def make_weights(strength, y_orig):
    mean_y = np.mean(np.clip(y_orig, 0, None))
    base = np.sqrt(np.clip(y_orig, 0, None) / mean_y)
    return 1.0 + strength * base

SW_STRENGTH = 0.7
BATCH_SIZE = 32
EPOCHS = 500
PATIENCE = 50  # v1 was 25

configs = [
    ('LSTM', 'sqrt',  sqrt_t,  y_tr_sqrt, y_va_sqrt),
    ('LSTM', 'log1p', log1p_t, y_tr_log,  y_va_log),
    ('GRU',  'sqrt',  sqrt_t,  y_tr_sqrt, y_va_sqrt),
    ('GRU',  'log1p', log1p_t, y_tr_log,  y_va_log),
]

results = {}
histories = {}
sw_tr = make_weights(SW_STRENGTH, y_tr)

for rnn_type, tname, transform, y_tr_t, y_va_t in configs:
    key = f'{rnn_type}_{tname}'
    print(f'\\n{"="*70}\\nTraining {key} (v2)\\n{"="*70}')

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model = build_model_v2(rnn_type, SEQ_LEN, n_seq_features, n_static_features, n_players)

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                restore_best_weights=True, min_delta=1e-4),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=15, min_lr=1e-6),
    ]

    t0 = time.time()
    hist = model.fit(
        [X_seq_tr, X_pid_tr, X_static_tr], y_tr_t,
        sample_weight=sw_tr,
        validation_data=([X_seq_va, X_pid_va, X_static_va], y_va_t),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=cb, verbose=0,
    )
    elapsed = time.time() - t0

    pred_te_raw = model.predict([X_seq_te, X_pid_te, X_static_te], verbose=0).flatten()
    pred_te_yards = transform.inverse(pred_te_raw)

    mae = mean_absolute_error(y_te, pred_te_yards)
    rmse = np.sqrt(mean_squared_error(y_te, pred_te_yards))
    r2 = r2_score(y_te, pred_te_yards)

    best_ep = int(np.argmin(hist.history['val_loss']))
    results[key] = {
        'rnn_type': rnn_type, 'target': tname,
        'test_mae': float(mae), 'test_rmse': float(rmse), 'test_r2': float(r2),
        'best_epoch': best_ep + 1,
        'total_epochs': len(hist.history['loss']),
        'train_loss': float(hist.history['loss'][best_ep]),
        'val_loss': float(hist.history['val_loss'][best_ep]),
        'loss_gap': float(hist.history['val_loss'][best_ep] - hist.history['loss'][best_ep]),
        'train_time_s': round(elapsed, 1),
    }
    histories[key] = hist.history

    print(f'  best_epoch={best_ep+1}/{len(hist.history["loss"])}')
    print(f'  train_loss={results[key]["train_loss"]:.4f}  val_loss={results[key]["val_loss"]:.4f}')
    print(f'  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}   ({elapsed:.0f}s)')
""")

md("---\n## 12. v2 vs v1 Comparison Table")
code("""rows = []
for key, r in results.items():
    rows.append({
        'Model': f'{key}_v2',
        'RNN': r['rnn_type'],
        'Target': r['target'],
        'Test MAE': round(r['test_mae'], 2),
        'Test RMSE': round(r['test_rmse'], 2),
        'Test R2': round(r['test_r2'], 4),
        'Best Epoch': r['best_epoch'],
        'Total Epochs': r['total_epochs'],
        'Loss Gap': round(r['loss_gap'], 4),
        'Time (s)': r['train_time_s'],
    })

# Load v1 results for comparison
try:
    with open('../results/rnn_attn_embed_results.json') as f:
        v1 = json.load(f)
    for key, r in v1['results'].items():
        rows.append({
            'Model': f'{key}_v1',
            'RNN': r['rnn_type'],
            'Target': r['target'],
            'Test MAE': round(r['test_mae'], 2),
            'Test RMSE': round(r['test_rmse'], 2),
            'Test R2': round(r['test_r2'], 4),
            'Best Epoch': r['best_epoch'],
            'Total Epochs': '-',
            'Loss Gap': round(r['loss_gap'], 4),
            'Time (s)': r['train_time_s'],
        })
except FileNotFoundError:
    print('v1 results not found - skipping comparison rows.')

df_res = pd.DataFrame(rows).sort_values('Test R2', ascending=False).reset_index(drop=True)
print(df_res.to_string(index=False))
""")

md("---\n## 13. Learning Curves (v2)")
code("""fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, (key, hist) in zip(axes.flat, histories.items()):
    ep = range(1, len(hist['loss']) + 1)
    ax.plot(ep, hist['loss'], label='Train', linewidth=1.4)
    ax.plot(ep, hist['val_loss'], label='Val', linewidth=1.4)
    best_ep = int(np.argmin(hist['val_loss'])) + 1
    ax.axvline(best_ep, color='red', linestyle='--', alpha=0.6, label=f'best @ {best_ep}')
    ax.set_title(f'{key} (v2)'); ax.set_xlabel('Epoch'); ax.set_ylabel('Huber loss')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/rnn_attn_embed_v2_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
""")

md("""---
## 14. Diagnostika: da li su HP-ovi krivi, ili arhitektura?

Pravilo: ako je u v1 `best_epoch < 10` za sve modele, a v2 popravi to (npr. `best_epoch >= 20`)
i pritom test metrike idu nagore ili bar ostaju iste - zakljucujemo da je **v1 bio preregularizovan**.

Ako **i v2** zavrsi sa `best_epoch < 10` - znaci ni opusteniji HP-ovi ne pomazu, pa sama arhitektura
(attention + embedding) **nije podesena za ovaj skup podataka** (premalo utakmica po igracu da bi
self-attention nauceio smislene tezine, premalo ponavljanja po igracu da bi embedding bio koristan).
""")
code("""best_epochs_v2 = [r['best_epoch'] for r in results.values()]
max_bep = max(best_epochs_v2)
min_bep = min(best_epochs_v2)
mean_bep = sum(best_epochs_v2) / len(best_epochs_v2)

print(f'v2 best_epoch: min={min_bep}  mean={mean_bep:.1f}  max={max_bep}')
print(f'v2 per model: {dict(zip(results.keys(), best_epochs_v2))}')

# Compare vs v1 if available
v1_conclusion = ''
try:
    with open('../results/rnn_attn_embed_results.json') as f:
        v1 = json.load(f)
    v1_beps = [r['best_epoch'] for r in v1['results'].values()]
    print(f'\\nv1 best_epoch: min={min(v1_beps)}  mean={sum(v1_beps)/len(v1_beps):.1f}  max={max(v1_beps)}')
    if max(v1_beps) < 10 and max_bep >= 20:
        v1_conclusion = 'v1 je bio preregularizovan - v2 trenira znatno duze.'
    elif max(v1_beps) < 10 and max_bep < 10:
        v1_conclusion = 'i v2 zaustavlja rano - arhitektura je problem, ne HP-ovi.'
    else:
        v1_conclusion = 'v1 je vec treniao dovoljno dugo - opustanje HP-ova nije promenilo sliku.'
except FileNotFoundError:
    pass

print('\\n' + '='*70)
print('ZAKLJUCAK:')
print('='*70)
if max_bep < 10:
    print('  ARHITEKTURA JE PROBLEM.')
    print(f'  I sa dropout=0.15, noise=0, wd=1e-4, lr=2e-4, patience=50,')
    print(f'  svi modeli se zaustavljaju pre 10. epohe (max best_epoch = {max_bep}).')
    print('  Attention pooling + player embeddings ne odgovaraju ovom skupu podataka:')
    print('    * premalo utakmica po igracu za smislene attention weights')
    print('    * premalo ponavljanja po igracu za korisne embedding vektore')
    print('  Preporuka: zadrzati standardni LSTM last-hidden-state pristup iz RNN_Improved.')
elif min_bep >= 20:
    print('  HP-OVI SU BILI KRIVI U v1.')
    print(f'  v2 sada trenira realnije (min best_epoch = {min_bep}).')
    print('  Uporedi test metrike v2 vs v1 - ako je v2 bolji, potvrdjeno je da je')
    print('  v1 bio preregularizovan, a arhitektura je u redu.')
else:
    print('  MESAVINA - neki modeli treniraju duze, neki jos uvek staju rano.')
    print(f'  Raspon best_epoch: [{min_bep}, {max_bep}]. Dodatno tuniranje moze pomoci.')
if v1_conclusion:
    print('\\nPoredjenje sa v1:', v1_conclusion)
""")

md("---\n## 15. Save Results")
code("""output = {
    'version': 'v2_relaxed',
    'config': {
        'seq_len': SEQ_LEN, 'batch_size': BATCH_SIZE, 'sw_strength': SW_STRENGTH,
        'n_players': int(n_players), 'player_emb_dim': 8,
        'rnn_units': 128, 'dropout': 0.15, 'rnn_dropout': 0.10,
        'noise': 0.0, 'huber_delta': 1.0, 'lr': 2e-4, 'wd': 1e-4,
        'patience': PATIENCE,
    },
    'n_seq_features': n_seq_features,
    'n_static_features': n_static_features,
    'results': results,
}
with open('../results/rnn_attn_embed_v2_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

df_res.to_csv('../results/rnn_attn_embed_v2_comparison.csv', index=False)
print('Saved: results/rnn_attn_embed_v2_results.json')
print('Saved: results/rnn_attn_embed_v2_comparison.csv')
print('Saved: results/rnn_attn_embed_v2_learning_curves.png')
""")

out = 'notebooks/WR_RNN_Attention_PlayerEmbed_v2.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Created {out} with {len(nb["cells"])} cells')
