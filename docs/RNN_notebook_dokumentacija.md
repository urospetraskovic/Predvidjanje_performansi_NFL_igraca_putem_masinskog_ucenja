# RNN.ipynb — Kompletna dokumentacija

Ovaj dokument detaljno opisuje sve korake, modele, arhitekture, parametre i eksperimente
koji su sprovedeni u `notebooks/RNN.ipynb`. Notebook se bavi **predikcijom prijemnih jardi
(receiving yards) NFL wide receivera** koristeći sekvencijalne i klasične ML modele.

---

## Sadrzaj

1. [Podaci](#1-podaci)
2. [Feature Engineering (Koraci 1–9)](#2-feature-engineering-koraci-19)
3. [Filtriranje i transformacija targeta (Koraci 10–12)](#3-filtriranje-i-transformacija-targeta-koraci-1012)
4. [Vremenski split (Korak 13)](#4-vremenski-split-korak-13)
5. [Skaliranje fičera (Korak 15)](#5-skaliranje-ficera-korak-15)
6. [Izgradnja sekvenci za LSTM/GRU (Korak 16)](#6-izgradnja-sekvenci-za-lstmgru-korak-16)
7. [Bazni LSTM i GRU (Korak 17)](#7-bazni-lstm-i-gru-korak-17)
8. [Evaluacija baznih modela (Korak 18)](#8-evaluacija-baznih-modela-korak-18)
9. [Selekcija fičera — Random Forest (Korak 19)](#9-selekcija-ficera--random-forest-korak-19)
10. [Povecana arhitektura — BatchNorm (Korak 20)](#10-povecana-arhitektura--batchnorm-korak-20)
11. [Kontrolni eksperimenti — Huber i sample weights (Korak 21)](#11-kontrolni-eksperimenti--huber-i-sample-weights-korak-21)
12. [Keras Tuner — Bayesian Optimization za LSTM i GRU (Korak 22)](#12-keras-tuner--bayesian-optimization-za-lstm-i-gru-korak-22)
13. [Tuned MLP na flat fičerima (Korak 23)](#13-tuned-mlp-na-flat-fičerima-korak-23)
14. [Sqrt target, Continuous Weights, BiLSTM, Attention (Korak 24)](#14-sqrt-target-continuous-weights-bilstm-attention-korak-24)
15. [MLP retrain — sqrt + continuous weights (Korak 25)](#15-mlp-retrain--sqrt--continuous-weights-korak-25)
16. [Uklanjanje roll3 fičera (Korak 26)](#16-uklanjanje-roll3-ficera-korak-26)
17. [Fixed-LR retraining sa 172 fičera (Korak 26A)](#17-fixed-lr-retraining-sa-172-ficera-korak-26a)
18. [Klasicni ML modeli — GridSearch (Korak 26B)](#18-klasicni-ml-modeli--gridsearch-korak-26b)
19. [Finalna tabela rezultata](#19-finalna-tabela-rezultata)

---

## 1. Podaci

- **Fajl:** `../data/fully combined/wr_all_weeks.csv`
- **Pozicija:** iskljucivo WR (wide receiver)
- **Sezone:** 2015–2025
- **Sortiranje:** po `receiver_player_id`, `season`, `week` (hronoloski)

---

## 2. Feature Engineering (Koraci 1–9)

### Korak 1 — Ucitavanje i pregled
Ucitavanje CSV-a, provera tipova kolona, missing values, distribucija po sezonama.

### Korak 6 — Ekstrakcija nedjelje (week)
```python
df['week'] = df['game_id'].str.split('_').str[1].astype(int)
df = df.sort_values(['receiver_player_id', 'season', 'week'])
```

### Korak 7 — Weeks since last game
```python
df['weeks_since_last_game'] = (
    df.groupby(['receiver_player_id', 'season'])['week']
      .diff().fillna(1).clip(lower=1).astype(int)
)
```

### Korak 8 — Previous-season career features
Za svakog igraca, izracunate su prosecne statistike iz prethodne sezone:

| Feature | Opis |
|---------|------|
| `career_avg_yards_last_season` | Prosek jardi prethodne sezone |
| `career_avg_target_share_last_season` | Prosek udela targeta |
| `career_avg_epa_last_season` | Prosek EPA |
| `career_avg_air_yard_share_last_season` | Udeo air yardi |
| `career_avg_catch_rate_last_season` | Prosek catch rate |
| `career_games_last_season` | Broj odigranih utakmica |

> Sve su `shift(1)` po sezoni — nema curenja buducih podataka.
> Nedostajuce vrednosti (pocetna sezona igraca) su popunjene nulom.

### Korak 9 — Lag1, Roll3, Roll5 fičeri

Za 70+ izvora kolona, napravljeni su **three derived features po koloni**:

| Sufiks | Opis |
|--------|------|
| `_lag1` | Vrednost iz prethodne utakmice (`shift(1)`) |
| `_roll3` | Rolling prosek prethodne 3 utakmice (`shift(1).rolling(3).mean()`) |
| `_roll5` | Rolling prosek prethodnih 5 utakmica (`shift(1).rolling(5).mean()`) |

**Izvorne kolone (rolling source):**
`targets`, `receptions`, `air_yards`, `yac`, `tds`, `epa`, `wpa`, `catch_rate`,
`avg_depth`, `adot`, `yac_per_reception`, `td_rate`, `explosive_plays`, `first_downs`,
`yards_per_target`, `team_pass_attempts`, `team_air_yards`, `team_epa`, `air_yard_share`,
`target_share`, `qb_completions`, `qb_attempts`, `qb_air_yards`, `qb_cpoe`, `qb_comp_pct`,
`avg_score_diff`, `trailing_pct`, `leading_pct`, `avg_quarter`, `success_rate`,
`big_play_rate`, `avg_start_yardline`, `red_zone_targets`, `end_zone_targets`,
`third_down_targets`, `fourth_down_targets`, `high_leverage_targets`,
`second_and_long_targets`, `third_and_medium_targets`, `wp_var`, `target_share_std`,
`reception_std`, `def_targets_dev`, `def_receptions_dev`, `def_yards_dev`, `def_tds_dev`,
`def_epa_dev`, `yards_Q1..Q4`, `receptions_Q1..Q4`, `targets_Q1..Q4`,
`lost_yards_due_to_penalty`, `yards_wp_*` (5 WP bucket-a), `receptions_wp_*`, `targets_wp_*`,
`weeks_since_last_game`

> Posle kreiranja lag/roll fičera, **originalne kolone su obrisane** iz dataseta da se sprice leakage.

### Pre-game Features (Korak 6)
Direktni pre-game fičeri koji su bezbedni bez shift-ovanja:
`pregame_spread`, `pregame_total`, `surface`, `is_dome`, `temp_f`, `wind_speed`,
`season`, `week`

**Ukupan broj fičera: ~248**

---

## 3. Filtriranje i transformacija targeta (Koraci 10–12)

### Korak 10 — Filter po broju utakmica
Zadrzani su samo player-season parovi sa **>= 6 odigranih utakmica** (potrebno za sekvence).

### Korak 11 — Brisanje redova bez istorije
Redovi gde su **sve** `_lag1` kolone NaN (sto znaci da igrac nema prethodnu utakmicu) su izbaceni.

### Korak 12 — Transformacija targeta: log1p
```python
model_df_final['receiving_yards_log'] = np.log1p(
    model_df_final['receiving_yards'].clip(lower=0)
)
```
Negativne vrednosti (kazneni yardi) su clipovane na 0 pre transformacije.
Za evaluaciju: `y_pred = np.expm1(y_pred_log)`.

---

## 4. Vremenski split (Korak 13)

Striktan hronoloski split — **nema data leakage** izmedju setova:

| Set | Sezone | Svrha |
|-----|--------|-------|
| Train | 2015–2021 | Treniranje modela |
| Validation | 2022–2023 | Selekcija modela i HP tuning |
| Test | 2024–2025 | Jednosmerna finalna evaluacija |

```
X_train.shape ~ (N_train, 248)
X_val.shape   ~ (N_val, 248)
X_test.shape  ~ (N_test, 248)
```

Target je `receiving_yards_log` (log1p skala). Identifikatori (`receiver_player_id`,
`game_id`, itd.) su isključeni iz feature matrice.

---

## 5. Skaliranje fičera (Korak 15)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit samo na train
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
```

> **Scaler je fittovan iskljucivo na trening skupu** — val i test dobijaju iste parametre.

---

## 6. Izgradnja sekvenci za LSTM/GRU (Korak 16)

```python
def build_sequences(scaled_df, target_series, model_df_source, sequence_length=5):
```

**Logika:**
- Grupisanje po `receiver_player_id` + `season` (bez krosovanja sezona)
- Sliding window: za svaku grupu, uzimaju se prozori od 5 uzastopnih utakmica
- **Target:** 6. utakmica (sledeća posle sekvence)
- Nikada ne prelazi granicu sezone

**Izlazni oblici:**
```
X_train_seq.shape: (N_seq_train, 5, 248)   — (uzorci, duzina_sekvence, fičeri)
y_train_seq.shape: (N_seq_train,)
```

> Svaki igrac-sezona sa manje od 6 utakmica ne daje nijedan sekvencijalni uzorak.

---

## 7. Bazni LSTM i GRU (Korak 17)

### Test run (mini modeli za validaciju pipeline-a)

Pre punog treninga, pokrenuti su mini modeli na podskupu (1000 train / 200 val):

```python
# LSTM test model
LSTM(16) -> Dropout(0.2) -> LSTM(8) -> Dropout(0.2) -> Dense(8, relu) -> Dense(1)

# GRU test model
GRU(16)  -> Dropout(0.2) -> GRU(8)  -> Dropout(0.2) -> Dense(8, relu) -> Dense(1)
```
Trening: 3 epohe, batch_size=256.

### Puni bazni modeli

**LSTM arhitektura:**
```
Input(shape=(5, 248))
LSTM(128, return_sequences=True)
Dropout(0.2)
LSTM(64, return_sequences=False)
Dropout(0.2)
Dense(32, activation='relu')
Dense(1)
```

**GRU arhitektura:**
```
Input(shape=(5, 248))
GRU(128, return_sequences=True)
Dropout(0.2)
GRU(64, return_sequences=False)
Dropout(0.2)
Dense(32, activation='relu')
Dense(1)
```

**Parametri treninga:**

| Parametar | Vrednost |
|-----------|----------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | MSE (`mean_squared_error`) |
| Epochs | 100 |
| Batch size | 64 |
| Seed | 42 (python, numpy, tensorflow) |

**Callbacks:**

| Callback | Parametri |
|----------|-----------|
| `ModelCheckpoint` | `monitor='val_loss'`, `save_best_only=True` |
| `EarlyStopping` | `patience=10`, `restore_best_weights=True` |
| `ReduceLROnPlateau` | `factor=0.5`, `patience=5`, `min_lr=1e-5` |

**Sacuvani fajlovi:** `lstm_best_model.keras`, `gru_best_model.keras`

---

## 8. Evaluacija baznih modela (Korak 18)

Predikcije u log skali, konvertovane nazad sa `np.expm1()`. Metrke na originalnoj skali:

```python
lstm_mae  = MAE(y_test_orig, np.expm1(lstm_pred_log))
lstm_rmse = RMSE(...)
lstm_r2   = R2(...)
```

**Poznati rezultati (baseline MSE):**

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| LSTM (baseline MSE) | 21.4732 | 31.4857 | 0.1053 |
| GRU (baseline MSE) | ~21.x | ~31.x | ~0.10x |

Dijagnosticki plotovi: scatter (predicted vs actual) + residual plot za oba modela (2x2 figura).

---

## 9. Selekcija fičera — Random Forest (Korak 19)

Selekcija top 25 fičera koristeci Random Forest na flat (ne-sekvencijalnim) podacima:

```python
rf_selector = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=2
)
rf_selector.fit(X_train_flat_df, y_train)
selected_features = importance_series.head(25).index.tolist()
```

Posle selekcije, sekvence su **rekonstruisane** sa samo 25 fičera:
```
X_train_seq_selected.shape: (N, 5, 25)
```

**Sacuvano:** `results/rf_feature_importance_top25.csv`

---

## 10. Povecana arhitektura — BatchNorm (Korak 20)

Novi modeli sa vise kapaciteta i BatchNormalization, trenirani na top-25 fičerima:

**LSTM BN+Large:**
```
Input(shape=(5, 25))
LSTM(256, return_sequences=True)
BatchNormalization()
Dropout(0.3)
LSTM(128, return_sequences=False)
BatchNormalization()
Dropout(0.3)
Dense(64, activation='relu')
Dense(32, activation='relu')
Dense(1)
```

**GRU BN+Large:** ista struktura sa GRU slojevima.

Isti optimizer, LR, epochs i callbacks kao u Koraku 17.
Sacuvano: `lstm_bn_large_top25_best.keras`, `gru_bn_large_top25_best.keras`

**Poredjenje (Korak 20b):**

| Model | Fičeri | MAE | RMSE | R2 |
|-------|--------|-----|------|----|
| LSTM (old, 248f) | 248 | ~21.47 | ~31.49 | ~0.105 |
| GRU (old, 248f) | 248 | ~21.x | ~31.x | ~0.10x |
| LSTM BN+Large (top25) | 25 | ... | ... | ... |
| GRU BN+Large (top25) | 25 | ... | ... | ... |

---

## 11. Kontrolni eksperimenti — Huber i sample weights (Korak 21)

Testiranje 4 varijante na originalna **248 fičera**, ista bazna arhitektura:
`LSTM(128) -> LSTM(64) -> Dense(32) -> Dense(1)`

**Varijante:**

| Naziv | Loss | Sample weights |
|-------|------|----------------|
| `baseline_mse` | MSE | Nema |
| `lstm_248_huber_only` | Huber(delta=1.0) | Nema |
| `lstm_248_mse_weights50` | MSE | Da (threshold 50 yardi) |
| `lstm_248_huber_weights50` | Huber(delta=1.0) | Da (threshold 50 yardi) |

**Sample weights logika:**
```python
weights = np.where(y_train_seq_original > 50, 2.0, 1.0)
# Utakmice sa >50 jardi dobijaju dupli tezinu
```

**Poznati rezultati:**

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| LSTM (baseline MSE) | 21.4732 | 31.4857 | 0.1053 |
| LSTM (huber only) | 20.5743 | 29.5129 | 0.2139 |

Huber loss znacajno poboljsava performanse u odnosu na cist MSE.

**Sacuvano:** `results/step21_huber_weights_on_248.csv`

---

## 12. Keras Tuner — Bayesian Optimization za LSTM i GRU (Korak 22)

### Search space

| Hiperparametar | Opcije |
|----------------|--------|
| `units_1` | [64, 128, 256] |
| `units_2` | [32, 64, 128] |
| `dropout_rate` | 0.1 – 0.4 (korak 0.1) |
| `learning_rate` | [1e-4, 1e-3, 1e-2] |
| `huber_delta` | [0.5, 1.0, 2.0, 3.0] |

Arhitektura za pretragu (LSTM i GRU varijanta):
```
Input(shape=(5, 248))
[LSTM|GRU](units_1, return_sequences=True)
BatchNormalization()
Dropout(dropout_rate)
[LSTM|GRU](units_2, return_sequences=False)
BatchNormalization()
Dropout(dropout_rate)
Dense(units_2 // 2, activation='relu')
Dense(1)
```

### Parametri pretrage

| Parametar | Vrednost |
|-----------|----------|
| Metod | `BayesianOptimization` |
| `max_trials` | 30 |
| Objektiv | `val_loss` (min) |
| Epohe tokom pretrage | 50 |
| EarlyStopping (pretraga) | `patience=5` |

Direktorijumi: `tuning_lstm/lstm_wr_tuning`, `tuning_gru/gru_wr_tuning`

### Finalni trening sa najboljim HP
```
Epochs: 100, batch_size: 64
Callbacks: ModelCheckpoint, EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=5)
```

**Poznata poredjenja:**

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| LSTM (baseline MSE) | 21.4732 | 31.4857 | 0.1053 |
| LSTM (huber only) | 20.5743 | 29.5129 | 0.2139 |
| LSTM (tuned) | ... | ... | ... |
| GRU (tuned) | ... | ... | ... |

**Sacuvano:** `lstm_tuned_final_saved.keras`, `gru_tuned_final_saved.keras`,
`results/tuned_lstm_gru_comparison.csv`

---

## 13. Tuned MLP na flat fičerima (Korak 23)

Multilayer Perceptron na **flat** (ne-sekvencijalnim) fičerima — bez sekvenci, direktan ulaz 248 fičera.

### Search space

| Hiperparametar | Opcije |
|----------------|--------|
| `units_1` | [128, 256, 512] |
| `units_2` | [64, 128, 256] |
| `units_3` | [32, 64, 128] |
| `dropout_rate` | 0.1 – 0.4 (korak 0.1) |
| `learning_rate` | [1e-4, 1e-3, 1e-2] |
| `huber_delta` | [0.5, 1.0, 2.0, 3.0] |

**Arhitektura:**
```
Input(shape=(248,))
Dense(units_1, activation='relu')
BatchNormalization()
Dropout(dropout_rate)
Dense(units_2, activation='relu')
BatchNormalization()
Dropout(dropout_rate)
Dense(units_3, activation='relu')
Dropout(dropout_rate / 2.0)
Dense(1)
```

BayesianOptimization, max_trials=30, direktorijum: `tuning_mlp/mlp_wr_tuning`

**Rezultati (log1p target, bez weights):**

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| MLP tuned (log1p) | 18.9713 | 27.1969 | 0.2716 |

**Sacuvano:** `mlp_final_best.keras`, `results/mlp_tuned_comparison.csv`

> MLP bez sekvenci vec pobeduje sve LSTM/GRU varijante iz prethodnih koraka!

---

## 14. Sqrt target, Continuous Weights, BiLSTM, Attention (Korak 24)

### Promena transformacije targeta

Prelaz sa **log1p** na **sqrt** transformaciju:
```python
y_orig  = np.clip(np.expm1(y_log), 0, None)
y_sqrt  = np.sqrt(y_orig)
# evaluacija: y_pred_orig = np.clip(y_pred_sqrt, 0, None) ** 2
```

### Agresivni kontinualni sample weights
```python
mean_y = np.mean(y_train_orig)
sample_weights = 1.0 + np.sqrt(y_train_orig / mean_y)
# Primeri: yards=0 -> weight=1.0, yards=30 -> ~1.8, yards=100 -> ~2.9, yards=150 -> ~3.5
```

### Korak 1: LSTM + sqrt + continuous weights

Ista bazna arhitektura `LSTM(128) -> LSTM(64) -> Dense(32) -> Dense(1)`,
ali sa sqrt targetom i kontinualnim tezinama.

### Korak 2: Bidirectional LSTM i GRU

**BiLSTM arhitektura:**
```
Input(shape=(5, 248))
Bidirectional(LSTM(128, return_sequences=True))
Dropout(0.25)
Bidirectional(LSTM(64, return_sequences=False))
Dropout(0.25)
Dense(64, activation='relu')
Dense(1)
```

**BiGRU arhitektura:** ista struktura sa GRU slojevima.

Loss: Huber(delta=1.0), Epochs: 100, Adam lr=0.001

### Korak 3: BiLSTM + Attention

```
Input(shape=(5, 248))
Bidirectional(LSTM(128, return_sequences=True))
Dropout(0.25)
Attention()([x, x])          -- self-attention po timestep-ima
GlobalAveragePooling1D()
Dense(64, activation='relu')
Dense(1)
```

Koristi Functional API umesto Sequential (zbog Attention sloja).

**Rezultati koraka 24:**

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| BiGRU (sqrt+cont_weights) | 20.8335 | 28.6505 | 0.2592 |
| LSTM (sqrt+cont_weights) | 20.8711 | 28.7993 | 0.2515 |
| BiLSTM (sqrt+cont_weights) | 21.4936 | 29.0174 | 0.2401 |
| BiLSTM+Attention | ... | ... | ... |

**Sacuvano:** `results/step24_sqrt_weights_bi_attention_comparison.csv`

---

## 15. MLP retrain — sqrt + continuous weights (Korak 25)

Reload best HP iz prethodnog Keras Tuner run-a (`tuning_mlp/mlp_wr_tuning`, `overwrite=False`),
pa retrain sa novom sqrt transformacijom i kontinualnim tezinama:

```python
weights_train = 1 + np.sqrt(y_train_original / y_train_original.mean())
mlp_improved.fit(X_train_scaled, y_train_sqrt, sample_weight=weights_train, ...)
```

**Poredjenje sa prethodnim MLP:**

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| MLP improved (sqrt+weights) | 19.0419 | 26.6142 | 0.3025 |
| MLP tuned (log1p, no weights) | 18.9713 | 27.1969 | 0.2716 |

Sqrt+weights daje bolji RMSE i R2, ali nesto losiji MAE.

**Sacuvano:** `mlp_sqrt_weights_best.keras`, `results/mlp_sqrt_weights_comparison.csv`

---

## 16. Uklanjanje roll3 fičera (Korak 26)

**Motivacija:** `_roll3` fičeri su redundantni — informacija je vec pokrivena `_lag1` i `_roll5`.

### Korak 1 — Identifikacija i uklanjanje
```python
roll3_cols_removed = [col for col in feature_columns if col.endswith('_roll3')]
# Uklonjeno ~76 kolona
# Ostalo: 248 - 76 = 172 fičera
```

### Korak 2 — Rekonstrukcija sekvenci
Iste sekvence (duzina 5), ali sa 172 fičera:
```
X_train_seq_r.shape: (N, 5, 172)
```

### Koraci 3-5 — Retrain MLP, LSTM, GRU

**MLP (isti tuned HP, novi input_dim=172):**
```
Dense(units_1) + BN -> Dense(units_2) + BN -> Dense(units_3) -> Dense(1)
```

**LSTM reduced:**
```
LSTM(128) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Dense(32) -> Dense(1)
Adam lr=0.001, Huber(delta=1.0), sqrt target, continuous weights, 100 epochs
```

**GRU reduced:** ista arhitektura sa GRU.

**Rezultati (korak 6) — poredjenje sa 248-feature baseline:**

| Model | Fičeri | MAE | RMSE | R2 |
|-------|--------|-----|------|----|
| MLP reduced (lag1+roll5) | 172 | 19.1721 | 26.5865 | 0.3040 |
| LSTM reduced | 172 | 20.5933 | 28.2596 | 0.2793 |
| BiGRU (lag1+roll3+roll5) | 248 | 20.8335 | 28.6505 | 0.2592 |
| GRU reduced | 172 | 21.0792 | 29.0023 | 0.2409 |
| LSTM huber only | 248 | 20.5743 | 29.5129 | 0.2139 |
| LSTM baseline MSE | 248 | 21.4732 | 31.4857 | 0.1053 |

**Sacuvano:** `mlp_r_best.keras`, `lstm_r_best.keras`, `gru_r_best.keras`,
`results/roll3_removed_reduced_vs_full_comparison.csv`

---

## 17. Fixed-LR retraining sa 172 fičera (Korak 26A)

Retrain LSTM i GRU na **172 fičera** (bez roll3) sa poboljsanim callback parametrima.

### Arhitektura (sa BatchNorm izmedju LSTM slojeva)

**LSTM fixed:**
```
Input(shape=(5, 172))
LSTM(128, return_sequences=True)
BatchNormalization()
Dropout(0.2)
LSTM(64, return_sequences=False)
BatchNormalization()
Dropout(0.2)
Dense(32, activation='relu')
Dense(1)
```

**GRU fixed:** ista struktura sa GRU slojevima.

### Poboljsani callback parametri

| Callback | Stari parametri | Novi parametri |
|----------|----------------|----------------|
| EarlyStopping | patience=10 | patience=15, min_delta=0.005 |
| ReduceLROnPlateau | factor=0.5, patience=5 | factor=0.3, patience=5 |

### Dodatni custom callback: LrTracker
Prati kada dolazi do pada learning rate-a:
```python
class LrTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_lr = ...
        if current_lr < prev_lr: self.lr_drop_epochs.append(epoch+1)
```

Plotovi ukljucuju vertikalne linije za svaki pad LR-a.

**Parametri treninga:**

| Parametar | Vrednost |
|-----------|----------|
| Loss | Huber(delta=1.0) |
| Target | sqrt(receiving_yards) |
| Sample weights | Kontinualni: 1 + sqrt(y / mean_y) |
| Epochs | 150 |
| Batch size | 64 |
| Adam LR | 0.001 |

**Sacuvano:** `lstm_fixed_best.keras`, `gru_fixed_best.keras`

**R2 poredjenje:**

| Model | Fičeri | LR | Epochs | R2 |
|-------|--------|-----|--------|----|
| MLP reduced | 172 | tuned | ~17 | 0.3040 |
| LSTM reduced (old) | 172 | 0.001 | 100 | 0.2793 |
| BiGRU | 248 | 0.001 | 100 | 0.2592 |
| GRU reduced (old) | 172 | 0.001 | 100 | 0.2409 |
| LSTM baseline MSE | 248 | 0.001 | 100 | 0.1053 |

---

## 18. Klasicni ML modeli — GridSearch (Korak 26B)

Finalni eksperiment: svi standardni ML modeli sa GridSearchCV na predefined split-u.

### Modeli i grids

| Model | Pretraga |
|-------|----------|
| LinearRegression | bez pretrage |
| Ridge | `alpha: [0.01, 0.1, 1.0, 10.0, 100.0]` |
| Lasso | `alpha: [0.001, 0.01, 0.1, 1.0]` |
| ElasticNet | `alpha: [0.01, 0.1, 1.0]`, `l1_ratio: [0.1, 0.5, 0.9]` |
| KNN | `n_neighbors: [3,5,7,9,11,15]`, `weights: ['uniform','distance']` |
| RandomForest | `n_estimators: [100,200,300]`, `max_depth: [5,10,None]`, `min_samples_split: [2,5]` |
| XGBRegressor | `n_estimators: [100,200,300]`, `max_depth: [3,5,7]`, `learning_rate: [0.01,0.05,0.1]`, `subsample: [0.8,1.0]` |
| LGBMRegressor | `n_estimators: [100,200,300]`, `max_depth: [3,5,7]`, `learning_rate: [0.01,0.05,0.1]`, `num_leaves: [15,31,63]` |

### Protokol

```python
# PredefinedSplit — val skup je fiksiran (nema random cross-validation)
val_fold = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])
ps = PredefinedSplit(test_fold=val_fold)

# Target: sqrt transformacija
y_train_sqrt = np.sqrt(np.clip(np.expm1(y_train), 0, None))

# GridSearchCV scoring
scoring='neg_root_mean_squared_error'

# Finalni retrain na train+val kombinovano
X_trainval = np.vstack([X_train_scaled, X_val_scaled])
final_model.fit(X_trainval, y_trainval_sqrt)
```

### Vizualizacije
- Top 20 feature importance plotovi za XGBoost i LightGBM
- RMSE comparison chart (svi modeli, klasicni vs neuralni)

---

## 19. Finalna tabela rezultata

Sve metrke su na **originalnoj skali receiving yardi** (posle inverzne transformacije).

### Neuralni modeli (hronoloski, po R2)

| Model | Target | Weights | Fičeri | MAE | RMSE | R2 |
|-------|--------|---------|--------|-----|------|----|
| MLP tuned | log1p | Ne | 248 | 18.9713 | 27.1969 | 0.2716 |
| MLP improved | sqrt | Kontinualni | 248 | 19.0419 | 26.6142 | 0.3025 |
| **MLP reduced** | **sqrt** | **Kontinualni** | **172** | **19.1721** | **26.5865** | **0.3040** |
| LSTM reduced | sqrt | Kontinualni | 172 | 20.5933 | 28.2596 | 0.2793 |
| BiGRU | sqrt | Kontinualni | 248 | 20.8335 | 28.6505 | 0.2592 |
| LSTM (sqrt+weights) | sqrt | Kontinualni | 248 | 20.8711 | 28.7993 | 0.2515 |
| BiLSTM | sqrt | Kontinualni | 248 | 21.4936 | 29.0174 | 0.2401 |
| GRU reduced | sqrt | Kontinualni | 172 | 21.0792 | 29.0023 | 0.2409 |
| LSTM huber only | log1p | Ne | 248 | 20.5743 | 29.5129 | 0.2139 |
| LSTM baseline MSE | log1p | Ne | 248 | 21.4732 | 31.4857 | 0.1053 |

### Kljucni zakljucci

1. **MLP nadmasuje LSTM/GRU** na ovom problemu — flat 172-feature vektor daje R2=0.304 vs LSTM R2=0.279
2. **Huber loss** je znacajno bolji od MSE (R2: 0.21 vs 0.10)
3. **Sqrt transformacija + kontinualni sample weights** poboljsavaju RMSE u odnosu na log1p
4. **Uklanjanje roll3 fičera** blago poboljsava MLP (172 > 248 fičera), uz manji model
5. **Bidirectionalni slojevi** ne donose dobit nad standard LSTM/GRU na ovim podacima
6. **BatchNormalization** je korisna kod vecih arhitektura (step 20), ali nije dramaticna promena
7. **Attention mehanizam** nije dramaticno poboljsao rezultate nad BiLSTM bazom

---

## Sacuvani fajlovi (checkpoints i rezultati)

| Fajl | Sadrzaj |
|------|---------|
| `lstm_best_model.keras` | Bazni LSTM (MSE, 248f) |
| `gru_best_model.keras` | Bazni GRU (MSE, 248f) |
| `lstm_bn_large_top25_best.keras` | LSTM BN+Large (top25 fičera) |
| `gru_bn_large_top25_best.keras` | GRU BN+Large (top25 fičera) |
| `lstm_248_huber_only.keras` | LSTM Huber loss (248f) |
| `lstm_248_mse_weights50.keras` | LSTM MSE + threshold weights |
| `lstm_248_huber_weights50.keras` | LSTM Huber + threshold weights |
| `lstm_final_best.keras` | LSTM tuned (Bayesian) |
| `gru_final_best.keras` | GRU tuned (Bayesian) |
| `lstm_tuned_final_saved.keras` | LSTM tuned (sacuvan posle eval) |
| `gru_tuned_final_saved.keras` | GRU tuned (sacuvan posle eval) |
| `mlp_final_best.keras` | MLP tuned (log1p, 248f) |
| `mlp_sqrt_weights_best.keras` | MLP improved (sqrt, 248f) |
| `mlp_r_best.keras` | MLP reduced (sqrt, 172f) |
| `lstm_r_best.keras` | LSTM reduced (sqrt, 172f) |
| `gru_r_best.keras` | GRU reduced (sqrt, 172f) |
| `lstm_fixed_best.keras` | LSTM fixed (BN, 172f, 150ep) |
| `gru_fixed_best.keras` | GRU fixed (BN, 172f, 150ep) |
| `tuning_lstm/` | Keras Tuner direktorijum za LSTM |
| `tuning_gru/` | Keras Tuner direktorijum za GRU |
| `tuning_mlp/` | Keras Tuner direktorijum za MLP |
| `results/rf_feature_importance_top25.csv` | RF feature importances |
| `results/step21_huber_weights_on_248.csv` | Kontrolni eksperimenti |
| `results/tuned_lstm_gru_comparison.csv` | Tuned modeli |
| `results/mlp_tuned_comparison.csv` | MLP vs ostali |
| `results/step24_sqrt_weights_bi_attention_comparison.csv` | BiLSTM/BiGRU/Attention |
| `results/mlp_sqrt_weights_comparison.csv` | MLP sqrt+weights |
| `results/roll3_removed_reduced_vs_full_comparison.csv` | 172f vs 248f |
