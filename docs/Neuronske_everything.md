# Neuronske Mreze za Predikciju NFL Wide Receiver Performansi — Kompletna Dokumentacija

> **Projekat:** Predvidjanje receiving yards NFL Wide Receiver-a po utakmici koristeci neuronske mreze, tree modele i ensemble metode.
> **Sezone:** 2015-2025 | **Dataset:** 46,115 game-level zapisa | **Igraci:** 1,719 WR-ova
> **Teorijski plafon:** R2 ~ 0.33-0.37 (dokazano ceiling analizom)

---

## Sadrzaj

1. [Uvod i Motivacija](#1-uvod-i-motivacija)
2. [Dataset i Feature Engineering](#2-dataset-i-feature-engineering)
3. [Organizacija Notebookova](#3-organizacija-notebookova)
4. [Faza 1 — EDA i Bazni Modeli](#4-faza-1--eda-i-bazni-modeli)
5. [Faza 2 — Objedinjeni Pipeline sa Optuna](#5-faza-2--objedinjeni-pipeline-sa-optuna)
6. [Faza 3 — Feature Selection](#6-faza-3--feature-selection)
7. [Faza 4 — MLP Varijante i Poboljsanja](#7-faza-4--mlp-varijante-i-poboljsanja)
8. [Faza 5 — RNN Redizajn](#8-faza-5--rnn-redizajn)
9. [Faza 6 — Napredne Tehnike](#9-faza-6--napredne-tehnike)
10. [Faza 7 — Ceiling Analiza](#10-faza-7--ceiling-analiza)
11. [Faza 8 — Feature Ablacija (TopN)](#11-faza-8--feature-ablacija-topn)
12. [Faza 9 — Finalni Modeli i Ensemble](#12-faza-9--finalni-modeli-i-ensemble)
13. [Kompletna Tabela Rezultata](#13-kompletna-tabela-rezultata)
14. [Kljucni Zakljucci i Naucene Lekcije](#14-kljucni-zakljucci-i-naucene-lekcije)
15. [Tehnicke Napomene](#15-tehnicke-napomene)
16. [Dalje Mogucnosti](#16-dalje-mogucnosti)

---

## 1. Uvod i Motivacija

### Problem

NFL Wide Receiver statistike su inherentno volatilne — igrac moze imati "boom" nedelju (100+ yards) ili "bust" (0-20 yards) bez jasnog obrasca. Ova varijansa stvara fundamentalni izazov za predikciju: koliko god model bio dobar, postoji **nepremostivi sum** u podacima koji ogranicava preciznost.

### Cilj

Predvideti koliko ce jarda (`receiving_yards`) NFL Wide Receiver imati u sledecoj utakmici na osnovu:
- Istorijskih statistika igraca (lag, rolling, career averages)
- Pregame informacija (spread, total, protivnik, vreme)
- Matchup podataka (defanzivna statistika protivnika)

### Pristup

Projekat je prosao kroz 9 faza razvoja sa 22 notebooka, testiraJuci:
- **Feed-forward mreze** (MLP) sa razlicitim regularizacionim tehnikama
- **Rekurentne mreze** (LSTM, GRU, BiRNN) sa attention mehanizmima i player embeddingom
- **Tree modele** (XGBoost, LightGBM, RandomForest) kao baseline
- **Ensemble metode** koje kombinuju vise modela
- **Kvantilnu regresiju** za intervale poverenja
- **Ceiling analizu** koja dokazuje teorijski plafon predikcije

### Kljucni Rezultati (ukratko)

| Metrika | Najbolji Rezultat | Model |
|---|---|---|
| **Najnizi MAE** | **17.708 yards** | Ensemble (constrained LS) |
| **Najvisi R2** | **0.3384** | Ensemble (XGB+LGB+MLP) |
| **Najnizi RMSE** | **25.05** | Ensemble (XGB+LGB+MLP) |
| **Najbolji single model (MAE)** | **17.75** | MLP Quantile q50 |
| **Najbolji RNN** | R2 = 0.3354 | LSTM Improved (1-layer, 128 units) |

---

## 2. Dataset i Feature Engineering

### 2.1 Izvorni Podaci

| Parametar | Vrednost |
|---|---|
| Izvor | `data/fully combined/wr_all_weeks.csv` |
| Zapisi | 46,115 game-level zapisa |
| Sezone | 2015-2025 (11 sezona) |
| Igraci | 1,719 unikatnih WR-ova |
| Originalne kolone | 99 |

**Target distribucija:** Mean = 29.6 yards, Std = 31.8, Max = 300. Jaka desna asimetrija. Oko 10-15% utakmica ima 0 yards.

### 2.2 Feature Engineering Pipeline

**Faza 1 — Bazne statistike:**
- `receiving_yards`, `targets`, `receptions`, `epa`, `target_share`, `air_yards_share`
- Pregame: `spread`, `total`, `home_away`, `opponent_def_rank`

**Faza 2 — Temporalne transformacije (po svakom obelezju):**

| Tip | Opis |
|---|---|
| `_lag1` | Vrednost iz prethodne utakmice |
| `_roll3` | Rolling prosek — poslednje 3 utakmice |
| `_roll5` | Rolling prosek — poslednje 5 utakmica |
| `_career_avg` | Karijerni prosek (expanding mean) |
| `_expanding_std` | Karijerna standardna devijacija |
| `_momentum` | Trend = roll5 - lag1 |

**Faza 3 — Interakciona obelezja:**
- `target_share x pregame_total` — sinergija volumena i tempa
- `weeks_since_last_game` — pauze, povrede, bye week
- `is_new_season` — flag za pocetak nove sezone

**Faza 4 — Matchup obelezja:**
- Expanding mean protivnikove defanzive (samo prosli podaci, bez leakage-a)
- Defanzivni EPA, yards allowed per game

**Ukupno: 184 inzenjerisana obelezja**, od kojih je **top 40** selektovano za vecinu modela.

### 2.3 Temporalni Split

Striktna hronoloska podela bez data leakage-a:

| Skup | Sezone | Velicina | Svrha |
|---|---|---|---|
| **Train** | 2015-2021 | 29,276 | Treniranje |
| **Val** | 2022-2023 | 8,921 | Tuning + early stopping |
| **Test** | 2024-2025 | 6,199 | Finalna evaluacija |

**StandardScaler** fitovan iskljucivo na train skupu. Target transformacija: `sqrt(receiving_yards)` (primarno) ili `log(1 + receiving_yards)` (RNN eksperimenti).

### 2.4 Sample Weights

`1 + strength x sqrt(y / mean_y)`, strength = 0.6, opseg [1.00, 2.87]. Vece utakmice (vise yards) dobijaju veci weight da model ne ignoriSe "boom" igrace.

---

## 3. Organizacija Notebookova

Svih 22 WR_ notebookova organizovano je u 5 logickih celina:

### `notebooks/01_eda/` — Eksploracija i Priprema

| Notebook | Opis |
|---|---|
| `WR_Analysis.ipynb` | Kompletna EDA, feature engineering, sequence kreiranje |
| `WR_NN.ipynb` | Inicijalni data prep za feedforward pristup (WIP) |
| `WR_Feature_Selection.ipynb` | XGB+LGB importance → top 40 features |

### `notebooks/02_mlp/` — Feed-Forward Modeli (MLP)

| Notebook | Opis |
|---|---|
| `WR_MLP_Comparison.ipynb` | A/B/C poredjenje: 184 vs 40 features, regularizacija |
| `WR_MLP_Hybrid.ipynb` | Modeli D/E: veliki kapacitet + GaussianNoise + Mixup |
| `WR_MLP_Quantile.ipynb` | Kvantilna regresija (q10/q50/q90), pinball loss |

### `notebooks/03_rnn/` — Rekurentni Modeli (RNN/GRU/LSTM)

| Notebook | Opis |
|---|---|
| `WR_GRU_Model.ipynb` | Baseline GRU (5-step sliding window) |
| `WR_Career_RNN_Optuna.ipynb` | Career-based sekvence, Optuna za MLP/RNN/TCN, ensemble |
| `WR_RNN_Improved.ipynb` | Redizajn: raw features, Masking, dual-input — **najbolji RNN** |
| `WR_RNN_Improved_GRU.ipynb` | LSTM vs GRU head-to-head ablacija |
| `WR_RNN_v2.ipynb` | Bug fix + engineered seq channels (lag1+roll3 = 45 kanala) |
| `WR_RNN_Attention_PlayerEmbed.ipynb` | AttentionPool + Player embeddings, 4 modela |
| `WR_RNN_Attention_PlayerEmbed_v2.ipynb` | Relaxirana regularizacija, dijagnostika |

### `notebooks/04_ceiling/` — Ceiling Analiza (Dijagnostika)

| Notebook | Opis |
|---|---|
| `WR_2_game_Ceiling_Smoothed.ipynb` | Centered 2-game rolling mean target |
| `WR_3_Ceiling_Smoothed.ipynb` | Centered 3-game rolling mean target |
| `WR_5_game_Ceiling_Smoothed.ipynb` | Centered 5-game rolling mean target |
| `WR_5_game_Ceiling_Smoothed_NoLeakage.ipynb` | 5-game sa per-split smoothingom (verifikacija) |
| `WR_7_game_Ceiling_Smoothed.ipynb` | Centered 7-game rolling mean target |

### `notebooks/05_ensemble_final/` — Finalni Modeli i Ensemble

| Notebook | Opis |
|---|---|
| `WR_Ensemble_Final.ipynb` | 5 base modela + 4 ensemble strategije |
| `WR_Final_Models.ipynb` | Svi finalni modeli sa fiksiranim hiperparametrima |
| `WR_TopN_Models_top5.ipynb` | Feature ablacija: samo top-5 features |
| `WR_TopN_Models_top10.ipynb` | Feature ablacija: samo top-10 features |

---

## 4. Faza 1 — EDA i Bazni Modeli

### WR_Analysis.ipynb

Polazna tacka — kompletna eksplorativna analiza podataka.

**Sta radi:**
- Ucitava 46,115 zapisa, 99 kolona, bez missing values
- EDA: distribucija receiving_yards, korelaciona analiza (top 20 features), games-per-player
- Feature engineering: 136 novih kolona (lag1-3, roll3/5/8, std, defanzivni matchup)
- Kreiranje sliding window sekvenci za GRU (8 timestep-ova x 236 features)

**Kljucno:** Postavio osnovu za ceo pipeline — pravila za shift(1) da se izbegne leakage, expanding mean za defanzivne matchupe, StandardScaler samo na train.

### WR_NN.ipynb

Kratak, nestrukturiran eksperiment (14 celija). Data prep za feedforward pristup sa log1p transformacijom. Koristi 95 raw features bez lag/rolling inzenjeringa.

### WR_GRU_Model.ipynb — Baseline

| Parametar | Vrednost |
|---|---|
| Arhitektura | GRU(32) → Dense(1) |
| Seq length | 5 utakmica |
| Features | 95 raw |
| Parametri | 12,417 |

| Metrika | Test |
|---|---|
| MAE | 20.31 |
| RMSE | 28.54 |
| R2 | 0.2590 |

Jednostavan baseline, referentna tacka za sve dalje eksperimente.

---

## 5. Faza 2 — Objedinjeni Pipeline sa Optuna

### WR_Career_RNN_Optuna.ipynb

Najobimniji notebook — unified pipeline koji poredi sve glavne arhitekture.

**Inovacije:**
- Career-based sekvence koje prelaze granice sezona
- `is_new_season` i `weeks_since_last_game` signali
- 184 inzenjerisanih obelezja
- Optuna TPE + MedianPruner (100 trials po modelu)

**Rezultati tree modela:**

| Model | Test MAE | Test RMSE | Test R2 |
|---|---|---|---|
| RandomForest | 18.01 | 25.14 | 0.3338 |
| LightGBM | 18.15 | 25.13 | 0.3346 |
| XGBoost | 18.15 | 25.15 | 0.3335 |

**Rezultati neuralnih modela:**

| Model | Test MAE | Test RMSE | Test R2 |
|---|---|---|---|
| MLP (Optuna) | 18.16 | 25.48 | 0.3156 |
| TCN (Dilated Conv1D) | 19.30 | 27.59 | 0.2453 |
| RNN (LSTM/GRU) | 19.73 | 27.93 | 0.2157 |

**Ensemble (XGB + LGB + MLP):**

| Test MAE | Test RMSE | Test R2 |
|---|---|---|
| 18.05 | 25.05 | **0.3384** |

**Kljucni nalaz:** Tree modeli drasticno nadmasuju neuralne mreze. Razlika R2 = 0.12 (LightGBM vs RNN). RNN je imao prevelik search space (1-4 sloja, 64-512 units, 184 features po timestep-u), bez padding/masking, sa gubitkom kratkih karijera. Ovo je pokrenulo sve dalje istrazivanje.

---

## 6. Faza 3 — Feature Selection

### WR_Feature_Selection.ipynb

**Metodologija:** XGBoost + LightGBM combined importance → top 40 obelezja.

**Top 5 najvaznih obelezja:**
1. `first_downs_roll5` (0.610)
2. `target_share_std_lag1` (0.588)
3. `targets_roll5` (0.581)
4. `air_yards_roll5` (0.285)
5. `pregame_total` (0.259)

**Kljucni nalaz:** 144 od 184 obelezja su sum za neuralne mreze. Top 40 pokriva 51.8% kumulativne vaznosti. Tree modeli jedva overfit-uju sa redukcijom (gap 0.10-0.24), ali NN modeli imaju znacajno bolju generalizaciju sa 40 features.

| Model | 184 feat R2 | 40 feat R2 | Overfit Gap (184) | Overfit Gap (40) |
|---|---|---|---|---|
| XGBoost | 0.3335 | 0.3310 | 0.115 | 0.102 |
| LightGBM | 0.3346 | 0.3278 | 0.237 | 0.225 |

Rezultat sacuvan u `results/selected_features_top40.json`.

---

## 7. Faza 4 — MLP Varijante i Poboljsanja

### WR_MLP_Comparison.ipynb — Kontrolisano poredjenje

Tri MLP modela za izolovanje pojedinacnih efekata:

| Model | Features | Arhitektura | Test MAE | Test RMSE | Test R2 |
|---|---|---|---|---|---|
| A — Original | 184 | [448,128,320,448,256] | 18.10 | 25.62 | 0.3079 |
| **B — Reduced** | **40** | ista | 18.19 | **25.23** | **0.3289** |
| C — Improved | 40 | kompaktan + reg. | **17.91** | 25.49 | 0.3151 |

**Tehnike u Modelu C:** GaussianNoise(0.2), Mixup(alpha=0.3), softened sample weights [1.0, 1.94], forsiran LayerNorm. Optuna 80 trials: 3 sloja [256,128,64].

**Zakljucak:** Feature redukcija 184→40 daje +0.021 R2. Trade-off: veci kapacitet = bolji R2; jaca regularizacija = bolji MAE.

### WR_MLP_Hybrid.ipynb — Kombinovanje najboljeg

**Model D (Fixed Hybrid):** Ista arhitektura kao B + GaussianNoise(0.15) + Mixup(alpha=0.2)
**Model E (Optuna Hybrid):** 100 trials, 4 sloja [320,384,192,128], noise=0.25, mixup=0.2

| Model | Test MAE | Test RMSE | Test R2 |
|---|---|---|---|
| D — Fixed Hybrid | **17.83** | 25.67 | 0.3055 |
| E — Optuna Hybrid | 17.88 | 25.35 | 0.3225 |

**Ukupni MLP rezultati (svih 5 modela):**

| Model | MAE | RMSE | R2 | Napomena |
|---|---|---|---|---|
| A (184 feat) | 18.10 | 25.62 | 0.3079 | Referentni |
| **B (40 feat)** | 18.19 | **25.23** | **0.3289** | Najbolji R2 |
| C (improved) | 17.91 | 25.49 | 0.3151 | Najmanji overfit |
| **D (fixed hybrid)** | **17.83** | 25.67 | 0.3055 | Najbolji MAE |
| E (optuna hybrid) | 17.88 | 25.35 | 0.3225 | Najbolji balans |

### WR_MLP_Quantile.ipynb — Kvantilna Regresija

**Arhitektura:** Ista MLP-B backbone [448,128,320,448,256], ali sa Dense(3) izlazom za q10, q50, q90.
**Loss:** Multi-kvantilna pinball loss.

| Metrika | Vrednost |
|---|---|
| MAE (q50) | **17.76 yards** |
| RMSE (q50) | 25.60 |
| R2 (q50) | 0.3093 |
| Coverage (q10-q90) | 78.8% (target 80%) |
| Interval width | 59.7 yards |

**Zakljucak:** Kvantilna regresija postize **najnizi MAE od svih single modela (17.76)**. Medijan je L1-robustan na outliere. Besplatno dobijamo intervale poverenja sa gotovo idealnom pokrivenoscu.

---

## 8. Faza 5 — RNN Redizajn

### WR_RNN_Improved.ipynb — Preokret

Totalni redizajn RNN-a koji ga je ucinio kompetitivnim sa tree modelima.

**Kljucne promene od originala:**

| Aspekt | Original RNN (R2=0.2157) | Improved RNN (R2=0.3354) |
|---|---|---|
| Seq features | 184 lag/roll po timestep-u | 33 **raw** game statistike |
| Static features | Sve u sekvenci | 20 pregame preko **zasebne grane** |
| Kratke karijere | Bacane (seq_len filter) | **Padding + Masking** |
| Arhitektura | 1-4 sloja, 64-512 units | Max 2 sloja, max 192 units |
| LayerNorm | Opciono (Optuna bira 'none') | **Forsiran** |
| Train samples | 12,709-21,943 | **29,276** (sve sacuvano) |

**Dual-input arhitektura:**
```
seq_input (T x 33 raw features)
  → Masking(0.0)
  → LSTM(128, unidirectional)
  → LayerNorm → Dropout(0.30)
                                    → Concat → Dense → Output
static_input (20 pregame features)
  → GaussianNoise(0.20)
  → Dense → LayerNorm → Dropout
```

**Rezultat:** MAE = 18.03, RMSE = 25.11, **R2 = 0.3354** — najbolji single model u tom trenutku, bolji od LightGBM (0.3346).

**Poboljsanje:** MAE -1.70, RMSE -2.82, R2 +0.1197 u odnosu na original.

### WR_RNN_Improved_GRU.ipynb — LSTM vs GRU

Head-to-head ablacija: isti hyperparametri, razlicit cell type.

| Model | Cell | MAE | RMSE | R2 |
|---|---|---|---|---|
| LSTM Optuna-best | LSTM | 18.03 | 25.11 | **0.3354** |
| GRU @ LSTM HPs | GRU | 18.14 | 25.13 | 0.3344 |
| GRU Optuna-best | GRU | 18.09 | 25.16 | 0.3326 |

**Zakljucak:** LSTM blago bolji, ali razlika je minimalna (R2 gap = 0.001). Arhitektura je vaznija od tipa celije.

### WR_RNN_v2.ipynb — Bug Fix i Eksperimenti

**Sest promena:** (1) Bug fix za target scaling, (2) engineered seq kanali (33 raw + 6 lag1 + 6 roll3 = 45), (3) kraci seq_len [4,6,8], (4) blaza regularizacija, (5) vise kapaciteta, (6) no-mask baseline.

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| RNN v2 (padded) | 18.13 | 25.27 | 0.3268 |
| RNN v2 (no-mask) | 18.78 | 25.90 | 0.3168 |

**Zakljucak:** Engineered kanali i kraci seq_len nisu pomogli — originalni Improved RNN sa raw features i seq_len=12 ostaje bolji. No-mask baseline je losiji, potvrdujuci da je Masking+padding koristan.

### WR_RNN_Attention_PlayerEmbed.ipynb — Tri-input dizajn

**Arhitektura:**
```
Sequence Branch:
  seq_input (T x 45 features)
  → Masking → BiRNN(return_seq=True) → AttentionPool → LayerNorm → Dropout

Player Embedding Branch:
  pid_input → Embedding(1308, 8, L2=1e-4) → Flatten

Static Branch:
  static_input (20 features) → GaussianNoise → Dense → LayerNorm → Dropout

                    ↓ Concatenate ↓
                  Dense(96) → LayerNorm → Dropout → Dense(1)
```

**Custom AttentionPool:**
```python
class AttentionPool(layers.Layer):
    # Uci tezine za sve timestep-ove umesto samo poslednjeg hidden state-a
    scores = Dense(1)(x)        # (B, T, 1)
    scores[padded] = -1e9       # Masking
    weights = softmax(scores)
    output = sum(x * weights)   # Weighted pooling
```

**4 modela (grid: {LSTM,GRU} x {sqrt,log1p}):**

| Model | MAE | RMSE | R2 | Best Epoch |
|---|---|---|---|---|
| GRU_sqrt | 18.40 | 25.33 | 0.3236 | 3 |
| LSTM_sqrt | 18.58 | 25.48 | 0.3158 | 3 |
| GRU_log1p | 18.58 | 25.53 | 0.3131 | 3 |
| LSTM_log1p | 18.65 | 25.58 | 0.3104 | 3 |

**Problem:** Svi modeli su stali na epoch 3 — overfitting ili preagresivna regularizacija. 49.9% test uzoraka je OOV za player embeddings.

### WR_RNN_Attention_PlayerEmbed_v2.ipynb — Dijagnostika

Relaxirana regularizacija: dropout 0.25→0.15, noise 0.05→0.00, wd 5e-4→1e-4, patience 25→50.

| Model | MAE | RMSE | R2 | Best Epoch |
|---|---|---|---|---|
| **GRU_log1p_v2** | **18.29** | **25.28** | **0.3262** | 4 |
| GRU_sqrt_v2 | 18.45 | 25.37 | 0.3214 | 4 |
| LSTM_sqrt_v2 | 18.50 | 25.47 | 0.3165 | 4 |
| LSTM_log1p_v2 | 18.30 | 25.48 | 0.3158 | 4 |

**Dijagnoza:** Cak i sa relaxiranom regularizacijom, modeli staju na epoch 4. Zakljucak: **ARHITEKTURA JE PROBLEM**. Attention pooling + player embeddings nisu pogodni za ovaj dataset jer:
- Premalo utakmica po igracu da self-attention nauci smislene temporalne tezine
- 50% OOV u test setu cini embeddings beskorisnim za evaluaciju
- **Preporuka:** Ostati na standard LSTM last-hidden-state pristupu iz WR_RNN_Improved

---

## 9. Faza 6 — Napredne Tehnike

Ova faza obuhvata kvantilnu regresiju (opisanu u Fazi 4, MLP sekcija) i attention/embedding eksperimente (opisane u Fazi 5, RNN sekcija). Obe tehnike predstavljaju najnaprednije metode testirane u projektu.

**Kvantilna regresija (MLP Quantile)** — uspesna tehnika:
- Postize najnizi MAE od svih single modela (17.76)
- Pruza intervale poverenja sa 78.8% coverage
- Kljucna za fantasy/betting kontekst gde je raspon ishoda vazan

**Attention + Player Embeddings** — neuspesna za ovaj problem:
- Ne nadmasuje jednostavniji Improved RNN
- Ogranicena podacima (OOV problem, kratke karijere)

---

## 10. Faza 7 — Ceiling Analiza

### Motivacija

Da li je R2 ~ 0.33 plafon nasih modela ili plafon samog problema? Ceiling analiza koristi **centered rolling mean** kao zamenu za target, sto simulira "sta bi modeli postigli da nema single-game suma".

**Metodologija:** Treniramo sve modele na smoothed target-u (2, 3, 5, 7 game window) i poredimo rezultate. Ovo je **dijagnosticki eksperiment** — koristi buducnost i nije za produkciju.

### Rezultati po Window Sirinе (LSTM Improved, najbolji model)

| Window | Test MAE (smooth) | Test R2 (smooth) | Test MAE (raw) | Test R2 (raw) | Noise Std |
|---|---|---|---|---|---|
| Raw target | — | — | 18.03 | 0.3354 | — |
| 2-game | 12.10 | 0.5840 | 18.48 | 0.2993 | 17.89 |
| 3-game | 11.12 | 0.6018 | 18.68 | 0.3114 | 20.76 |
| 5-game | 8.85 | 0.7119 | 19.17 | 0.2988 | 23.06 |
| 7-game | 7.73 | **0.7626** | 19.41 | 0.2911 | — |

**Kljucni zakljucci:**

1. **Modeli su vec blizu teorijskog limita.** Kad se ukloni single-game sum, R2 skace na 0.76+ i MAE pada na 7.7.
2. **Vs raw target, performansa je potpuno nepromenjena** (~18.0-19.4 MAE, ~0.29-0.34 R2) bez obzira na arhitekturu.
3. **Oracle floor** (smoothed vs raw ground truth): MAE = 11.97 (2-game) do 17.06 (7-game), sto pokazuje koliko suma postoji u samom zadatku.
4. **Spearman korelacija** vs smoothed: 0.73-0.85; vs raw: 0.52-0.56. Modeli dobro rangiraju igrace, ali sum sprecava tacne apsolutne predikcije.

### Verifikacija: No-Leakage Test (5-game)

`WR_5_game_Ceiling_Smoothed_NoLeakage.ipynb` replicira 5-game eksperiment sa per-split smoothingom (umesto globalnog). Rezultat: razlika < 0.01 R2. **Nema cross-split leakage-a u originalnoj metodologiji.**

### Sta ovo znaci za praksu

- Nasi modeli su dostigli fundamentalni limit za single-game predikciju
- Dalje poboljsanje zahteva **fundamentalno drugacije informacije** (insider data, real-time injury status, snap count projekcije)
- Za **multi-game predikcije** (sezonski prosek), nasi modeli bi postigli R2 > 0.70

---

## 11. Faza 8 — Feature Ablacija (TopN)

### WR_TopN_Models_top5.ipynb

Svih 8 finalnih modela trenirano samo sa top-5 features:
1. first_downs_roll5, 2. target_share_std_lag1, 3. targets_roll5, 4. air_yards_roll5, 5. pregame_total

| Model | Test MAE | Test R2 |
|---|---|---|
| MLP Quantile q50 | 17.87 | 0.2843 |
| MLP Hybrid | 17.99 | 0.3150 |
| LSTM Improved | 18.11 | 0.3216 |

### WR_TopN_Models_top10.ipynb

Top-10 features (top-5 + avg_yards_last_season, avg_start_yardline_roll5, wind_mph, target_share_std_roll5, air_yard_share_roll5):

| Model | Test MAE | Test R2 |
|---|---|---|
| MLP Quantile q50 | 17.78 | 0.2954 |
| RandomForest | 17.90 | 0.3247 |
| MLP Hybrid | 17.92 | 0.3234 |

### Diminishing Returns

| Feature Set | MLP Quantile MAE | Delta |
|---|---|---|
| Top-5 | 17.87 | — |
| Top-10 | 17.78 | -0.09 |
| Top-40 | 17.75 | -0.03 |

**Zakljucak:** Top-5 features vec nosi ogromnu vecinu signala (MAE degradacija samo +0.12 vs top-40). Poboljsanje od 10→40 je samo 0.03 MAE. Ovo potvrduje da su nasi feature importance rezultati tacni.

---

## 12. Faza 9 — Finalni Modeli i Ensemble

### WR_Final_Models.ipynb

Cist referentni notebook — svi finalni modeli sa fiksiranim hiperparametrima, bez eksperimentisanja.

| Model | Test MAE | Test R2 |
|---|---|---|
| **MLP Quantile q50** | **17.75** | 0.3069 |
| RandomForest | 17.85 | 0.3321 |
| MLP Hybrid | 18.03 | 0.3217 |
| XGBoost | 18.09 | 0.3148 |
| ElasticNet | 18.10 | 0.3192 |
| LightGBM | 18.11 | 0.3061 |
| BiGRU + Attention | 18.16 | 0.3181 |
| BiLSTM + Attention | 18.27 | 0.3107 |

### WR_Ensemble_Final.ipynb — Ensemble Strategije

**5 base modela:** XGBoost, LightGBM, ElasticNet, MLP Hybrid, MLP Quantile

**Residual korelaciona matrica:**

|  | XGB | LGB | EN | MLP H | MLP Q |
|---|---|---|---|---|---|
| XGB | 1.000 | 0.987 | 0.964 | 0.970 | 0.973 |
| LGB | 0.987 | 1.000 | 0.964 | 0.968 | 0.970 |
| EN | 0.964 | 0.964 | 1.000 | 0.975 | 0.978 |
| MLP H | 0.970 | 0.968 | 0.975 | 1.000 | 0.993 |
| MLP Q | 0.973 | 0.970 | 0.978 | 0.993 | 1.000 |

Visoka korelacija (>0.96) ogranicava ensemble gain — svi modeli prave slicne greske.

**Opcija 1 (MLP H + MLP Q + LGB + ElasticNet):**

| Strategija | Test MAE | Test R2 | Tezine |
|---|---|---|---|
| simple_mean | 17.798 | 0.3284 | po 0.250 |
| inverse_mae | 17.796 | 0.3283 | MLP Q=0.254 |
| **constrained_ls** | **17.708** | 0.3184 | MLP Q=**0.704**, LGB=0.192 |
| ridge_stack | 18.147 | 0.3321 | meta-learner |

**Opcija 2 (XGB + LGB + MLP Hybrid):**

| Strategija | Test MAE | Test R2 |
|---|---|---|
| constrained_ls | 17.921 | 0.3262 |

**Head-to-Head: Novi vs Stari Ensemble:**

| Ensemble | Test MAE | Test R2 |
|---|---|---|
| Opt1 / constrained_ls (novi) | **17.708** | 0.3184 |
| Stari (XGB+LGB+MLP) | 18.050 | **0.3384** |

**Per-bin analiza:**

| Opseg (yards) | N | MLP Q (best base) | Ensemble (best) | Delta |
|---|---|---|---|---|
| 0-30 | 3,837 | **11.24** | 11.57 | -0.33 |
| 30-60 | 1,359 | 18.21 | **17.70** | +0.51 |
| 60-100 | 677 | 33.96 | **33.06** | +0.90 |
| 100-300 | 239 | 73.74 | **72.53** | +1.21 |

**Zakljucak:** Ensemble pomaze za igrace sa >30 yards. Za niske outpute, MLP Quantile je bolji sam. Novi ensemble ima rekordno nizak MAE ali nizi R2. Simple mean daje najbolji balans.

---

## 13. Kompletna Tabela Rezultata

### Svi Modeli — Sortirano po MAE

| # | Model | Notebook | Features | Test MAE | Test RMSE | Test R2 |
|---|---|---|---|---|---|---|
| 1 | **Ensemble Opt1/constrained_ls** | Ensemble_Final | 40 | **17.708** | 25.430 | 0.3184 |
| 2 | **MLP Quantile q50 (refined)** | Ensemble_Final | 40 | **17.750** | 25.644 | 0.3069 |
| 3 | MLP Quantile q50 (standalone) | MLP_Quantile | 40 | 17.760 | 25.600 | 0.3093 |
| 4 | Ensemble Opt1/inverse_mae | Ensemble_Final | 40 | 17.796 | 25.244 | 0.3283 |
| 5 | Ensemble Opt1/simple_mean | Ensemble_Final | 40 | 17.798 | 25.243 | 0.3284 |
| 6 | Model D (Fixed Hybrid) | MLP_Hybrid | 40 | 17.830 | 25.670 | 0.3055 |
| 7 | RandomForest | Final_Models | 40 | 17.853 | — | 0.3321 |
| 8 | Model E (Optuna Hybrid) | MLP_Hybrid | 40 | 17.880 | 25.350 | 0.3225 |
| 9 | Model C (Improved MLP) | MLP_Comparison | 40 | 17.910 | 25.490 | 0.3151 |
| 10 | Ensemble Opt2/constrained_ls | Ensemble_Final | 40 | 17.921 | 25.285 | 0.3262 |
| 11 | RandomForest | Career_RNN_Optuna | 184 | 18.010 | 25.140 | 0.3338 |
| 12 | **LSTM Improved** | RNN_Improved | 33+20 | **18.030** | **25.110** | **0.3354** |
| 13 | MLP Hybrid (refined) | Ensemble_Final | 40 | 18.030 | 25.369 | 0.3217 |
| 14 | **Stari Ensemble (XGB+LGB+MLP)** | Career_RNN_Optuna | 184 | 18.050 | **25.050** | **0.3384** |
| 15 | XGBoost | Ensemble_Final | 40 | 18.090 | 25.498 | 0.3148 |
| 16 | Model A (Original MLP) | MLP_Comparison | 184 | 18.100 | 25.620 | 0.3079 |
| 17 | ElasticNet | Ensemble_Final | 40 | 18.100 | 25.416 | 0.3192 |
| 18 | LightGBM | Ensemble_Final | 40 | 18.110 | 25.659 | 0.3061 |
| 19 | GRU @ LSTM HPs | RNN_Improved_GRU | 33+20 | 18.140 | 25.130 | 0.3344 |
| 20 | RNN v2 (padded) | RNN_v2 | 45+20 | 18.130 | 25.270 | 0.3268 |
| 21 | XGBoost (184) | Career_RNN_Optuna | 184 | 18.150 | 25.150 | 0.3335 |
| 22 | LightGBM (184) | Career_RNN_Optuna | 184 | 18.150 | 25.130 | 0.3346 |
| 23 | MLP (Optuna) | Career_RNN_Optuna | 184 | 18.160 | 25.480 | 0.3156 |
| 24 | Model B (Reduced MLP) | MLP_Comparison | 40 | 18.190 | 25.230 | 0.3289 |
| 25 | GRU_log1p_v2 | Attention_v2 | 45+20 | 18.290 | 25.280 | 0.3262 |
| 26 | GRU_sqrt (Attention) | Attention | 45+20 | 18.400 | 25.330 | 0.3236 |
| 27 | TCN (Dilated Conv1D) | Career_RNN_Optuna | 184 | 19.300 | 27.590 | 0.2453 |
| 28 | Original RNN (Optuna) | Career_RNN_Optuna | 184 | 19.730 | 27.930 | 0.2157 |
| 29 | GRU Baseline | GRU_Model | 95 | 20.310 | 28.540 | 0.2590 |

### Pobednici po Metrici

| Metrika | Pobednik | Vrednost |
|---|---|---|
| **Najnizi MAE** | Ensemble Opt1/constrained_ls | **17.708** |
| **Najvisi R2** | Stari Ensemble (XGB+LGB+MLP) | **0.3384** |
| **Najnizi RMSE** | Stari Ensemble (XGB+LGB+MLP) | **25.05** |
| **Najbolji single model (MAE)** | MLP Quantile q50 | **17.75** |
| **Najbolji RNN** | LSTM Improved | R2 = **0.3354** |
| **Najbrzi** | ElasticNet | 1s |

---

## 14. Kljucni Zakljucci i Naucene Lekcije

### 14.1 Sta Radi

1. **Feature redukcija je kljucna za NN.** 184→40 obelezja daje +0.021 R2 za MLP. Tree modeli to ne trebaju.

2. **Padding + Masking sacuvava sve podatke.** Improved RNN koristi svih 29,276 train uzoraka umesto 12,709-21,943 kad se kratke karijere filtriraju.

3. **Raw features za RNN, engineered za MLP.** RNN uci temporalne obrasce bolje iz sirovih podataka. MLP-u trebaju eksplicitni lag/roll features.

4. **Jednostavni RNN pobeduje kompleksne.** 1-layer LSTM(128) > 4-layer BiGRU(512) + Attention + Embeddings.

5. **Kvantilna regresija daje najnizi MAE.** Medijan (q50) je L1-optimalan i robustan na outliere. Plus besplatni intervali poverenja.

6. **AdamW sa pravim weight decay** je dramaticno bolji od obicnog Adam-a.

7. **Forsiran LayerNorm** stabilizuje trening — Optuna inace bira 'none' i dobije losiji rezultat.

8. **Huber loss sa searched delta** je bolji od cistog MSE ili MAE.

### 14.2 Sta Ne Radi / Zamke

1. **Attention + Player Embeddings ne pomazu** za ovaj problem. Premalo podataka po igracu, 50% OOV u testu.

2. **GaussianNoise + Masking su NEKOMPATIBILNI.** Sum razbija padded nule, Masking detekcija ne radi.

3. **Mixup + padding su nekompatibilni.** Mesanje padded i real pozicija kvari podatke.

4. **Preagresivna regularizacija ubija RNN.** Underfit (val < train loss) je podmukliji od overfit-a.

5. **Target u seq_feature_cols je opasan bug.** Uvek sacuvati neskaliranu kopiju: `df['receiving_yards_orig']`.

6. **Dugi seq_len sa padding-om je iluzija.** Ako je 60%+ timestep-ova padded, RNN "vidi" samo par real step-ova.

### 14.3 Trade-off: R2 vs MAE

Jasna tenzija:
- **Veci kapacitet** → bolji R2 (Model B, R2=0.3289)
- **Jaca regularizacija** → bolji MAE (Quantile, MAE=17.76)
- **Constrained LS ensemble** → rekordno nizak MAE (17.708) ali nizi R2 (0.3184)

**Preporuka:**
- Za **rangiranje igraca** → R2 fokus → Stari ensemble ili Opt1/simple_mean
- Za **tacnu procenu yards** → MAE fokus → MLP Quantile ili Opt1/constrained_ls

### 14.4 Teorijski Plafon

Ceiling analiza definitivno dokazuje: R2 ~ 0.33-0.37 je **fundamentalni limit** za single-game predikciju. Modeli vec ekstrahuju gotovo sav dostupan signal. Preostali "gap" je ireducibilni single-game sum (std ~ 17-23 yards).

---

## 15. Tehnicke Napomene

### Regularizacione Tehnike

| Tehnika | Konfiguracija | Gde |
|---|---|---|
| Dropout | 0.2-0.5 | Sve NN |
| LayerNorm | Forsiran | Svi moderni NN |
| GaussianNoise | stddev 0.05-0.25 | MLP i static grana RNN |
| Mixup | alpha 0.1-0.4 | MLP Comparison/Hybrid |
| Weight Decay (AdamW) | 1e-4 do 5e-3 | Sve NN |
| L2 na Embeddings | 1e-4 | RNN Attention |
| Early Stopping | patience 12-30 | Sve NN |
| ReduceLROnPlateau | factor=0.3, patience=5-10 | Vecina NN |

### Loss Funkcije

| Loss | Modeli |
|---|---|
| MSE | Rani (GRU baseline) |
| Huber (delta searched) | Vecina modernih NN |
| Pinball (multi-quantile) | MLP Quantile |

### Okruzenje

- Python 3.x, TensorFlow 2.21.0
- XGBoost, LightGBM, scikit-learn
- Optuna za hyperparameter optimization
- Pandas, NumPy za data processing
- Matplotlib, Seaborn za vizualizacije

---

## 16. Dalje Mogucnosti

1. **Multi-task learning** — predvidjanje targets + receptions + yards istovremeno
2. **Transformer arhitekture** — moguce poboljsanje nad BiGRU za sekvencni aspekt
3. **Vise player-level info** — age curve, injury history, snap count %
4. **Ensemble diversity** — razliciti feature sets ili loss funkcije za manje korelisane modele
5. **Residual learning** — MLP trenira na rezidualima LightGBM-a
6. **Seed ensemble** — ista arhitektura sa 5 seedova i prosecavanje (+0.005-0.015 R2)
7. **Multi-game predikcije** — sezonski prosek gde bi R2 > 0.70

---

*Dokumentacija pokriva 22 WR_ notebooka razvijenih tokom projekta. Generisano 2026-04-17.*
