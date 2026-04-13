# NFL Wide Receiver — Kompletna Analiza i Rezultati

> **Cilj projekta:** Predvidjanje broja *receiving yards* NFL Wide Receiver igraca u sledecoj utakmici na osnovu istorijskih statistika, karijernih obelezja i pregame informacija (sezone 2015–2025).

---

## Sadrzaj

1. [Pregled projekta](#1-pregled-projekta)
2. [Podaci i inzenjerstvo obelezja](#2-podaci-i-inzenjerstvo-obelezja)
3. [Strategija podele podataka](#3-strategija-podele-podataka)
4. [Razvojni put — hronologija notebooka](#4-razvojni-put--hronologija-notebooka)
5. [Rezultati svih modela](#5-rezultati-svih-modela)
6. [Finalni Ensemble Notebook](#6-finalni-ensemble-notebook)
7. [Kljucni zakljucci i lekcije](#7-kljucni-zakljucci-i-lekcije)
8. [Tehnicke napomene](#8-tehnicke-napomene)

---

## 1. Pregled projekta

### Problem i izazov

NFL Wide Receiver statistike su inherentno volatilne — igraci mogu imati "boom" nedelju (100+ yards) ili "bust" (0–20 yards) bez jasnog obrasca. Zbog toga postoji **teorijski plafon** za R² od otprilike **0.33–0.37** na ovom problemu.

### Dataset

| Parametar | Vrednost |
|---|---|
| Izvor | `wr_all_weeks.csv` |
| Broj zapisa | 46 115 game-level zapisa |
| Sezone | 2015–2025 (11 sezona) |
| Originalni broj kolona | 99 |
| Finalni broj obelezja (posle FE) | 184 |
| Odabrana obelezja (top-40) | 40 |

---

## 2. Podaci i inzenjerstvo obelezja

### 2.1 Bazne statistike

- `receiving_yards`, `targets`, `receptions`, `epa`, `target_share`, `air_yards_share`
- Pregame: `spread`, `total`, `home_away`, `opponent_def_rank`

### 2.2 Temporalne transformacije (po svakom obelezju)

| Tip | Opis |
|---|---|
| `_lag1` | Vrednost iz prethodne utakmice |
| `_roll3` | Rolling prosek — poslednje 3 utakmice |
| `_roll5` | Rolling prosek — poslednje 5 utakmica |
| `_career_avg` | Karijerni prosek (expanding mean) |
| `_expanding_std` | Karijerna standardna devijacija |

### 2.3 Interakciona obelezja

- `target_share × pregame_total` — sinergija volumena i ocekivanog tempa
- `weeks_since_last_game` — pauze, povrede, bye week
- `is_new_season` — flag za pocetak nove sezone (koriscen u career-based sekvencama)

### 2.4 Matchup obelezja (bez data leakage-a)

- Expanding mean protivnikove defanzive (samo prosli podaci)
- Defanzivni EPA pro, yards allowed per game pro

### 2.5 Feature Selection

Notebook `WR_Feature_Selection.ipynb`:
- Trenirati XGBoost + LightGBM na svih 184 obelezja
- Kombinovane importance ocene normalizovane
- **Top 40 obelezja pokrivaju 51.8% kumulativne vaznosti**
- 144 od 184 obelezja su suštinski šum za neuralne mreže
- Rezultat sacuvan u `selected_features_top40.json`

**Najvaznije kategorije obelezja:**
1. `receiving_yards` lag/roll varijante
2. `target_share` metrike
3. `epa` i derivati
4. Pregame informacije (`spread`, `total`)
5. Karijerni proseci igraca

---

## 3. Strategija podele podataka

Striktna **temporalna** podela bez ikakvog data leakage-a:

| Skup | Sezone | Velicina | Svrha |
|---|---|---|---|
| **Train** | 2015–2021 | 29 276 zapisa | Treniranje modela |
| **Validation** | 2022–2023 | 8 921 zapisa | Hyperparameter tuning, early stopping |
| **Test** | 2024–2025 | 6 199 zapisa | Finalna evaluacija (nikad koriscen za odluke) |

**StandardScaler** fitovan iskljucivo na train skupu.

**Sample weights:** `1 + strength × √(y / mean_y)`, strength=0.6, opseg `[1.00, 2.87]`
- Vece utakmice (vise yards) dobijaju veci weight
- Sprecat model da ignoriše "boom" igrace

**Target transformacija:** `√(receiving_yards)` — koriscena u vecini modela
- Alternativa: `log(1 + receiving_yards)` — bolji za RNN modele

---

## 4. Razvojni put — hronologija notebooka

### Faza 0: Bazni GRU (`WR_GRU_Model.ipynb`)

Polazna tacka — jednostavan GRU sa sliding window pristupom.

**Arhitektura:**
```
Sequence input (5 timesteps × 95 features)
→ GRU(32 units)
→ Dense(1, linear)
```

**Rezultati:**

| Metrika | Vrednost |
|---|---|
| Test MAE | 20.31 yards |
| Test RMSE | 28.54 yards |
| Test R² | 0.2590 |
| Parametri | 12 417 |

**Zakljucak:** Jednostavan baseline, sluzi kao referentna tacka.

---

### Faza 1: Kompleksni modeli sa Optuna (`WR_Career_RNN_Optuna.ipynb`)

Sveobuhvatna eksploracija — tree modeli, MLP, RNN, TCN sa Optuna optimizacijom.

**Priprema podataka:**
- 184 inzenjerisana obelezja
- Career-based sekvence (prelaze granice sezona)
- `weeks_since_last_game` tracking
- Optuna TPE sampler + MedianPruner (100 trials po modelu)

**Rezultati tree modela (grid search):**

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| **RandomForest** | **18.01** | 25.14 | 0.3338 |
| **LightGBM** | 18.15 | **25.13** | **0.3346** |
| XGBoost | 18.15 | 25.15 | 0.3335 |

**Rezultati neuralnih modela (Optuna, 100 trials):**

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| MLP (Optuna) | 18.35 | 25.80 | 0.3156 |
| TCN (Dilated Conv1D) | 19.30 | 27.59 | 0.2453 |
| RNN (LSTM/GRU) | 19.73 | 27.93 | 0.2157 |

**Ensemble (XGB + LGB + MLP, optimized weights):**

| Test MAE | Test RMSE | Test R² |
|---|---|---|
| **18.05** | **25.05** | **0.3384** |

**Kljucni zakljucak:** Tree modeli drasticno nadmasuju neuralne mreze. Razlika R² = 0.12 (LightGBM vs RNN). Ovo je pokrenulo dublje istrazivanje problema sa RNN arhitekturama.

---

### Faza 2: Feature Selection (`WR_Feature_Selection.ipynb`)

Identifikovano: **144 od 184 obelezja su sum za neuralne mreze**.
Top-40 obelezja sacuvana za koriscenje u svim kasnijim modelima.

---

### Faza 3: MLP Varijante (`WR_MLP_Comparison.ipynb`)

Sistematicno poredjenje uticaja broja obelezja i regularizacije.

**Arhitektura Model A i B:** `[448 → 128 → 320 → 448 → 256]`, LayerNorm, Dropout(0.5)
**Arhitektura Model C:** Kompaktniji (max 3 sloja, max 256 jed.), GaussianNoise + Mixup

| Model | Obelezja | Test MAE | Test RMSE | Test R² | Napomena |
|---|---|---|---|---|---|
| A — Original MLP | 184 | 18.10 | 25.62 | 0.3079 | Referentni |
| **B — Ista arhitektura** | **40** | **18.19** | **25.23** | **0.3289** | Najbolji R² |
| C — Pobolj. regularizacija | 40 | **17.91** | 25.49 | 0.3151 | Najbolji MAE |

**Zakljucak:** Smanjenje obelezja 184→40 poboljsava R² za +0.021. Trade-off: veci kapacitet = bolji R²; jaca regularizacija = bolji MAE.

---

### Faza 4: MLP Hybrid (`WR_MLP_Hybrid.ipynb`)

Kombinovanje visokog kapaciteta sa jakom regularizacijom.

**Model D — Fixed Hybrid:**
- Bazira se na Model B arhitekturi + `GaussianNoise(0.15)` + Mixup(α=0.2)

**Model E — Optuna Hybrid:**
- Optuna pretraga: 3–5 slojeva, 128–448 jed., noise [0.05–0.25], mixup α [0.1–0.4]
- Optimalna konfiguracija: 4 sloja `[320, 384, 192, 128]`, noise=0.25, mixup=0.2
- LR=0.000245, weight decay=0.00021, batch=32, best epoch ~25

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| D — Fixed Hybrid | 17.83 | 25.67 | 0.3055 |
| E — Optuna Hybrid | 17.88 | 25.35 | 0.3225 |

---

### Faza 5: Quantile Regresija (`WR_MLP_Quantile.ipynb`)

Predvidjanje intervala poverenja kroz multi-kvantilnu regresiju.

**Arhitektura:** Ista kao Model B, ali sa `Dense(3)` izlaznim slojem za q10, q50, q90.
**Loss:** Multi-kvantilna pinball loss.

**Hyperparametri:**
- LR: 0.0037
- Weight decay: 1.5e-5
- Dropout: 0.5

| Varijanta | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| MLP-B (point estimate) | 18.19 | 25.23 | 0.3289 |
| **MLP Quantile q50** | **17.76** | 25.60 | 0.3093 |

**Kljucni zakljucak:** Kvantilna regresija postize **najnizi MAE od svih MLP modela (17.76)**. q50 (medijan) je robustni na outliere. Pored tacnog predvidjanja, model daje i interval poverenja (q10, q90).

---

### Faza 6: RNN sa Paznjom (`WR_RNN_Attention_PlayerEmbed_v2.ipynb`)

Najnaprednija sekvencna arhitektura sa tri input grane.

**Arhitektura — Tri-input dizajn:**

```
Sequence Branch:
  seq_input (T × 45 raw features)
  → Masking
  → BiRNN (LSTM ili GRU)
  → AttentionPool (custom layer)
  → LayerNorm → Dropout

Player Embedding Branch:
  pid_input → Embedding(n_players, 8, L2=1e-4) → Flatten

Static Features Branch:
  static_input (20 pregame features)
  → GaussianNoise → Dense → LayerNorm → Dropout

  ↓ Concatenate sve tri grane ↓
  → Dense(96) → LayerNorm → Dropout → Dense(1)
```

**Custom AttentionPool sloj:**
```python
class AttentionPool(layers.Layer):
    # Uci da dodeli tezine svim timestep-ovima
    # Umesto da koristi samo poslednji hidden state
    score = Dense(1)
    weights = softmax(scores)
    output = reduce_sum(features * weights)
```

**Sekvencne karakteristike:**
- 45 raw feature kanala po timestep-u
- 6 baznih statistika × (originalna + lag1 + roll3) = `yards, targets, receptions, EPA, target_share, air_yards_share`
- Padding + Masking za varijabilne karijerne duzine
- Bez ogranicenja duzine sekvence

**Cetiri varijante modela:**

| Model | RNN tip | Target transform | Test MAE | Test RMSE | Test R² | Vreme treninga |
|---|---|---|---|---|---|---|
| LSTM_sqrt_v2 | BiLSTM | √yards | 18.50 | 25.47 | 0.3165 | 231s |
| LSTM_log1p_v2 | BiLSTM | log(1+y) | 18.30 | 25.48 | 0.3158 | 230s |
| GRU_sqrt_v2 | BiGRU | √yards | 18.45 | 25.37 | 0.3214 | 201s |
| **GRU_log1p_v2** | **BiGRU** | **log(1+y)** | **18.29** | **25.28** | **0.3262** | 202s |

**Poboljsanje u odnosu na v1:** Relaxovana regularizacija (visi LR, manji dropout) povecala R² sa 0.3158 → 0.3262.

**Zakljucak:** GRU sa log1p transformacijom i pažnjom je **najbolji RNN model**. Attention pool + player embeddings donose znacajno poboljsanje u odnosu na standard GRU.

---

## 5. Rezultati svih modela

### Kompletna tabela svih testiranih modela

| Notebook | Model | Arhitektura | Obelezja | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| WR_GRU_Model | GRU Baseline | GRU(32) → Dense(1) | 95 | 20.31 | 28.54 | 0.2590 |
| WR_Career_RNN_Optuna | RNN (Optuna) | LSTM/GRU multi-layer | 184 | 19.73 | 27.93 | 0.2157 |
| WR_Career_RNN_Optuna | TCN | Dilated Conv1D | 184 | 19.30 | 27.59 | 0.2453 |
| WR_Career_RNN_Optuna | MLP (Optuna) | 5-slojni MLP | 184 | 18.35 | 25.80 | 0.3156 |
| WR_MLP_Comparison | Model A | [448,128,320,448,256] | 184 | 18.10 | 25.62 | 0.3079 |
| WR_Career_RNN_Optuna | XGBoost | Gradient boosted trees | 184 | 18.15 | 25.15 | 0.3335 |
| WR_Career_RNN_Optuna | LightGBM | Alt tree impl. | 184 | 18.15 | 25.13 | 0.3346 |
| WR_Career_RNN_Optuna | RandomForest | Tree ensemble | 184 | 18.01 | 25.14 | 0.3338 |
| WR_Ensemble_Final | XGBoost (refined, top-40) | Gradient boosted trees | 40 | 18.09 | 25.498 | 0.3148 |
| WR_Ensemble_Final | LightGBM (refined, top-40) | Alt tree impl. | 40 | 18.11 | 25.659 | 0.3061 |
| WR_Ensemble_Final | ElasticNet (top-40) | Linearni model | 40 | 18.10 | 25.416 | 0.3192 |
| WR_Ensemble_Final | MLP Hybrid (refined) | [448,128,320,448,256]+Noise | 40 | 18.03 | 25.369 | 0.3217 |
| WR_MLP_Comparison | Model B | [448,128,320,448,256] | 40 | 18.19 | 25.23 | 0.3289 |
| WR_MLP_Hybrid | Model E (Optuna) | 4-slojni opt. | 40 | 17.88 | 25.35 | 0.3225 |
| WR_MLP_Hybrid | Model D (Fixed) | [448,128,320,448,256]+Mixup | 40 | 17.83 | 25.67 | 0.3055 |
| WR_MLP_Comparison | Model C | Kompaktan+reg. | 40 | 17.91 | 25.49 | 0.3151 |
| WR_RNN_Attention_PlayerEmbed_v2 | GRU_sqrt_v2 | BiGRU+Attention+Embed | 45 | 18.45 | 25.37 | 0.3214 |
| WR_RNN_Attention_PlayerEmbed_v2 | LSTM_log1p_v2 | BiLSTM+Attention+Embed | 45 | 18.30 | 25.48 | 0.3158 |
| WR_RNN_Attention_PlayerEmbed_v2 | LSTM_sqrt_v2 | BiLSTM+Attention+Embed | 45 | 18.50 | 25.47 | 0.3165 |
| WR_RNN_Attention_PlayerEmbed_v2 | GRU_log1p_v2 | BiGRU+Attention+Embed | 45 | 18.29 | 25.28 | 0.3262 |
| WR_MLP_Quantile | MLP Quantile q50 (standalone) | 3-izlazna glava | 40 | 17.76 | 25.60 | 0.3093 |
| WR_Ensemble_Final | **MLP Quantile q50 (refined)** | 3-izlazna glava | 40 | **17.75** | 25.644 | 0.3069 |
| WR_Career_RNN_Optuna | Stari Ensemble (XGB+LGB+MLP) | Weighted combo | 184 | 18.05 | 25.05 | **0.3384** |
| WR_Ensemble_Final | Opt2 / constrained_ls | XGB+LGB+MLP Hybrid | 40 | 17.921 | 25.285 | 0.3262 |
| WR_Ensemble_Final | Opt2 / simple_mean | XGB+LGB+MLP Hybrid | 40 | 17.927 | 25.295 | 0.3256 |
| WR_Ensemble_Final | Opt1 / simple_mean | MLP H+MLP Q+LGB+EN | 40 | 17.798 | 25.243 | 0.3284 |
| WR_Ensemble_Final | Opt1 / inverse_mae | MLP H+MLP Q+LGB+EN | 40 | 17.796 | 25.244 | 0.3283 |
| WR_Ensemble_Final | **Opt1 / constrained_ls** | MLP H+MLP Q+LGB+EN | 40 | **17.708** | 25.430 | 0.3184 |

### Pobednici po metrici

| Metrika | Pobednik | Vrednost |
|---|---|---|
| **Najnizi MAE** | Ensemble Opt1 / constrained_ls | **17.708** |
| **Najvisi R²** | Stari Ensemble (XGB+LGB+MLP) | **0.3384** |
| **Najnizi RMSE** | Stari Ensemble (XGB+LGB+MLP) | **25.05** |
| **Najbrzi trening** | ElasticNet | 1s |
| **Najnizi MAE — single model** | MLP Quantile q50 (refined) | **17.75** |
| **Najbolji NN tabular (R²)** | MLP Hybrid (refined) | R²=0.3217 |
| **Najboljji sekvencni** | GRU_log1p_v2 | R²=0.3262 |

---

## 6. Finalni Ensemble Notebook

### `WR_Ensemble_Final.ipynb` — Kompletni rezultati

Ovaj notebook konsoliduje sve rezultate i testira razlicite ensemble strategije sa uniformnim trening receptom.

**Uniformni trening recept (primenjen na sve modele):**
- Top-40 obelezja iz `selected_features_top40.json`
- `√(receiving_yards)` target transformacija
- Sample weights strength=0.6 → opseg `[1.00, 2.87]`
- Temporal split 2015-21 / 2022-23 / 2024-25

### Base modeli — rezultati

| Model | Val MAE | Test MAE | Test RMSE | Test R² | Napomena |
|---|---|---|---|---|---|
| XGBoost | 18.17 | 18.09 | 25.498 | 0.3148 | 8s |
| LightGBM | 18.20 | 18.11 | 25.659 | 0.3061 | 11s |
| ElasticNet | 18.33 | 18.10 | 25.416 | 0.3192 | 1s |
| MLP Hybrid (Huber) | 18.25 | 18.03 | 25.369 | 0.3217 | MODEL_B_PARAMS + GaussianNoise(0.15) |
| **MLP Quantile q50** | **17.91** | **17.75** | 25.644 | 0.3069 | Pinball loss, best_ep=35 |

### Residual korelaciona matrica

Svi modeli su **visoko korelisani** (r = 0.964–0.993):

|  | XGBoost | LightGBM | ElasticNet | MLP Hybrid | MLP Quantile |
|---|---|---|---|---|---|
| XGBoost | 1.000 | 0.987 | 0.964 | 0.970 | 0.973 |
| LightGBM | 0.987 | 1.000 | 0.964 | 0.968 | 0.970 |
| ElasticNet | 0.964 | 0.964 | 1.000 | 0.975 | 0.978 |
| MLP Hybrid | 0.970 | 0.968 | 0.975 | 1.000 | 0.993 |
| MLP Quantile | 0.973 | 0.970 | 0.978 | 0.993 | 1.000 |

**Zakljucak:** Visoka korelacija (>0.96) ogranicava ensemble gain — svi modeli prave slicne greske na istim primerima.

### Ensemble rezultati — Opcija 1

**Sastav:** `{MLP Hybrid, MLP Quantile, LightGBM, ElasticNet}`

| Strategija | Test MAE | Test RMSE | Test R² | Tezine |
|---|---|---|---|---|
| simple_mean | 17.798 | 25.243 | 0.3284 | svaki po 0.250 |
| inverse_mae | 17.796 | 25.244 | 0.3283 | MLP Q=0.254, ostali ~0.249 |
| **constrained_ls** | **17.708** | 25.430 | 0.3184 | MLP Q=**0.704**, LGB=0.192, EN=0.104, MLP H=0 |
| ridge_stack | 18.147 | 25.173 | 0.3321 | meta-learner |

### Ensemble rezultati — Opcija 2

**Sastav:** `{XGBoost, LightGBM, MLP Hybrid}` — isti kao stari ensemble

| Strategija | Test MAE | Test RMSE | Test R² | Tezine |
|---|---|---|---|---|
| simple_mean | 17.927 | 25.295 | 0.3256 | svaki po 0.333 |
| inverse_mae | 17.927 | 25.295 | 0.3256 | gotovo jednake |
| constrained_ls | 17.921 | 25.285 | 0.3262 | XGB=0.307, LGB=0.327, MLP H=0.366 |
| ridge_stack | 18.220 | 25.213 | 0.3300 | meta-learner |

### Per-bin MAE (po kategorijama yards)

| Opseg (yards) | N uzoraka | MLP Quantile (best base) | Opt1/constrained_ls (best ensemble) | Delta |
|---|---|---|---|---|
| 0–30 | 3 837 | 11.24 | 11.57 | **-0.33** (ensemble losiji!) |
| 30–60 | 1 359 | 18.21 | 17.70 | +0.51 |
| 60–100 | 677 | 33.96 | 33.06 | +0.90 |
| 100–300 | 239 | 73.74 | 72.53 | +1.21 |

**Vazna napomena:** Ensemble pomaze samo za igrace sa >30 yards. Za niske/nulte outpute, MLP Quantile je bolji sam.

### Head-to-Head: Novi vs Stari Ensemble

| Ensemble | Sastav | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| **Opt1 / constrained_ls** | MLP H+MLP Q+LGB+EN | **17.708** | 25.430 | 0.3184 |
| **Opt2 / constrained_ls** | XGB+LGB+MLP Hybrid | 17.921 | 25.285 | 0.3262 |
| **Stari ensemble (referenca)** | XGB+LGB+MLP (old) | 18.050 | **25.050** | **0.3384** |

**Razlika novih vs starog:**
- Opt1 best: dMAE = **-0.342** (bolji), dRMSE = +0.380 (losiji), dR² = -0.020 (losiji)
- Opt2 best: dMAE = **-0.129** (bolji), dRMSE = +0.235 (losiji), dR² = -0.012 (losiji)

**Kljucni uvid:** Novi ensemble dostiže **rekordno nizak MAE (17.708)**, ali stari ensemble zadrzava **rekordno visok R² (0.3384)** i najnizi RMSE. Ovo je direktna posledica constrained_ls optimizacije — maximalno tezvanje MLP Quantile (0.704) optimizuje MAE na ustrb R².

### Sacuvani fajlovi

```
results/ensemble_final_results.json
results/ensemble_final_comparison.csv
results/ensemble_final_h2h.csv
results/ensemble_final_comparison.png
results/ensemble_final_residual_corr.png
```

---

## 7. Kljucni zakljucci i lekcije

### 7.1 Tree modeli vs Neuralne mreze

Tree modeli (LightGBM, XGBoost) konzistentno nadmasuju standardne neuralne mreze:
- LightGBM: R²=0.3346 vs RNN (Optuna): R²=0.2157 — razlika od **+0.119**

Razlozi:
1. Neuralne mreze "pate" od sumnih obelezja (184 feat → 40 feat znacajno pomaže)
2. Tree modeli su robustniji na iregularne distribucije (spikes u yards)
3. NFL statistike imaju nelinearne ali ne nuzno sekvencne obrasce

### 7.2 Feature reduction je kljucan za NN

| Konfiguracija | Test R² |
|---|---|
| MLP sa 184 obelezja | 0.3079 |
| MLP sa 40 obelezja | 0.3289 |
| **Poboljsanje** | **+0.021** |

### 7.3 Quantile regresija — niži MAE, intervali poverenja

MLP Quantile q50 postige najnizi MAE (17.76) jer:
- Medijan je optimalan za L1 loss (MAE)
- Robustnost na outliere (igrac koji ima 0 yards ili 150+ yards)
- Bonus: intervali poverenja (q10, q90) za risk procenu

### 7.4 Trade-off: R² vs MAE

Postoji jasni trade-off:
- Visoki kapacitet modela → bolji R² (Model B, R²=0.3289)
- Jaka regularizacija → bolji MAE (Model C/D/Quantile, MAE ≈ 17.76–17.91)

Za primenu: ako je cilj rangiranje igraca — preferi R². Ako je cilj tacna procena yards — preferi MAE.

### 7.5 Attention + Player embeddings poboljsavaju RNN

Prelaz sa standardnog GRU (R²=0.2590) na BiGRU + AttentionPool + PlayerEmbed (R²=0.3262):
- AttentionPool uci da vaznuje sve timestep-ove (ne samo poslednji)
- Player embeddings hvataju talenат varijans izmedju igraca
- Masking + padding cuva karijerne podatke bez gubitka

### 7.6 Ensemble — trade-off MAE vs R²

Finalni ensemble notebook otkrio je jasnu tenziju izmedju metrika:

| Konfiguracija | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Stari Ensemble (XGB+LGB+MLP) | 18.05 | **25.05** | **0.3384** |
| Opt1 / constrained_ls (novi) | **17.708** | 25.43 | 0.3184 |
| Opt1 / simple_mean (novi) | 17.798 | 25.24 | 0.3284 |

Constrained LS optimizuje val MAE agresivno → dodeljuje 70.4% tezine MLP Quantile-u koji dominira na toj metrici, ali na ustrb R² i RMSE. Simple mean daje uravnotezeni kompromis (MAE=17.798, R²=0.3284).

**Preporuka:**
- Za **rangiranje igraca** (ukupni kvalitet predvidjanja) → stari ensemble ili Opt1/simple_mean
- Za **tacnu procenu yards** (minimizacija greske) → Opt1/constrained_ls ili MLP Quantile standalone

### 7.7 Teorijski plafon

Maksimalni dostignutiR² = 0.3384, sto je blizu teorijskog plafona od ~0.33–0.37. Daljnje poboljsanje zahtevalo bi fundamentalno drugacije informacije (npr. detaljna insider matchup data, fizicko stanje igraca).

---

## 8. Tehnicke napomene

### Regularizacione tehnike koriscene

| Tehnika | Primena | Konfiguracija |
|---|---|---|
| Dropout | Sve NN | 0.2–0.5 |
| LayerNorm | MLP i RNN | Standardna |
| GaussianNoise | Input augmentacija | stddev 0.05–0.25 |
| Mixup | Trening augmentacija | α ∈ [0.1–0.4] |
| Weight decay (AdamW) | Sve NN | 1e-4 do 5e-3 |
| L2 na embeddings | Player embed | 1e-4 |
| Early stopping | Sve NN | patience 12–30 |
| ReduceLROnPlateau | Vecina NN | factor=0.3, patience=5–10 |

### Loss funkcije

| Loss | Modeli | Napomena |
|---|---|---|
| MAE | Rani modeli | Direktna optimizacija metrike |
| Huber | Vecina NN | δ ∈ [0.5, 2.0], balans MAE/MSE |
| Pinball | Quantile MLP | Per-kvantil asimetricna loss |

### Hyperparameter Optimization

| Alat | Koriscen za | Konfiguracija |
|---|---|---|
| **Optuna TPE** | Sve NN, tree modeli | 100 trials, MedianPruner (min 3 epohe) |
| Grid search | Tree modeli (baseline) | Standardna mreza |
| Keras Tuner | Rani eksperimenti | Zamenjeno Optuna-om |

### Sequence modeling pristupi

| Pristup | Notebook | Prednosti | Mane |
|---|---|---|---|
| Sliding window (fixed 5) | WR_GRU_Model | Jednostavan | Gubi karijerne podatke |
| Career-based (8–24) | WR_Career_RNN_Optuna | Vise konteksta | Sezonski prekidi |
| Padding + Masking | WR_RNN_Attention_v2 | Cuva sve podatke | Racunski skuplje |

### Okruzenje

- Python 3.x, TensorFlow 2.21.0
- XGBoost, LightGBM, scikit-learn
- Optuna za hyperparameter optimizaciju
- Pandas, NumPy za data processing
- Matplotlib, Seaborn za vizualizacije

---

## Napomene o daljem radu

1. **Testirati RNN na top-40 obelezjima** — da li ce RNN sa 40 obelezja (umesto 45 raw) premostiti preostali gap sa tree modelima
2. **Dodati vise player-level informacija** — age curve, injury history, snap count %
3. **Testirati transformer arhitekture** — moguce poboljsanje nad BiGRU+Attention za sekvencni aspekt
4. **Multi-task learning** — predvidjanje targets + receptions + yards istovremeno
5. **Istraziiti zasto su svi modeli visoko korelisani (r>0.96)** — moguci put ka boljem ensemble diversitetu kroz drugacije feature sets ili loss funkcije

---

*Dokumentacija obuhvata 11 notebooks razvijenih tokom projekta: WR_GRU_Model, WR_Feature_Selection, WR_MLP_Comparison, WR_MLP_Hybrid, WR_MLP_Quantile, WR_RNN_Improved, WR_RNN_v2, WR_RNN_Attention_PlayerEmbed, WR_RNN_Attention_PlayerEmbed_v2, WR_Career_RNN_Optuna, WR_Ensemble_Final.*
