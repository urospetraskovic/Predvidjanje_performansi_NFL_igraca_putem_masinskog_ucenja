# WR Receiving Yards — Proces razvoja modela

Dokumentacija celokupnog procesa predviđanja jarda NFL Wide Receiver-a po utakmici.
Pokriva sve eksperimente, tehnike, arhitekture i rezultate koje smo testirali kroz 11 notebook-a.

---

## Sadržaj

1. [Problem i cilj](#1-problem-i-cilj)
2. [Dataset i temporalni split](#2-dataset-i-temporalni-split)
3. [Lista svih notebook-a (vremenska linija)](#3-lista-svih-notebook-a-vremenska-linija)
4. [Faza 1 — Rano istraživanje](#faza-1--rano-istraživanje)
5. [Faza 2 — Objedinjeni Optuna pipeline](#faza-2--objedinjeni-optuna-pipeline)
6. [Faza 3 — Feature redukcija](#faza-3--feature-redukcija)
7. [Faza 4 — MLP poboljšanja](#faza-4--mlp-poboljšanja)
8. [Faza 5 — RNN redizajn](#faza-5--rnn-redizajn)
9. [Faza 6 — Napredne tehnike](#faza-6--napredne-tehnike)
10. [Pregled svih tehnika i alata](#10-pregled-svih-tehnika-i-alata)
11. [Rezultati — finalna tabela](#11-rezultati--finalna-tabela)
12. [Ključni zaključci i naučene lekcije](#12-ključni-zaključci-i-naučene-lekcije)

---

## 1. Problem i cilj

**Problem:** Predvideti koliko će jarda (`receiving_yards`) NFL Wide Receiver imati u sledećoj utakmici, koristeći istoriju utakmica i pregame informacije (protivnik, vreme, spread, total, itd.).

**Target distribucija:** Jaka desna asimetrija (heavy right tail). Većina utakmica: 0–80 jarda. Retki outlier-i: 150+ jarda. Oko 10–15% utakmica ima 0 jarda (igrač nije bio targetovan, povreda, bench).

**Glavni izazov:** Week-to-week varijansa WR-a je ogromna zbog koncepta "boom or bust" — teoretski maksimalni R² koji iko ikad postigne na ovom problemu je oko 0.33–0.37.

---

## 2. Dataset i temporalni split

**Izvor:** `data/fully combined/wr_all_weeks.csv` — agregirani play-by-play podaci za sve NFL WR-ove po utakmici, sezone 2015–2025.

**Temporalni split (strogi hronološki, isti u svim modernim notebook-ima):**

| Skup | Sezone | Svrha |
|---|---|---|
| **Train** | 2015–2021 | Treniranje modela |
| **Val** | 2022–2023 | Hyperparameter tuning + early stopping |
| **Test** | 2024–2025 | Finalna evaluacija (nikad se ne koristi za odluke) |

**Pravilo:** Sve rolling/lag featurese radimo sa `.shift(1)` — nikad ne koristimo podatke iz trenutne utakmice (ni za trening). StandardScaler se uvek fituje **samo na train** set.

---

## 3. Lista svih notebook-a (vremenska linija)

| # | Notebook | Faza | Svrha |
|---|---|---|---|
| 1 | [WR_Analysis.ipynb](../notebooks/WR_Analysis.ipynb) | Rano istraživanje | EDA + inicijalni feature engineering |
| 2 | [WR_NN.ipynb](../notebooks/WR_NN.ipynb) | Rano istraživanje | Prvi NN eksperiment (nepotpun, bez markdown-a) |
| 3 | [WR_GRU_Model.ipynb](../notebooks/WR_GRU_Model.ipynb) | Rano istraživanje | Baseline GRU sa sliding window 5 utakmica |
| 4 | [WR_Career_RNN_Optuna.ipynb](../notebooks/WR_Career_RNN_Optuna.ipynb) | Objedinjeni pipeline | Career-based sekvence, Optuna za MLP/RNN/TCN, ensemble |
| 5 | [WR_Feature_Selection.ipynb](../notebooks/WR_Feature_Selection.ipynb) | Feature redukcija | XGBoost+LightGBM importance → top 40 features |
| 6 | [WR_MLP_Comparison.ipynb](../notebooks/WR_MLP_Comparison.ipynb) | MLP poboljšanja | Modeli A / B / C — efekat redukcije + regularizacije |
| 7 | [WR_MLP_Hybrid.ipynb](../notebooks/WR_MLP_Hybrid.ipynb) | MLP poboljšanja | Modeli D / E — veliki kapacitet + Mixup/GaussianNoise |
| 8 | [WR_RNN_Improved.ipynb](../notebooks/WR_RNN_Improved.ipynb) | RNN redizajn | Dual-input, raw features, Masking, GaussianNoise |
| 9 | [WR_RNN_v2.ipynb](../notebooks/WR_RNN_v2.ipynb) | RNN redizajn | Bug fix (target scaling) + lag/roll kanali + no-mask baseline |
| 10 | [WR_RNN_Attention_PlayerEmbed.ipynb](../notebooks/WR_RNN_Attention_PlayerEmbed.ipynb) | Napredne tehnike | Attention pooling + Player embeddings, LSTM×GRU × sqrt×log1p |
| 11 | [WR_MLP_Quantile.ipynb](../notebooks/WR_MLP_Quantile.ipynb) | Napredne tehnike | Kvantilna regresija (q10/q50/q90) sa pinball loss |

---

## Faza 1 — Rano istraživanje

### [WR_Analysis.ipynb](../notebooks/WR_Analysis.ipynb)

Prvi notebook — EDA i postavka feature engineering pravila.

**Šta se radi:**
- Učitavanje i pregled WR dataseta (sve sezone 2015–2025)
- Analiza nedostajućih vrednosti i osnovna statistika
- EDA target varijable `receiving_yards` — histogrami, distribucija po sezoni
- Inicijalni feature engineering:
  - **FE-1**: sortiranje po igraču + playoff flag
  - **FE-2**: Defanzivni matchup — expanding mean jarda koje je WR skupio protiv svake defanze (expanding, ne rolling, da se izbegne leakage)
  - **FE-3**: Rolling/lag featuri po igraču — `lag_1, lag_2, lag_3`, `roll3_mean, roll5_mean, roll8_mean`, `roll3_std, roll5_std`
- Filtriranje igrača sa premalim brojem utakmica
- StandardScaler na features (target se **ne normalizuje**)
- Build sliding-window sekvenci za GRU ulaz (3D tensor `(samples, timesteps, features)`)

**Alati/tehnike:**
- `pandas`, `numpy`, `sklearn.preprocessing.StandardScaler`
- `sqrt` transformacija target-a

### [WR_NN.ipynb](../notebooks/WR_NN.ipynb)

Kratak, nestrukturiran eksperiment — samo kod, bez markdown dokumentacije. Koristi `log1p` transformaciju target-a, ali ne definiše eksplicitan model (verovatno WIP).

### [WR_GRU_Model.ipynb](../notebooks/WR_GRU_Model.ipynb)

Prvi baseline RNN — jednostavan GRU.

**Arhitektura:**
```
Input (5 timesteps × n_features)
  → GRU(32 units)
  → Dense(1, linear)
```

**Šta se radi:**
- Sliding window sekvence dužine **5 utakmica**
- Train/test split pre fitovanja scaler-a (protiv leakage)
- Trening na MSE lossu sa Adam optimizer-om
- Evaluacija na MAE / RMSE / R²

**Rezultat:** Baseline — koristi se samo kao početna tačka. Skromna performansa.

---

## Faza 2 — Objedinjeni Optuna pipeline

### [WR_Career_RNN_Optuna.ipynb](../notebooks/WR_Career_RNN_Optuna.ipynb)

Najobimniji notebook u projektu — unified pipeline koji poredi sve glavne arhitekture.

**Ključna poboljšanja u odnosu na ranije:**
1. **Career-based sekvence** — sekvence se šire preko granica sezona (ne samo u okviru jedne), sa `is_new_season` i `weeks_since_last_game` signalima
2. **Optuna hyperparameter optimization** za MLP, RNN i TCN
3. **Tree baseline** — XGBoost, LightGBM, RandomForest
4. **Ensemble** — težinsku kombinaciju najboljih modela

**Feature engineering (184 feature-a):**
- **Career-based rolling**: `lag1`, `roll5`, `momentum` (trend = roll5 − lag1), bez `roll3` (redundantno)
- **Interaction features**: `target_share × pregame_total`, itd.
- `is_new_season`, `weeks_since_last_game`

**Target transformacija:** `sqrt(receiving_yards)` — bolje od log1p za ovu distribuciju
**Sample weights:** kontinualni, veća težina za high-yardage utakmice

**Optuna search space za RNN:**
- Cell type: LSTM vs GRU
- Bidirectional: yes/no
- Depth: 1–4 sloja
- Width: 64–512 units po sloju
- Attention: MultiHeadAttention (2/4/8 glava, on/off)
- Normalizacija: LayerNorm / BatchNorm / none
- Dense head: 1–3 sloja, 32–256 units
- Sequence length: 8, 12, 16, 24
- Optimizer: AdamW (LR, weight decay)
- Loss: Huber (delta searched)
- Pruner: `MedianPruner` (rano zaustavljanje bezperspektivnih trial-a)

**TCN search space:**
- Dilated causal convolutions (`Conv1D`)
- Residual blocks sa povećanjem dilation-a (1, 2, 4, 8, ...)
- Broj blokova, kernel size, filter count

**Sumarni rezultati ove faze:**

| Model | Test R² | Test MAE | Test RMSE |
|---|---|---|---|
| **LightGBM** | **0.3346** | 18.15 | 25.13 |
| **XGBoost** | **0.3335** | 18.15 | 25.15 |
| MLP (Optuna) | 0.3156 | 18.35 | 25.80 |
| TCN | 0.2453 | 19.30 | 27.59 |
| RNN (Optuna best) | 0.2157 | 19.73 | 27.93 |

**Ključni nalaz:** Kompleksni neural modeli (RNN, TCN) **gori** od jednostavnih tree modela. Ovo je pokretalo sve dalje eksperimente.

---

## Faza 3 — Feature redukcija

### [WR_Feature_Selection.ipynb](../notebooks/WR_Feature_Selection.ipynb)

Sistemska analiza: koje od 184 feature-a su zapravo korisne?

**Metodologija:**
1. Trenira XGBoost + LightGBM na svih 184 feature-a
2. Ekstraktuje `feature_importance` iz oba modela
3. Normalizuje i kombinuje skorove → top 40 feature-a
4. Poredi **learning curves** (RMSE po boosting round-u) sa `eval_set`:
   - XGBoost, LightGBM sa **184 features** vs **40 features**

**Vizualizacije:**
- Elbow plot + kumulativna importance
- Train vs val loss po rundi (učenje po epohi za tree modele)
- Overfit gap (train - val) — mera overfit-a

**Rezultati:**
- Top 40 features pokriva **51.8%** kumulativne importance
- Tree modeli jedva overfit-uju (gap 0.10–0.24) — feature redukcija ima **minimalan uticaj** na tree performansu
- Zaključak: 144 od 184 feature-a su **noise za mreže** — zato MLP/RNN overfit-uju kad ih sve dobiju

**Izlaz:** `results/selected_features_top40.json` — lista top 40 imena kolona + importance skorovi

---

## Faza 4 — MLP poboljšanja

### [WR_MLP_Comparison.ipynb](../notebooks/WR_MLP_Comparison.ipynb)

Prvi put kontrolisano izolujemo efekte pojedinačnih promena na MLP-u.

**Tri modela:**

| Model | Features | Arhitektura | Regularizacija |
|---|---|---|---|
| **A) Original MLP** | 184 | 5 slojeva [448,128,320,448,256], dropout 0.5 | Original sample weights |
| **B) Reduced MLP** | 40 | Ista kao A | Ista kao A |
| **C) Improved MLP** | 40 | Constrained (max 3 sloja, max 256 units) | GaussianNoise + Mixup + softened weights + forced LayerNorm |

**Novi alati u ovoj fazi:**
- **MixupGenerator** — custom Keras Sequence koji miksuje parove uzoraka `(x1, y1)` i `(x2, y2)` u `(λx1+(1-λ)x2, λy1+(1-λ)y2)`. λ ~ Beta(α, α).
- **GaussianNoise** layer — dodaje Gaussian šum na ulaz tokom treninga (ne tokom inferencije)
- **Softened sample weights** — `1.0 + 0.3*sqrt(y/mean)`, capped at 2.0 (umesto ekstremnih [1, 4.12])
- **Forced LayerNorm** — na svakom sloju, bez opcije da Optuna isključi

**Rezultati (A / B / C):**

| Model | R² | MAE | RMSE | Loss gap |
|---|---|---|---|---|
| A) Original MLP (184) | 0.3079 | 18.10 | 25.62 | -0.624 |
| **B) Same arch (40)** | **0.3289** | 18.19 | **25.23** | -0.640 |
| C) Improved (40) | 0.3151 | **17.91** | 25.49 | -0.110 |

**Zaključci:**
- Model B (samo feature redukcija) je najbolji po R²/RMSE
- Model C (sve poboljšanja) je najbolji po MAE (medijana), ali nešto gori po R²
- Postoji **trade-off**: veći kapacitet → bolji R²; jača regularizacija → bolji MAE ali manji R²

### [WR_MLP_Hybrid.ipynb](../notebooks/WR_MLP_Hybrid.ipynb)

Pokušaj da se spoji najbolje iz B i C: **veliki kapacitet + regularizacija**.

**Dva nova modela:**

**Model D — Fixed Hybrid:** Tačno ista arhitektura kao Model B [448,128,320,448,256], plus:
- GaussianNoise(0.15) na ulaz
- Mixup(α=0.2)
- Original sample weights
- Cilj: izolovati čist efekat dodavanja regularizacije na već poznatu-dobru arhitekturu

**Model E — Optuna Hybrid:** Optuna pretraga sa **forsiranom** regularizacijom:
- Force: GaussianNoise + LayerNorm + Mixup uvek uključeni
- Search: 3–5 slojeva, 128–448 units, noise stddev [0.05–0.25], mixup α [0.1–0.4], sw_strength [0.3–1.0], dropout [0.2–0.5], LR, WD, Huber delta, batch size
- 100 trial-a

**Rezultati svih 5 MLP modela:**

| Model | R² | MAE | RMSE | Feat |
|---|---|---|---|---|
| A) Original MLP (184 feat) | 0.3079 | 18.10 | 25.62 | 184 |
| **B) Same arch (40 feat)** | **0.3289** | 18.19 | **25.23** | 40 |
| C) Improved MLP (40 feat) | 0.3151 | 17.91 | 25.49 | 40 |
| D) Fixed Hybrid (40 feat) | 0.3055 | **17.83** | 25.67 | 40 |
| E) Optuna Hybrid (40 feat) | 0.3225 | 17.88 | 25.35 | 40 |

**Najbolji Optuna E parametri:**
- 4 sloja [320, 384, 192, 128]
- noise=0.25, mixup=0.2, sw_strength=0.9, dropout=0.4
- lr=0.000245, wd=0.00021, batch=32

**Zaključak:** Hybrid modeli D/E imaju bolji MAE, ali **nisu pobili Model B po R²**. Postoji plafon oko R²=0.33 za MLP na ovom problemu.

---

## Faza 5 — RNN redizajn

### [WR_RNN_Improved.ipynb](../notebooks/WR_RNN_Improved.ipynb)

Pokušaj da se sistemski isprave sve slabosti originalnog RNN-a (R²=0.2157).

**Ključne promene:**

| Aspekt | Original RNN | Improved RNN |
|---|---|---|
| Sequence features | 184 lag1/roll5 po timestep-u | ~33 **sirove** game statistike |
| Static features | Nema (sve u sekvenci) | 20 pregame/career preko **zasebne grane** |
| Kratke karijere | Bacane (seq_len filter) | **Padding + Masking** (nema izgubljenih podataka) |
| Arhitektura | 1–4 sloja, 64–512 units | Max 2 sloja, max 192 units |
| Normalizacija | Opciono (Optuna bira 'none') | **Forsiran** LayerNorm |
| Input regularizacija | Nema | GaussianNoise (na static grani) |
| Sample weights | Originalni [1, 4.12] | Searched strength [0.5, 1.0] |

**Dual-input arhitektura:**
```
seq_input (T, 33 raw features)
  → Masking(0.0)
  → BiLSTM/BiGRU (1-2 sloja, max 192 units)
  → LayerNorm → Dropout
                                          ↓
static_input (20 pregame features)        → Concat → Dense → Output
  → GaussianNoise
  → Dense → LayerNorm → Dropout           ↑
```

**Kritično otkriće:** `GaussianNoise + Masking` su **nekompatibilni** — šum dodaje non-zero vrednosti na padded pozicije, što razbija Masking detekciju. Rešenje: GaussianNoise **samo** na static grani.

**Kritičan bug (otkriven kasnije):** Target `receiving_yards` je bio u listi `seq_feature_cols`, pa je StandardScaler primenjen na njega tokom pripreme podataka. Zatim je `y_orig` uzet iz `df_scaled` što je značilo da je i target bio standardizovan. `sqrt(max(z, 0))` je nulirao polovinu ciljnih vrednosti. Rezultat: nesmislen R²=0.1806 u standardizovanoj skali, ne jardima.

### [WR_RNN_v2.ipynb](../notebooks/WR_RNN_v2.ipynb)

Ispravka bug-a iz v1 + četiri targetovane promene na osnovu dijagnoze:

**Šta se menja:**
1. **Bug fix** — `df_scaled['receiving_yards_orig'] = df['receiving_yards'].values` pre skaliranja; target se vadi iz neskalirane kolone
2. **Engineered sequence channels** — dodajemo `lag1` i `roll3` za 6 ključnih statova (receiving_yards, targets, receptions, epa, target_share, air_yard_share) → **45 kanala po timestep-u**. RNN više ne mora sam da nauči temporalne statistike.
3. **Kraće sekvence** — `seq_len ∈ [4, 6, 8]` (originalno je bilo 12, koje je dominirao padding)
4. **Umerenija regularizacija** — `noise ∈ {0, 0.05, 0.1}`, `dropout ∈ [0.2, 0.3]`
5. **Više kapaciteta** — `n_layers ∈ [1, 2]`, `units ∈ [96, 128, 192, 256]`
6. **No-masking baseline** — drugi trening samo na uzorcima sa `real_len ≥ 6` (nema paddinga) da testiramo hipotezu "da li padding hendikepira RNN"

**Optuna:** 60 trial-a (smanjeno sa 100 zbog smanjenog search space-a).

---

## Faza 6 — Napredne tehnike

### [WR_RNN_Attention_PlayerEmbed.ipynb](../notebooks/WR_RNN_Attention_PlayerEmbed.ipynb)

Dve ideje iz liste "dodatnih predloga":

**1) Attention Pooling — custom Keras layer**

```python
class AttentionPool(layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.score = layers.Dense(1)
        self.supports_masking = True
    def call(self, x, mask=None):
        scores = self.score(x)  # (B, T, 1)
        if mask is not None:
            mask_f = tf.cast(mask, tf.float32)[..., None]
            scores = scores + (1 - mask_f) * -1e9
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(x * weights, axis=1)
```

Umesto uzimanja poslednjeg hidden state-a, `AttentionPool` uči da pondera sve timestep-ove. Poštuje Keras mask (padded pozicije dobiju -1e9 pre softmax-a).

**2) Player Embeddings**

- `player_to_idx` mapa: fit samo na train igrače, OOV=0 za nepoznate
- `Embedding(n_players, 8)` sa L2 regularizacijom (1e-4)
- Uči "talent vektor" po igraču — hvata nemerljive osobine koje game stats ne vide

**Tri-ulazna arhitektura:**
```
seq_input → Masking → BiRNN(return_seq=True) → AttentionPool → LayerNorm → Dropout
pid_input → Embedding(n_players, 8) → Flatten
static_input → GaussianNoise → Dense → LayerNorm → Dropout

                    ↓ Concatenate ↓
                  Dense(96) → LayerNorm → Dropout → Dense(1)
```

**Matrica eksperimenta (4 modela):**

| Model | RNN tip | Target transform |
|---|---|---|
| LSTM_sqrt | LSTM | √yards |
| LSTM_log1p | LSTM | log(1+yards) |
| GRU_sqrt | GRU | √yards |
| GRU_log1p | GRU | log(1+yards) |

`TargetTransform` klasa pruža `forward` i `inverse` metode da se rezultat uvek konvertuje nazad u jarde pre evaluacije.

Bez Optuna — koristi se jedna razumna konfiguracija za sva četiri modela radi čistog head-to-head poređenja.

### [WR_MLP_Quantile.ipynb](../notebooks/WR_MLP_Quantile.ipynb)

Kvantilna regresija umesto point estimate.

**Motivacija:**
- q50 (medijana) kao point prediction je **L1-robustan** na outliere → bolji MAE
- Besplatno se dobija **interval neizvesnosti** (q90 − q10)
- Model uči pravi oblik distribucije, ne samo jednu tačku

**Arhitektura:** Identična MLP-B ([448, 128, 320, 448, 256], LayerNorm, dropout 0.5, lr 0.0037, wd 1.5e-5) — jedina razlika je **Dense(3)** izlazna glava umesto Dense(1).

**Pinball loss za tri kvantila:**
```python
def multi_pinball_loss(quantiles):
    q_tensor = tf.constant(quantiles, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, 1))
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q_tensor * e, (q_tensor - 1) * e))
    return loss
```

**Evaluacija:**
- `q50` → MAE / RMSE / R² u jardima (posle inverznog sqrt-a)
- `q10-q90` interval → **coverage** (cilj 80%) + prosečna širina intervala
- Crossing quantile korekcija: `q10 = min(q10, q50)`, `q90 = max(q90, q50)`

---

## 10. Pregled svih tehnika i alata

### 10.1 Modeli / arhitekture

| Kategorija | Konkretno |
|---|---|
| **Tree modeli** | XGBoost, LightGBM, RandomForest |
| **Feed-forward** | MLP (2–5 sloja, LayerNorm / BatchNorm) |
| **Rekurentni** | LSTM, GRU, BiLSTM, BiGRU |
| **Konvolucijski (temporal)** | TCN (dilated causal Conv1D + residual) |
| **Dual-input** | Sequence branch (RNN) + Static branch (Dense) |
| **Tri-input** | Sequence + Player Embedding + Static |
| **Attention** | MultiHeadAttention (u originalu), Custom AttentionPool (v3) |

### 10.2 Ulazne reprezentacije

| Tip | Gde korišćeno |
|---|---|
| Lag1, lag2, lag3 | `WR_Analysis`, `WR_Career_RNN_Optuna` |
| Roll3, roll5, roll8 (mean) | `WR_Analysis` |
| Roll3, roll5 (std) — volatilnost | `WR_Analysis` |
| Career-wide roll5 (preko sezona) | `WR_Career_RNN_Optuna` + kasniji |
| Momentum (roll5 − lag1) | `WR_Career_RNN_Optuna` |
| Interaction features (product) | `WR_Career_RNN_Optuna` |
| Expanding defensive matchup mean | `WR_Analysis` |
| Pregame (spread, total, vreme) | Svi |
| Career averages prošle sezone | Svi moderni |
| Raw per-game stats (bez lag/roll) | `WR_RNN_Improved` (prvi put) |
| Raw + lag1 + roll3 (mix) | `WR_RNN_v2` i kasniji |

### 10.3 Target transformacije

| Transform | Inverse | Gde korišćeno |
|---|---|---|
| `sqrt(y)` | `pred²` | Većina (default) |
| `log1p(y)` | `expm1(pred)` | `WR_NN`, `WR_RNN_Attention_PlayerEmbed` |
| Nijedno (raw yards) | — | `WR_GRU_Model` |

### 10.4 Normalizacija feature-a

Svuda: `StandardScaler`, fit **samo na train**. Target se nikad ne normalizuje (osim bug-a u `WR_RNN_Improved`).

### 10.5 Loss funkcije

| Loss | Gde |
|---|---|
| MSE (`mean_squared_error`) | `WR_GRU_Model` |
| MAE | Rano |
| **Huber** (δ searched 0.5–2.0) | Svi moderni |
| **Pinball** (multi-quantile q10/q50/q90) | `WR_MLP_Quantile` |

### 10.6 Optimizeri

| Optimizer | Gde |
|---|---|
| Adam | Rani (`WR_GRU_Model`) |
| **AdamW** (sa weight decay) | Svi moderni |

### 10.7 Regularizacija

| Tehnika | Gde | Detalji |
|---|---|---|
| Dropout | Svi NN | 0.2–0.5, na svakom Dense/RNN sloju |
| LayerNorm | Svi moderni | Forsiran na svakom sloju |
| BatchNorm | `WR_Career_RNN_Optuna` (opcija) | Optuna je birala |
| Recurrent dropout | RNN | 0.1–0.3 |
| Weight decay (AdamW) | Svi moderni | 1e-4 do 5e-3 |
| **GaussianNoise** na ulazu | MLP Improved/Hybrid, RNN | stddev 0.05–0.25 |
| **Mixup** data augmentation | `WR_MLP_Comparison`, `WR_MLP_Hybrid` | α 0.1–0.4 |
| L2 na Embedding | `WR_RNN_Attention_PlayerEmbed` | 1e-4 |
| **Early stopping** | Svi | patience 12–30 |
| **ReduceLROnPlateau** | Svi | factor 0.3, patience 5–10 |
| **ModelCheckpoint** (best weights) | Svi | monitor val_loss |

### 10.8 Sample weights (za sqrt target)

- **Original**: diskretni `[1.0, 4.12]` (binarne kategorije high/low yards)
- **Softened**: `1.0 + 0.3 * sqrt(y/mean_y)`, cap 2.0
- **Continuous (parametrized)**: `1.0 + strength * sqrt(y/mean_y)`, `strength ∈ [0.5, 1.0]`

### 10.9 Sequence handling

| Pristup | Gde |
|---|---|
| Fixed-length sliding window (5 games) | `WR_GRU_Model` |
| Fixed-length career sequences (seq_len ∈ [8, 12, 16, 24]) | `WR_Career_RNN_Optuna` |
| Career-based + filter kratkih | `WR_Career_RNN_Optuna` |
| **Padding + Masking** (nema izgubljenih podataka) | `WR_RNN_Improved` i dalje |
| No-masking baseline (filter real_len ≥ 6) | `WR_RNN_v2` |

### 10.10 Hyperparameter optimization

| Alat | Gde |
|---|---|
| Manuelno (grid po intuiciji) | Rano |
| **Optuna** TPE sampler + MedianPruner | `WR_Career_RNN_Optuna` (100 trial-a), `WR_MLP_Comparison` (80), `WR_MLP_Hybrid` (100), `WR_RNN_Improved` (100), `WR_RNN_v2` (60) |
| `TFKerasPruningCallback` | Optuna rano zaustavljanje NN trial-a |

### 10.11 Ensemble

- `WR_Career_RNN_Optuna`: **weighted ensemble** najboljih MLP + RNN + TCN, težine optimizovane na val setu

### 10.12 Napredne tehnike (Faza 6)

| Tehnika | Notebook |
|---|---|
| **Attention pooling** (custom Keras layer) | `WR_RNN_Attention_PlayerEmbed` |
| **Player embeddings** (sa OOV i L2) | `WR_RNN_Attention_PlayerEmbed` |
| **Kvantilna regresija** (pinball loss, q10/q50/q90) | `WR_MLP_Quantile` |
| **Target transform uporedivanje** (sqrt vs log1p) | `WR_RNN_Attention_PlayerEmbed` |

---

## 11. Rezultati — finalna tabela

**Referentni rezultati na test setu (sezone 2024–2025):**

| Model | Features | Test R² | Test MAE | Test RMSE | Loss gap |
|---|---|---|---|---|---|
| **LightGBM** | 184 | **0.3346** | 18.15 | 25.13 | malo |
| **XGBoost** | 184 | 0.3335 | 18.15 | 25.15 | malo |
| **MLP-B (Reduced)** | 40 | **0.3289** | 18.19 | **25.23** | -0.640 |
| MLP-E (Optuna Hybrid) | 40 | 0.3225 | 17.88 | 25.35 | -0.542 |
| MLP-C (Improved) | 40 | 0.3151 | 17.91 | 25.49 | -0.110 |
| MLP (Career_RNN_Optuna) | 184 | 0.3156 | 18.35 | 25.80 | — |
| MLP-A (Original) | 184 | 0.3079 | 18.10 | 25.62 | -0.624 |
| MLP-D (Fixed Hybrid) | 40 | 0.3055 | **17.83** | 25.67 | -0.592 |
| Original TCN | 184 | 0.2453 | 19.30 | 27.59 | — |
| Original RNN (Optuna) | 184 | 0.2157 | 19.73 | 27.93 | — |
| RNN Improved (v1) | raw 33 | **⚠ bug** | — | — | — |

**⚠ Napomena:** `WR_RNN_Improved` je imao bug u pripremi target-a (standardizovan pre sqrt-a) — rezultati nisu u jardima. Ispravka u `WR_RNN_v2`. Rezultati za v2, RNN_Attention_PlayerEmbed i MLP_Quantile će se popuniti nakon pokretanja.

---

## 12. Ključni zaključci i naučene lekcije

### 12.1 Šta radi

1. **Feature redukcija pomaže mrežama, ne drvima.** Tree modeli jedva overfit-uju i mogu da pojedu 184 feature-a bez problema. MLP-ovi overfit-uju i dobijaju +0.02 R² samo od redukcije na 40 feature-a (Model A → Model B).

2. **AdamW sa pravim weight decay** je dramatično bolji od običnog Adam-a. Optuna je u MLP Hybrid-u našla da je 14× jači weight decay od originalnog optimalan.

3. **Career-based sekvence** (preko granica sezona, sa `is_new_season` signalom) su bolje od per-season sekvenci.

4. **Padding + Masking** umesto filtriranja kratkih karijera — ne gubimo podatke, Masking layer se brine za padded timestep-ove.

5. **Forsiran LayerNorm** na svakom sloju — stabilan trening, Optuna inače bira 'none' i loš rezultat.

6. **Huber loss** sa searched delta-om je bolji od čistog MSE — manje osetljiv na outlier-e.

7. **Sqrt target transformation** je bolji od log1p i raw za ovu distribuciju.

8. **Continuous sample weights** (parametrizovani strength, ne diskretni [1, 4.12]) su bolji.

### 12.2 Šta ne radi / zamke

1. **Kompleksni neural modeli su gori od tree modela** na ovom problemu. RNN je počeo sa R²=0.2157, tree modeli dostižu 0.3346. Razlog: NFL WR week-to-week je toliko šumovit da temporalna struktura ne donosi dovoljno signala u odnosu na cenu kapaciteta.

2. **GaussianNoise + Masking su nekompatibilni.** Šum razbije padded nule, Masking detekcija ne radi.

3. **Mixup + padding su nekompatibilni.** Isti razlog — mešanje padded i real pozicija.

4. **Previše regularizacije ubija RNN.** U `WR_RNN_Improved` Optuna je izabrala noise_stddev=0.3 + dropout=0.4 + aggressive weight decay, što je dovelo do **underfit-a** (val loss < train loss).

5. **Target u seq_feature_cols je opasan.** Ako se `receiving_yards` stavi u listu za StandardScaler, a zatim se target uzme iz istog DataFrame-a, dobije se standardizovan target → nesmislena metrika. Uvek čuvati neskaliranu kopiju: `df_scaled['receiving_yards_orig'] = df['receiving_yards']`.

6. **Dugi seq_len sa padding-om je iluzija.** Ako je 60%+ timestep-ova padded, RNN "vidi" samo 3–5 real step-ova, a dodatni kapacitet seq_len=12 samo troši parametre.

7. **Raw sirove sekvence su teže za RNN nego engineered.** RNN mora da uči lag1/roll5 iz nule, što pri ~5k training uzoraka ne uspe bolje od eksplicitnih engineered feature-a.

### 12.3 Opšti uvidi

- **Plafon za WR yards prediction je oko R² 0.33–0.37.** Tree modeli ga već skoro dostižu. Vrednost mreža je u komplementarnim greškama (za ensemble) i u uncertainty intervalima (kvantilna regresija).

- **Overfit != loš model.** MLP-B ima `loss_gap = -0.640` (val loss daleko veći od train), ali je **najbolji** po R². Overfit na train ne znači automatski overfit test-a — depending on data leakage. Gledati **val performansu**, ne gap.

- **Underfit je podmukliji od overfit-a.** `WR_RNN_Improved` je dobio `loss_gap = -0.049` (val **bolji** od train), što je dijagnoza preterane regularizacije — to je jasan alarm za smanjenje šuma/dropout-a.

- **Optuna može da nađe "rešenja" koja minimizuju pogrešnu metriku.** U `WR_RNN_Improved` Optuna je minimizovala val RMSE u standardizovanoj skali (zbog bug-a), pa je našla ultra-regularizovan model koji nikad ne predviđa visoke vrednosti. Uvek validirati da je metrika u pravoj skali.

- **Head-to-head poređenje je vrednije od velikog search space-a kad imaš hipotezu.** U `WR_RNN_Attention_PlayerEmbed` svesno izbegavamo Optuna i treniramo 4 modela sa **istom** konfiguracijom da izolujemo efekat `{LSTM vs GRU}` i `{sqrt vs log1p}`.

### 12.4 Preporučeni naredni koraci

1. Pokrenuti `WR_RNN_v2`, `WR_RNN_Attention_PlayerEmbed`, `WR_MLP_Quantile` i uporediti rezultate
2. **Stacking ensemble** (LightGBM + MLP-B + najbolji RNN) — različiti modeli prave različite greške
3. **Seed ensemble** — treniranje iste MLP-B arhitekture sa 5 seeds i prosečavanje predikcija (+0.005–0.015 R²)
4. **Residual learning** — MLP trenira na rezidualima LightGBM-a
5. Nova feature engineering: `teammate_target_share`, `boom_rate`, `bust_rate`, `qb_change` flag

---

*Dokumentacija generisana 2026-04-13 na osnovu 11 WR_ notebook-a u projektu.*
