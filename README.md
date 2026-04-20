# NFL Player Performance Prediction

This project predicts NFL players performance using machine learning models including neural networks, ensemble methods, and traditional ML algorithms.

## Dependencies

### Required Python Version
- **Python 3.13.x** (testirano na 3.13.11)
- TensorFlow 2.21 podržava Python 3.9–3.13
- Napomena: TensorFlow ≥2.11 nema GPU podršku na native Windows — za GPU koristiti WSL2

Sve verzije su pinovane u `requirements.txt` radi reprodukcije okruženja. Ispod je pregled.

### Data Science & Machine Learning
- pandas==2.3.3
- numpy==2.4.0
- scipy==1.16.3
- scikit-learn==1.8.0
- joblib==1.5.3
- shap==0.50.0
- optuna==4.8.0
- optuna-integration==4.8.0
- xgboost==3.2.0
- lightgbm==4.6.0
- numba==0.64.0

### Visualization
- matplotlib==3.10.8
- seaborn==0.13.2
- adjustText==1.3.0

### Jupyter Environment
- ipykernel==7.2.0
- ipython==9.10.0
- (VS Code sa Jupyter ekstenzijom koristi `ipykernel` direktno — `jupyter` meta-paket nije obavezan)

### Neural Networks
- tensorflow==2.21.0
- keras==3.13.2
- keras-tuner==1.4.8

### Reporting / Utilities
- python-docx==1.2.0
- python-dotenv==1.2.1
- tabulate==0.10.0
- tqdm==4.67.3

### Web Scraping (Optional)
- requests==2.32.5
- beautifulsoup4==4.14.3
- lxml==6.0.2
- selenium==4.40.0
- webdriver-manager==4.0.2
- cloudscraper==1.2.71

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd nfl-analytics-project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Use Python 3.13.x explicitly — the project is pinned to this version
   py -3.13 -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
Or use VS Code with Jupyter extension to open `.ipynb` files directly.

## Project Structure

This project is organized around Jupyter notebooks that follow a systematic development approach. The main analysis is contained in the `notebooks/` directory:

### 00_eda/ - Exploratory Data Analysis
Data exploration, analysis, and feature selection

### 01_init_workbench/ - Initial Experiments
- **00_init_experiments/** - Early experimental work and preprocessing
- **01_primary_optuna/** - Main Optuna hyperparameter optimization for all model types

### 02_baseline_models/ - Baseline Model Development
- **classical_ml/** - Traditional machine learning models (Random Forest, XGBoost, LightGBM)
- **mlp/** - Multi-layer perceptron neural networks

### 03_rnn/ - Recurrent Neural Networks
RNN models with various architectures (LSTM, GRU)

### 04_additional_approaches/ - Additional Techniques
- **ensemble/** - Ensemble methods combining multiple models
- **ceiling/** - Multi-game prediction analysis (2-7 game rolling predictions)
- **topn_features/** - Models using only top N features (5 and 10 features)

### 05_final/ - Final Models
Final optimized models and architectures for each approach



