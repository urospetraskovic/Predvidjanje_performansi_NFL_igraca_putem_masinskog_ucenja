# NFL Player Performance Prediction

This project predicts NFL players performance using machine learning models including neural networks, ensemble methods, and traditional ML algorithms.

## Dependencies

### Required Python Version
- Python 3.10 or 3.11 (preporučeno)
- TensorFlow 2.15/2.16 ne podržava Python 3.12+, a pandas 2.1 zahteva Python ≥3.9
- Napomena: TensorFlow ≥2.11 nema GPU podršku na native Windows — za GPU koristiti WSL2

This project requires the following Python packages:

### Data Science & Machine Learning
- pandas==2.1.3
- numpy>=1.24,<2.0
- scipy>=1.11.0
- scikit-learn>=1.3.0
- joblib>=1.3.0
- shap>=0.42.0,<0.46
- optuna>=3.0.0
- xgboost>=1.7.0
- lightgbm>=4.0.0

### Visualization
- matplotlib>=3.7.0
- seaborn>=0.12.0
- adjustText>=1.3.0

### Jupyter Environment
- jupyter>=1.0.0
- ipykernel>=6.25.0
- ipython>=8.0.0

### Neural Networks
- tensorflow>=2.15,<2.17

### Web Scraping (Optional)
- requests==2.31.0
- beautifulsoup4==4.12.2
- lxml>=5.0
- selenium==4.15.2
- webdriver-manager==4.0.1

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd nfl-analytics-project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
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



