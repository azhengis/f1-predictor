# F1 Race Prediction System

Predicts the full finishing order of Formula 1 races using machine learning. Given qualifying results and historical data, the model ranks all 20 drivers from P1 to P20 before the race starts.

## What it does

Takes pre-race information such as qualifying times, recent driver form, championship standings, and team pit stop history, then outputs a predicted finishing order with explanations for each prediction.

## How it works

A LightGBM regression model assigns each driver a performance score. Those scores are sorted to produce the final ranking. SHAP values explain which features drove each prediction.

## Results

Tested on the 2024 and 2025 seasons:
- Spearman rank correlation: 0.760
- Winner accuracy: 58.3%
- Top 10 accuracy: 86.2%

For comparison, always predicting the pole sitter wins gives about 40% accuracy. Random guessing gives 5%.

## Data sources

- Kaggle F1 World Championship dataset (1950 to 2023)
- Jolpica API for 2024 and 2025 results

## Tech stack

- LightGBM for the model
- SHAP for explainability
- Streamlit for the dashboard
- FastAPI for the REST endpoint
- Deployed on Streamlit Cloud

## Run locally
```bash
git clone https://github.com/azhengis/f1-predictor
cd f1-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python features/pipeline.py
python models/train.py
streamlit run app/streamlit_app.py
```

## Project structure
```
data/          raw CSVs and processed features
features/      feature engineering pipeline
models/        training and evaluation
inference/     prediction pipeline and SHAP explainer
app/           Streamlit dashboard
```
