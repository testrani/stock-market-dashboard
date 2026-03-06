# 📈 Stock Market Analytics Dashboard

A Dash web application for global stock market analysis featuring EDA charts, ML model performance, and sentiment analysis.

## Features
- **EDA Tab**: Box plots, bar charts, heatmaps, correlation matrices
- **Model Performance Tab**: Confusion matrix & ROC curves for BLR, GNB, Decision Tree, Random Forest
- **Sentiment Analysis Tab**: VADER/FinBERT sentiment charts (requires `web_scrape.csv`)

## Local Run
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8050
```

## Optional: Sentiment Tab
Place your `web_scrape.csv` in the `data/` folder before deploying.
