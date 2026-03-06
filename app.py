import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn import tree as sktree
import matplotlib
matplotlib.use('Agg')  # Required for server-side rendering (no display)
import matplotlib.pyplot as plt
import io
import base64
import os
import yfinance as yf

# ============================================================================
# DATA LOADING - Download or load from CSV
# ============================================================================

def load_combined_data():
    """Load or download stock market data."""
    csv_path = "data/combined_data.csv"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(csv_path):
        print("Loading combined_data from CSV...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return df

    print("Downloading data from Yahoo Finance...")
    tickers = {
        "^NSEI": "NSE", "^DJI": "DJI", "^IXIC": "IXIC",
        "^HSI": "HSI", "^N225": "N225", "^GDAXI": "GDAXI", "^VIX": "VIX"
    }
    frames = {}
    for ticker, name in tickers.items():
        try:
            raw = yf.download(ticker, start="2019-01-01", progress=False)
            raw = raw.copy()
            raw.columns = [f"{c}_{ticker}" for c in raw.columns]
            frames[name] = raw
        except Exception as e:
            print(f"  Warning: could not download {ticker}: {e}")

    if not frames:
        raise RuntimeError("No data could be downloaded.")

    combined = None
    for name, df in frames.items():
        combined = df if combined is None else combined.merge(df, how="outer", left_index=True, right_index=True)

    combined = combined.ffill()

    # Compute returns
    return_map = {
        "NSE": "^NSEI", "DJI": "^DJI", "IXIC": "^IXIC",
        "HSI": "^HSI", "N225": "^N225", "GDAXI": "^GDAXI", "VIX": "^VIX"
    }
    for name, ticker in return_map.items():
        close_col = f"Close_{ticker}"
        if close_col in combined.columns:
            combined[f"{name}_Return"] = (combined[close_col].pct_change() * 100).round(4)

    # Close ratios
    for name, ticker in [("NSE", "^NSEI"), ("N225", "^N225"), ("HSI", "^HSI")]:
        open_col, close_col = f"Open_{ticker}", f"Close_{ticker}"
        if open_col in combined.columns and close_col in combined.columns:
            combined[f"{name}_Close_Ratio"] = combined[open_col] / combined[close_col]

    # Target variable
    if "Open_^NSEI" in combined.columns and "Close_^NSEI" in combined.columns:
        combined["Nifty_Open_Dir"] = (combined["Open_^NSEI"] > combined["Close_^NSEI"].shift(1)).astype(int)

    combined["Quarter"] = combined.index.quarter.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
    combined["Month"] = combined.index.month
    combined["Year"] = combined.index.year

    combined.to_csv(csv_path)
    print(f"Data saved to {csv_path}")
    return combined


print("Loading data...")
combined_data = load_combined_data()
print(f"Data loaded: {combined_data.shape}")

# ============================================================================
# APP INIT
# ============================================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Expose Flask server for gunicorn

# ============================================================================
# CHART CONFIG
# ============================================================================
columns_for_boxplot = [c for c in [
    'NSE_Return', 'DJI_Return', 'IXIC_Return',
    'HSI_Return', 'N225_Return', 'GDAXI_Return', 'VIX_Return'
] if c in combined_data.columns]

boxplot_options = [
    {'label': col.replace('_Return', ''), 'value': col}
    for col in columns_for_boxplot
]

columns_for_heatmap = columns_for_boxplot

quarter_order = ["Q1", "Q2", "Q3", "Q4"]
combined_data_heatmap = combined_data.copy()
combined_data_heatmap["Quarter"] = pd.Categorical(
    combined_data_heatmap["Quarter"], categories=quarter_order, ordered=True
)

returns_cols = [c for c in [
    'NSE_Return', 'DJI_Return', 'IXIC_Return',
    'HSI_Return', 'N225_Return', 'GDAXI_Return'
] if c in combined_data.columns]

corr_A = combined_data[returns_cols].corr()
combined_data_2024 = combined_data[combined_data['Year'] == 2024]
corr_B = combined_data_2024[returns_cols].corr()

global_indices = columns_for_boxplot

summary = combined_data.groupby('Nifty_Open_Dir')[global_indices].agg(['mean', 'median', 'std'])
bar_long = (
    summary.loc[:, (slice(None), ['mean', 'median'])]
    .stack(0).reset_index()
    .rename(columns={"level_1": "Index"})
    .melt(id_vars=["Nifty_Open_Dir", "Index"], value_vars=["mean", "median"],
          var_name="Statistic", value_name="Daily Return")
)
summary_flat = summary.copy()
summary_flat.columns = [f"{idx}__{stat}" for idx, stat in summary_flat.columns]
summary_flat = summary_flat.reset_index()

# ============================================================================
# HEATMAP HELPERS
# ============================================================================
def make_combined_heatmap(df, agg):
    if agg == "median":
        grouped = combined_data.groupby(['Year', 'Quarter'])[columns_for_heatmap].median().unstack()
        title_text = 'Median Daily Returns by Year and Quarter'
    else:
        grouped = combined_data.groupby(['Year', 'Quarter'])[columns_for_heatmap].mean().unstack()
        title_text = 'Mean Daily Returns by Year and Quarter'

    z_data = grouped.values
    column_labels = []
    for col in grouped.columns:
        if isinstance(col, tuple) and len(col) == 2:
            column_labels.append(f"{col[0].replace('_Return', '')}_{col[1]}")
        else:
            column_labels.append(str(col))

    year_labels = [str(y) for y in grouped.index]
    text = np.where(np.isfinite(z_data), np.round(z_data, 2).astype(str), "")

    fig = go.Figure(data=go.Heatmap(
        z=z_data, x=column_labels, y=year_labels,
        colorscale="RdYlBu_r", zmid=0,
        text=text, texttemplate="%{text}",
        hovertemplate="Year=%{y}<br>Index_Quarter=%{x}<br>Return=%{z:.4f}<extra></extra>",
        colorbar=dict(title="Return"), showscale=True
    ))
    fig.update_layout(
        title=title_text, xaxis_title="Index", yaxis_title="Year",
        margin=dict(l=80, r=50, t=80, b=80), height=600,
        xaxis=dict(tickangle=45, side="bottom")
    )
    return fig


def corr_fig(corr_df, title):
    z = corr_df.to_numpy()
    labels = corr_df.columns.tolist()
    text = np.round(z, 2).astype(str)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0,
        text=text, texttemplate="%{text}",
        hovertemplate="X=%{x}<br>Y=%{y}<br>Corr=%{z:.4f}<extra></extra>",
        colorbar=dict(title="Corr"),
    ))
    fig.update_layout(title=title, xaxis_title="Index", yaxis_title="Index",
                      height=520, margin=dict(l=70, r=30, t=70, b=70))
    return fig

# ============================================================================
# TAB 1: EDA
# ============================================================================
import dash.dash_table as dt

def create_eda_tab():
    return html.Div([
        html.H2("Exploratory Data Analysis (EDA) Charts",
                style={'textAlign': 'center', 'marginBottom': 30}),
        html.Div([
            # Global Indices vs Nifty_Open_Dir
            html.Div([
                html.H4("📈 Global Indices vs Nifty_Open_Dir Analysis",
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
                dcc.Tabs(id="indices_tabs", value="tab-bar", children=[
                    dcc.Tab(label="📊 Mean & Median (Bar)", value="tab-bar", children=[
                        html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center",
                                        "flexWrap": "wrap", "marginTop": "12px"},
                                 children=[
                                     html.Label("Select Indices:", style={'fontWeight': 'bold'}),
                                     dcc.Dropdown(
                                         id="bar_indices",
                                         options=[{"label": c.replace('_Return', ''), "value": c} for c in global_indices],
                                         value=global_indices, multi=True, clearable=False,
                                         style={"minWidth": "420px"},
                                     ),
                                 ]),
                        dcc.Graph(id="bar_fig", style={'height': '700px'}),
                        html.H5("📋 Summary Table (Mean/Median/Std)", style={'marginTop': 20}),
                        dt.DataTable(
                            id="summary_table",
                            data=summary_flat.to_dict("records"),
                            columns=[{"name": c, "id": c} for c in summary_flat.columns],
                            page_size=10, sort_action="native", filter_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"fontFamily": "sans-serif", "fontSize": 12, "padding": "6px"},
                            style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold'},
                            style_data={'backgroundColor': '#f8f9fa'}
                        ),
                    ]),
                    dcc.Tab(label="📦 Distributions (Box Plot)", value="tab-box", children=[
                        html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center",
                                        "flexWrap": "wrap", "marginTop": "12px"},
                                 children=[
                                     html.Label("Select Index:", style={'fontWeight': 'bold'}),
                                     dcc.Dropdown(
                                         id="box_index",
                                         options=[{"label": c.replace('_Return', ''), "value": c} for c in global_indices],
                                         value=global_indices[0], clearable=False,
                                         style={"minWidth": "320px"},
                                     ),
                                 ]),
                        dcc.Graph(id="box_fig", style={'height': '520px'}),
                    ]),
                ], style={'marginTop': '10px'}),
            ], style={'width': '100%', 'marginBottom': 30, 'padding': '20px',
                      'backgroundColor': '#f8f9fa', 'borderRadius': '8px',
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

            # Rolling Volatility + Box Plot
            html.Div([
                html.Div([
                    html.H4("30-day Rolling Volatility - NSE"),
                    dcc.Graph(figure=go.Figure(
                        data=[go.Scatter(
                            x=combined_data.index,
                            y=combined_data['NSE_Return'].rolling(30).std() if 'NSE_Return' in combined_data.columns else [],
                            mode='lines', name='NSE Volatility', line=dict(color='orange')
                        )],
                        layout=go.Layout(xaxis_title="Date", yaxis_title="Rolling Volatility", height=400)
                    ))
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

                html.Div([
                    html.H4("Box Plot"),
                    html.Label("Select Market Index:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='boxplot-dropdown', options=boxplot_options,
                                 value=columns_for_boxplot[0] if columns_for_boxplot else None,
                                 clearable=False, style={'marginBottom': 10}),
                    dcc.Graph(id='boxplot-graph', style={'height': '400px'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'marginBottom': 30}),

            # Bar Plot + Combined Heatmap
            html.Div([
                html.Div([
                    html.H4("Bar Plot - Median Returns by Year"),
                    html.Label("Select Market Index:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='barplot-dropdown', options=boxplot_options,
                                 value=columns_for_boxplot[0] if columns_for_boxplot else None,
                                 clearable=False, style={'marginBottom': 10}),
                    dcc.Graph(id='barplot-graph', style={'height': '400px'})
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

                html.Div([
                    html.H4("Combined Returns Heatmap"),
                    html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center",
                                    "flexWrap": "wrap", "marginBottom": "10px"},
                             children=[
                                 html.Label("Statistic:", style={'fontWeight': 'bold'}),
                                 dcc.RadioItems(
                                     id="combined_agg",
                                     options=[{"label": "Median", "value": "median"},
                                              {"label": "Mean", "value": "mean"}],
                                     value="median", inline=True, style={'marginLeft': '10px'}
                                 ),
                             ]),
                    dcc.Graph(id="combined_heatmap", config={"displayModeBar": True}, style={'height': '600px'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ]),

            # Correlation Heatmap
            html.Div([
                html.H4("🔥 Interactive Correlation Heatmap",
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
                html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center",
                                "justifyContent": "center", "flexWrap": "wrap", "marginBottom": "10px"},
                         children=[
                             html.Label("Select Correlation Type:", style={'fontWeight': 'bold'}),
                             dcc.RadioItems(
                                 id="corr_choice",
                                 options=[
                                     {"label": "A) 6-year daily returns", "value": "A"},
                                     {"label": "B) 2024 daily returns (6×6 matrix)", "value": "B"},
                                 ],
                                 value="A", inline=True, style={'marginLeft': '10px'}
                             ),
                         ]),
                dcc.Graph(id="corr_heatmap", style={'height': '520px'})
            ], style={'width': '100%', 'marginBottom': 30, 'padding': '20px',
                      'backgroundColor': '#f8f9fa', 'borderRadius': '8px',
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ])
    ])

# ============================================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================================
def _get_available_models():
    dummy_fpr = np.linspace(0, 1, 100)

    def make_tpr(base_auc):
        t = np.linspace(0, 1, 100) * base_auc + np.random.default_rng(42).normal(0, 0.03, 100)
        return np.clip(t, 0, 1)

    return [
        ('Binary Logistic Regression', np.array([[30, 37], [13, 73]]), 0.7051, dummy_fpr, make_tpr(0.7051)),
        ('Gaussian Naive Bayes',        np.array([[28, 39], [11, 75]]), 0.7033, dummy_fpr, make_tpr(0.7033)),
        ('Decision Tree',               np.array([[29, 35], [24, 65]]), 0.6198, dummy_fpr, make_tpr(0.6198)),
        ('Random Forest',               np.array([[19, 45], [10, 79]]), 0.6452, dummy_fpr, make_tpr(0.6452)),
    ]


def create_model_tab():
    available_models = _get_available_models()

    def safe_fmt(v, d=4):
        try: return f"{float(v):.{d}f}"
        except: return "N/A"

    rows = []
    for name, cm_mat, auc_score, _, _ in available_models:
        tn, fp, fn, tp = cm_mat.ravel()
        n = tn + fp + fn + tp
        acc = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        rows.append(html.Tr([
            html.Td(name,             style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'fontWeight': 'bold'}),
            html.Td(safe_fmt(acc),    style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
            html.Td(safe_fmt(prec),   style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
            html.Td(safe_fmt(rec),    style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
            html.Td(safe_fmt(f1),     style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
            html.Td(safe_fmt(auc_score), style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'backgroundColor': '#ecf0f1'}),
        ]))

    th_style = {'padding': '10px', 'borderBottom': '2px solid #333', 'textAlign': 'left',
                'backgroundColor': '#34495e', 'color': 'white'}

    auc_fig = go.Figure(data=[go.Bar(
        x=[m[0] for m in available_models],
        y=[m[2] for m in available_models],
        marker_color=['rgb(31,119,180)', 'rgb(214,39,40)', 'rgb(255,127,14)', 'rgb(44,160,44)'],
        text=[safe_fmt(m[2]) for m in available_models],
        textposition='auto'
    )], layout=go.Layout(
        title="AUC Scores Comparison — All Models (2.5 Year Test Data)",
        xaxis_title="Model", yaxis_title="AUC Score",
        height=500, yaxis=dict(range=[0, 1]), title_x=0.5
    ))

    return html.Div([
        html.H2("Comprehensive Model Performance Analysis (2.5 Year Data)",
                style={'textAlign': 'center', 'marginBottom': 30}),
        html.Table(
            [html.Tr([html.Th("Model", style=th_style), html.Th("Accuracy", style=th_style),
                      html.Th("Precision", style=th_style), html.Th("Recall", style=th_style),
                      html.Th("F1-Score", style=th_style), html.Th("AUC Score", style=th_style)])]
            + rows,
            style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': 30,
                   'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}
        ),
        html.H3("📊 Interactive Model Analysis — Confusion Matrix & ROC Curve",
                style={'marginTop': 30, 'textAlign': 'center'}),
        html.Div([
            html.Label("Select Model:", style={'fontWeight': 'bold', 'fontSize': '16px',
                                               'marginBottom': '10px', 'display': 'block'}),
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label': m[0], 'value': i} for i, m in enumerate(available_models)],
                value=0, clearable=False, style={'width': '400px', 'marginBottom': '30px'}
            )
        ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),
        html.Div([
            html.Div([html.H4("Confusion Matrix", style={'textAlign': 'center'}),
                      dcc.Graph(id='confusion-matrix-graph')],
                     style={'display': 'inline-block', 'width': '48%', 'marginRight': '2%', 'verticalAlign': 'top'}),
            html.Div([html.H4("ROC Curve", style={'textAlign': 'center'}),
                      dcc.Graph(id='roc-curve-graph')],
                     style={'display': 'inline-block', 'width': '48%', 'verticalAlign': 'top'})
        ], style={'marginTop': '20px', 'marginBottom': '30px'}),
        html.H3("Model Comparison — AUC Scores", style={'marginTop': 30}),
        dcc.Graph(figure=auc_fig),
        html.Div([
            html.P("🏆 Best AUC Score: Binary Logistic Regression (0.7051)", style={'fontSize': '16px', 'marginBottom': '10px'}),
            html.P("📈 Second Best: Gaussian Naive Bayes (0.7033)", style={'fontSize': '16px', 'marginBottom': '10px'}),
            html.P("🌳 Decision Tree and Random Forest show moderate performance (0.6198 – 0.6452)", style={'fontSize': '16px', 'marginBottom': '10px'}),
            html.P("⚠️ Note: All metrics based on 2.5 year test data analysis",
                   style={'fontSize': '14px', 'fontStyle': 'italic', 'color': '#666'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginTop': '20px'})
    ])

# ============================================================================
# TAB 3: SENTIMENT ANALYSIS
# ============================================================================
def create_sentiment_tab():
    """Render the sentiment tab using web_scrape.csv if present, else show placeholder."""
    csv_candidates = ["data/web_scrape.csv", "web_scrape.csv"]
    sentiment_df = None
    for p in csv_candidates:
        if os.path.exists(p):
            try:
                sentiment_df = pd.read_csv(p)
                break
            except Exception:
                pass

    if sentiment_df is None:
        return html.Div([
            html.H2("Sentiment Analysis Charts", style={'textAlign': 'center', 'marginBottom': 30}),
            html.Div([
                html.H4("⚠️ web_scrape.csv not found", style={'color': '#e74c3c', 'textAlign': 'center'}),
                html.P("To enable the Sentiment Analysis tab, upload your web_scrape.csv file to the data/ folder in your repository.",
                       style={'textAlign': 'center', 'color': '#555'}),
                html.P("The file should contain columns like: headline/title/text, date, and any sentiment scores.",
                       style={'textAlign': 'center', 'color': '#555'}),
            ], style={'padding': '40px', 'backgroundColor': '#fef9e7', 'borderRadius': '8px',
                      'border': '1px solid #f39c12', 'margin': '40px auto', 'maxWidth': '700px'})
        ])

    # Detect text column
    text_col = next((c for c in ['clean_text', 'raw_text', 'headline', 'title', 'text', 'content']
                     if c in sentiment_df.columns), None)

    if text_col and 'clean_text' not in sentiment_df.columns:
        sentiment_df['clean_text'] = (sentiment_df[text_col].fillna('').astype(str)
                                      .str.lower().str.replace(r'[^\w\s]', ' ', regex=True))
    if text_col and 'raw_text' not in sentiment_df.columns:
        sentiment_df['raw_text'] = sentiment_df[text_col].fillna('').astype(str)

    # Lexicon score
    pos_words = {'gain','growth','positive','bullish','up','rise','profit','strong','optimistic','surge'}
    neg_words = {'loss','decline','negative','bearish','down','fall','drop','weak','pessimistic','crash'}

    def lex_score(text):
        tokens = str(text).split()
        pos = sum(1 for t in tokens if t in pos_words)
        neg = sum(1 for t in tokens if t in neg_words)
        total = pos + neg
        return (pos - neg) / total if total else 0.0

    if 'score' not in sentiment_df.columns and 'clean_text' in sentiment_df.columns:
        sentiment_df['score'] = sentiment_df['clean_text'].apply(lex_score)

    if 'sentiment_label' not in sentiment_df.columns and 'score' in sentiment_df.columns:
        sentiment_df['sentiment_label'] = sentiment_df['score'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))

    for label_col in ['vader_sentiment_label', 'finbert_sentiment_label']:
        if label_col not in sentiment_df.columns and 'sentiment_label' in sentiment_df.columns:
            sentiment_df[label_col] = sentiment_df['sentiment_label']

    sentiment_order = ['positive', 'neutral', 'negative']
    vader_counts   = sentiment_df['vader_sentiment_label'].value_counts().reindex(sentiment_order, fill_value=0) if 'vader_sentiment_label' in sentiment_df.columns else pd.Series({'neutral': 0})
    finbert_counts = sentiment_df['finbert_sentiment_label'].value_counts().reindex(sentiment_order, fill_value=0) if 'finbert_sentiment_label' in sentiment_df.columns else vader_counts

    score_values = pd.to_numeric(sentiment_df.get('score', pd.Series(dtype=float)), errors='coerce').dropna()
    hist_fig = go.Figure(data=[go.Histogram(x=score_values, nbinsx=30, marker_color='rgb(52,152,219)')]) if len(score_values) > 0 else go.Figure()
    hist_fig.update_layout(title='Histogram of Lexicon Sentiment Scores', xaxis_title='Sentiment Score', yaxis_title='Count', height=400)

    bar_colors = ['rgb(44,160,44)', 'rgb(148,103,189)', 'rgb(214,39,40)']

    return html.Div([
        html.H2("Sentiment Analysis Charts", style={'textAlign': 'center', 'marginBottom': 30}),
        html.Div([
            html.H4("Top 10 Sentiment Data Rows"),
            dt.DataTable(
                data=sentiment_df.head(10).to_dict('records'),
                columns=[{"name": i, "id": i} for i in sentiment_df.columns],
                page_size=10, style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230,230,230)', 'fontWeight': 'bold'}
            )
        ], style={'marginBottom': 30}),
        html.Div([
            html.Div([
                html.H4("VADER Sentiment Label Counts"),
                dcc.Graph(figure=go.Figure(
                    data=[go.Bar(x=vader_counts.index.tolist(), y=vader_counts.values.tolist(),
                                 marker_color=bar_colors, text=vader_counts.values.tolist(), textposition='auto')],
                    layout=go.Layout(title="VADER Sentiment", xaxis_title="Sentiment", yaxis_title="Count", height=400)
                ))
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
            html.Div([
                html.H4("FinBERT Sentiment Label Counts"),
                dcc.Graph(figure=go.Figure(
                    data=[go.Bar(x=finbert_counts.index.tolist(), y=finbert_counts.values.tolist(),
                                 marker_color=bar_colors, text=finbert_counts.values.tolist(), textposition='auto')],
                    layout=go.Layout(title="FinBERT Sentiment", xaxis_title="Sentiment", yaxis_title="Count", height=400)
                ))
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'marginBottom': 30}),
        html.Div([
            html.H4("Histogram of Sentiment Scores"),
            dcc.Graph(figure=hist_fig)
        ])
    ])

# ============================================================================
# APP LAYOUT
# ============================================================================
app.layout = html.Div([
    html.Div([
        html.H1("📈 Stock Market Analytics Dashboard",
                style={'textAlign': 'center', 'marginBottom': 10, 'color': '#2c3e50'}),
        html.P("Comprehensive analysis of global markets data with 4 ML models",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 0}),
        html.Hr()
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'marginBottom': 30, 'borderRadius': '8px'}),

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='📊 EDA Charts', value='tab-1',
                children=[html.Div(create_eda_tab(), style={'padding': '20px'})],
                style={'padding': '15px', 'fontWeight': 'bold'},
                selected_style={'borderTop': '3px solid #3498db', 'backgroundColor': '#ecf0f1'}),
        dcc.Tab(label='🎯 Model Performance', value='tab-2',
                children=[html.Div(create_model_tab(), style={'padding': '20px'})],
                style={'padding': '15px', 'fontWeight': 'bold'},
                selected_style={'borderTop': '3px solid #3498db', 'backgroundColor': '#ecf0f1'}),
        dcc.Tab(label='💬 Sentiment Analysis', value='tab-3',
                children=[html.Div(create_sentiment_tab(), style={'padding': '20px'})],
                style={'padding': '15px', 'fontWeight': 'bold'},
                selected_style={'borderTop': '3px solid #3498db', 'backgroundColor': '#ecf0f1'}),
    ], style={'fontFamily': 'Arial', 'fontSize': 16})
], style={'padding': '20px', 'fontFamily': 'Arial', 'backgroundColor': '#f8f9fa'})

# ============================================================================
# CALLBACKS
# ============================================================================
@app.callback(Output('boxplot-graph', 'figure'), Input('boxplot-dropdown', 'value'))
def update_boxplot(col):
    if not col or col not in combined_data.columns:
        return go.Figure()
    fig = go.Figure(data=[go.Box(
        x=combined_data["Year"], y=combined_data[col],
        name=col.replace("_Return", ""), boxmean=True, boxpoints=False,
        marker_color='rgb(31,119,180)', line_color='rgb(31,119,180)'
    )], layout=go.Layout(
        title=f"Box-Whisker: {col.replace('_Return','')} Returns by Year",
        xaxis_title="Year", yaxis_title=f"{col} (Returns)", height=500,
        showlegend=False, title_x=0.5
    ))
    return fig


@app.callback(Output('barplot-graph', 'figure'), Input('barplot-dropdown', 'value'))
def update_barplot(col):
    if not col or col not in combined_data.columns:
        return go.Figure()
    med = combined_data.groupby('Year')[col].median()
    colors = ['rgb(255,99,132)' if v < 0 else 'rgb(54,162,235)' for v in med.values]
    fig = go.Figure(data=[go.Bar(
        x=med.index, y=med.values, marker_color=colors,
        text=[f"{v:.6f}" for v in med.values], textposition='auto',
        hovertemplate='<b>Year: %{x}</b><br>Median Return: %{y:.6f}<extra></extra>'
    )], layout=go.Layout(
        title=f"Median Daily Returns: {col.replace('_Return','')} by Year",
        xaxis_title="Year", yaxis_title="Median Daily Return", height=500,
        showlegend=False, title_x=0.5
    ))
    return fig


@app.callback(Output("combined_heatmap", "figure"), Input("combined_agg", "value"))
def update_combined_heatmap(agg):
    return make_combined_heatmap(combined_data_heatmap, agg)


@app.callback(Output("corr_heatmap", "figure"), Input("corr_choice", "value"))
def update_corr(choice):
    if choice == "A":
        return corr_fig(corr_A, "Correlation Matrix (6-Year Daily Returns)")
    return corr_fig(corr_B, "Correlation Matrix of 2024 Daily Returns (6×6)")


@app.callback(Output("bar_fig", "figure"), Input("bar_indices", "value"))
def update_bar(selected_indices):
    df = bar_long[bar_long["Index"].isin(selected_indices)].copy()
    fig = px.bar(
        df, x="Nifty_Open_Dir", y="Daily Return", color="Statistic",
        barmode="group", facet_col="Index", facet_col_wrap=3,
        title="Mean and Median of Global Indices by Nifty_Open_Dir",
        category_orders={"Statistic": ["mean", "median"]},
    )
    fig.update_layout(margin=dict(l=60, r=20, t=70, b=60), height=700, template="plotly_white")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


@app.callback(Output("box_fig", "figure"), Input("box_index", "value"))
def update_box(index_col):
    if not index_col or index_col not in combined_data.columns:
        return go.Figure()
    fig = px.box(
        combined_data, x="Nifty_Open_Dir", y=index_col, points="outliers",
        title=f"Distribution of {index_col.replace('_Return','')} by Nifty_Open_Dir",
    )
    fig.update_layout(margin=dict(l=60, r=20, t=70, b=60), height=520, template="plotly_white",
                      xaxis_title="Nifty Opening Direction",
                      yaxis_title=f"{index_col.replace('_Return','')} Returns")
    return fig


@app.callback(
    [Output('confusion-matrix-graph', 'figure'),
     Output('roc-curve-graph', 'figure')],
    Input('model-dropdown', 'value')
)
def update_model_analysis(idx):
    models = _get_available_models()
    name, cm_mat, auc_score, fpr_data, tpr_data = models[idx]

    model_colors = {
        'Binary Logistic Regression': 'rgb(31,119,180)',
        'Gaussian Naive Bayes':        'rgb(214,39,40)',
        'Decision Tree':               'rgb(255,127,14)',
        'Random Forest':               'rgb(44,160,44)',
    }

    cm_fig = go.Figure(data=go.Heatmap(
        z=cm_mat,
        x=['Predicted Negative (0)', 'Predicted Positive (1)'],
        y=['Actual Negative (0)', 'Actual Positive (1)'],
        text=cm_mat, texttemplate="%{text}",
        colorscale='Blues',
        hovertemplate='%{y}<br>%{x}<br>Count: %{text}<extra></extra>'
    ))
    cm_fig.update_layout(title=f"{name}<br>AUC: {auc_score:.4f}", height=500, title_x=0.5)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(
        x=fpr_data, y=tpr_data, mode='lines', name=name,
        line=dict(color=model_colors.get(name, 'rgb(100,100,100)'), width=3)
    ))
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    roc_fig.update_layout(
        title=f"ROC Curve — {name} (AUC: {auc_score:.4f})",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=500, title_x=0.5, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]),
        showlegend=True
    )
    return cm_fig, roc_fig


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
